# Configurable LMCache routing source

The regular HTTP Router can use LMCache's `POST /lookup` and `POST /health`
instead of the native `/routing/*` collector. All worker/controller URLs and
instance IDs come from `--routing-state-config`; none is built into the adapter.
The three policies still share selection, compatibility, the dispatch ledger,
retry/stream cleanup and session affinity.

```json
{
  "lmcache": {
    "renderer_base_url": "https://renderer.example",
    "model": "local",
    "fingerprint": null,
    "workers": {
      "https://worker-one.example": {
        "controller_url": "https://controller-one.example",
        "instance_id": "worker-one",
        "block_size": 16,
        "fingerprint": null
      },
      "https://worker-two.example": {
        "controller_url": "https://controller-two.example",
        "instance_id": "worker-two",
        "block_size": 16,
        "fingerprint": null
      }
    }
  },
  "telemetry_timeout_ms": 2000,
  "render_timeout_ms": 5000,
  "max_evidence_age_ms": 3000
}
```

Replace every URL/ID and block size with the deployment's values. `block_size`
is the pinned native vLLM block size, not a guessed LMCache chunk size. The
renderer and each worker have separate serving fingerprints: populate them only
after verifying the actual model/tokenizer/serving configuration. Equal model
aliases or tokens do not establish equal fingerprints. Null fingerprints allow
data-path development and cause `missing_serving_fingerprint` fallback.

```sh
vllm-router --worker-urls "$WORKER_ONE" "$WORKER_TWO" \
  --policy kv_batch_ect --routing-state-config routing.json \
  --health-check-endpoint /v1/models
```

The adapter currently supports the native text Chat renderer, including tool
history and chat template arguments. It forwards the original JSON to both
render and inference, without injected routing tokens. Responses, completions,
multimodal content, LoRA and cache-salted requests remain unsupported on this
source and explicitly fall back. The native bridge retains its existing
Chat/Completions support. Credentials use the existing `header_env` configuration.

The live c/d entry points route `/health` to the Controller: GET returns 405
while POST accepts an instance ID. The example therefore uses vLLM's GET
`/v1/models` for inference health, including startup and worker additions.
Controller health is checked separately by the adapter.

## Per-attempt lifecycle

1. Render the exact request once with `/v1/chat/completions/render` on the
   configured renderer, outside the dispatch lock.
2. Before each attempt, query each available worker's configured controller.
   Lookup receives the rendered `tokens`; health receives its `instance_id`.
   Calls run concurrently, with at most eight worker observations active.
3. Use only that instance's lookup entry. Other entries do not stand in for
   workers whose controller is unavailable. An empty match is accepted only
   after the controller reports the configured instance's sole rank `0` healthy.
4. Build the common snapshot under the ledger lock, re-reading in-flight counts
   and observation ages; choose and reserve atomically.
5. Stream the unchanged inference request/response through the existing lifecycle.
   Retry performs a new lookup, rather than reusing the previous attempt's cache.

No background native KV collector starts in LMCache mode. Lookup/health traffic
uses its own HTTP client, outside the critical section. This temporary lookup
source adds request preprocessing latency; it does not supply a local native
block-event index. Observations are scoped to this request, never reused across
different prompts. The age includes lookup round trip and time waiting for
other workers; it does not claim knowledge of the controller's internal lag.

## Evidence and remaining deployment requirements

`LmCacheObserved` records `instance_id`, `location`, `cached_tokens`, usable
`tokens`, optional `engine_epoch`, and age. The current supported tier is
`LocalCPUBackend`. The usable prefix is capped at complete native blocks before
the final prompt token; restoration accounting retains the original stored
length. These values do not claim GPU residency or include an unknown GPU hit.
The baselines rank restorable CPU prefix when this source is trusted; native
GPU-prefix baselines and controller-prefix baselines must be labeled separately
in experimental results.

To establish engine identity, lookup and health must both return matching
`x-routing-worker-id` and `x-routing-engine-epoch` headers, bound to the actual
inference engine. The inference response must agree. These are identity
extensions of the deployment entry point, **not built-in LMCache headers**.
Current c/d replies lack them; the adapter preserves `engine_epoch: null` and
uses `unverified_lmcache_identity` fallback when fingerprints are supplied.
It never turns operation IDs, metric timestamps or configured instance names
into restart epochs. A mismatch latches invalidation until a new verified epoch.
Out-of-order replies cannot replace newer identity observations. Controller
HTTP observations alone never release unknown inference attempts from the
ledger; termination still needs an inference terminal event or independently
verified lifecycle reconciliation.

HTTP/schema/health failures, unsupported tiers, overlong/non-block prefixes,
missing worker configurations, or stale observations become unknown/stale for
the common selector. No candidate is removed just because its controller failed.
Known incompatible fingerprints are excluded even when other metadata is absent.
The collector counter `router_lmcache_observations_total` reports the acquisition
outcome, while `router_routing_decisions_total` reports the actual fallback.
An `observed` acquisition alone does not mean the policy trusted that evidence.

## ECT restoration model

Existing `cost_models` remain indexed by worker/instance ID. A CPU hit additionally
requires a measured `restore_models` entry for that worker, with these fields:

- `fingerprint`, `calibration_version`, and `location: "LocalCPUBackend"`;
- `token_range: [minimum, maximum]` for stored tokens being restored;
- nonnegative finite `fixed_ms` and `per_token_ms`.

The initial tier extension is:

```text
R = fixed_ms + per_token_ms * cached_tokens
E = (P(L, H_restorable) + R + D(L, O_prior)) * (1 + beta * n) + Q
```

`CostEstimate` exposes `restore_ms` and its calibration version. This model
assumes restoration belongs inside the service/load multiplier; it must be
calibrated with that measurement boundary. It conservatively models restoration
of the reported CPU prefix and does not infer simultaneous GPU residency.
If a later source provides both GPU and CPU evidence, overlapping prefixes must
be combined with `max`, and only the additional restored portion charged.

No restore coefficients are supplied by default. Missing, incompatible,
non-finite or out-of-range restore models cause whole-decision least-load
fallback; a proven zero CPU match needs no restore model. Do not use synthetic
test coefficients in production. The [input analysis](ect-observable-inputs.md)
explains what existing metrics can measure and why the current c/d warm repeats
did not provide CPU restoration samples.

## Verification

`bash scripts/routing/smoke.sh` runs all three policies against native fixtures
and LMCache fixtures through the real Router binary. LMCache tests verify exact
lookup token forwarding, per-controller instance mapping, no `/routing/*`
calls, fresh lookups on retries, stream-held reservations, and controller-health
failure fallback. Snapshot tests verify missing identity and restoration costs,
including a case where accounting for restoration changes the ECT winner.

The live [c/d observations](results/lmcache-cd-observations-2026-09-12.json)
demonstrate controller scope and token parity; they do not establish trusted
engine identity, GPU cache migration or a calibrated ECT benchmark.

The [live Router smoke](results/router-lmcache-cd-smoke-2026-09-12.json) then ran
all three CLI policies against c/d, two concurrent requests per policy. All six
returned 200; each policy dispatched one request to each endpoint, acquired
observations from both controllers for both attempts, and ended with zero
in-flight requests. Fingerprints were explicitly null, so all six decisions
reported `missing_serving_fingerprint` fallback. This validates the real data
path and common lifecycle, not trusted ECT selection or calibrated latency.
