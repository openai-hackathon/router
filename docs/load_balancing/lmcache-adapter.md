# Configurable LMCache routing source

The regular HTTP Router can use LMCache's `POST /lookup` and `POST /health`
instead of the native `/routing/*` collector. All worker/controller URLs and
instance IDs come from `--routing-state-config`; none is built into the adapter.
The three policies still share selection, compatibility, the dispatch ledger,
retry/stream cleanup and session affinity.

```json
{
  "lmcache": {
    "identity_mode": "endpoint",
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
is the pinned native vLLM block size, not a guessed LMCache chunk size.

`identity_mode: "endpoint"` enables the current experiment's explicit assumption:
all workers use the same model/tokenizer/serving configuration, and each configured
inference endpoint maps to one fixed engine and controller instance. The selector
uses that mapping without requiring fingerprints or engine identity headers.
Fingerprints can remain null; missing verification no longer forces fallback.
This is an operator assumption, not measured engine identity. Decisions and their
metric counter include `identity_mode="endpoint"`; evidence keeps unknown epochs
null. Successful session home is scoped by model/session and records the endpoint.
If endpoints are redeployed or repointed during an experiment, restart the Router
to reset session state; this mode cannot detect an engine replacement behind a URL.

Omitting the field selects `identity_mode: "verified"`. In that mode, populate
the renderer and worker fingerprints from their verified serving configurations;
null fingerprints cause `missing_serving_fingerprint` fallback. Equal aliases or
tokens alone do not verify the configurations. Unknown mode names are rejected.

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
The raw CPU match may end inside a native block. The adapter preserves that
length as `cached_tokens` and computes `tokens` as
`floor(min(cached_tokens, L - 1) / block_size) * block_size`.
For example, a 1043-token full CPU match remains 1043 for restoration accounting
and supplies 1040 routing tokens with block size 16. Repeated lookups refresh
both values, including increases and decreases.
The baselines rank restorable CPU prefix when this source is trusted; native
GPU-prefix baselines and controller-prefix baselines must be labeled separately
in experimental results.

In `verified` mode, to establish engine identity, lookup and health must both return matching
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
verified lifecycle reconciliation. Endpoint mode does not assign an invented
epoch to attempts or use controller identity changes to release them.

HTTP/schema/health failures, unsupported tiers, overlong/non-integer prefixes,
missing worker configurations, or stale observations become unknown/stale for
the common selector. No candidate is removed just because its controller failed.
In verified mode, known incompatible fingerprints are excluded even when other
metadata is absent. Endpoint mode retains model-name, configured-instance,
block-size, tier, health and evidence-age checks, while assuming serving compatibility.
The collector counter `router_lmcache_observations_total` reports the acquisition
outcome, while `router_routing_decisions_total` reports the actual fallback.
An `observed` acquisition alone does not mean the policy trusted that evidence.

## ECT restoration model

This section describes the default `ect_model: "decomposed"` path. The new
[`completion_time` mode](completion-time-routing.md) learns net cache benefit
from per-attempt completion times and does not require a separate restore model.
It uses the same raw CPU lookup evidence plus independently collected backend
running/waiting and optional GPU KV pressure.

Existing `cost_models` remain indexed by worker/instance ID. A CPU hit additionally
requires a measured `restore_models` entry for that worker, with these fields:

- `fingerprint`, `calibration_version`, and `location: "LocalCPUBackend"`;
- `token_range: [minimum, maximum]` for stored tokens being restored;
- nonnegative finite `fixed_ms` and `per_token_ms`.

Endpoint mode assigns both models by the configured instance ID and skips serving
fingerprint comparison. Calibration version, ranges, finite/nonnegative costs and
restore tier still must be valid. Each worker keeps its own coefficients; assuming
the same serving configuration does not assume identical hardware performance.

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
did not provide CPU restoration samples. Later [d samples](results/router-d-partial-prefix-smoke-2026-09-12.json)
include external hits, but do not yet isolate a restoration-time curve; see the
[revised policy proposal](cache-aware-ect-design.md).

## Verification

`bash scripts/routing/smoke.sh` runs all three policies against native fixtures
and LMCache fixtures in both identity modes through the real Router binary. LMCache tests verify exact
lookup token forwarding, per-controller instance mapping, no `/routing/*`
calls, fresh lookups on retries, stream-held reservations, and controller-health
failure fallback. Snapshot tests verify missing identity and restoration costs,
including a case where accounting for restoration changes the ECT winner.
Endpoint tests omit all fingerprints and routing identity headers, exercise all
three rankings and bounded affinity, and verify that unknown/stale evidence and
missing restoration costs still cause the common fallback. Session lifecycle
tests require successful completion to update home and preserve unknown attempts.

The live [c/d observations](results/lmcache-cd-observations-2026-09-12.json)
demonstrate controller scope and token parity; they do not establish trusted
engine identity, GPU cache migration or a calibrated ECT benchmark.

The earlier [live Router smoke](results/router-lmcache-cd-smoke-2026-09-12.json) ran
all three CLI policies against c/d, two concurrent requests per policy. All six
returned 200; each policy dispatched one request to each endpoint, acquired
observations from both controllers for both attempts, and ended with zero
in-flight requests. Fingerprints were explicitly null, so all six decisions
reported `missing_serving_fingerprint` fallback. This validates the real data
path and common lifecycle, not trusted ECT selection or calibrated latency.

The subsequent [endpoint-mode live attempt](results/router-lmcache-endpoint-smoke-2026-09-12.json)
could not reach policy dispatch: c's Chat renderer timed out after the configured
120-second HTTP timeout. Consequently it does not establish live cache ranking
in endpoint mode. A subsequent 20-second availability check found d's lookup
responding 200 and c's lookup timing out. That check used one synthetic token,
so its empty layout is not evidence that d's cache was empty.
Local verification passed 505 Rust unit tests, 14 snapshot
acceptance tests and 26 Python tests, including all three policies through the
real binary with no fingerprints or routing identity headers.

After accepting partial CPU prefixes, the [d-only Router smoke](results/router-d-partial-prefix-smoke-2026-09-12.json)
sent six successful requests through the real binary. Both baselines acquired
partial-prefix evidence without fallback; ECT acquired the same evidence and
reported `missing_cost_model`. These single-worker runs validate adapter
integration, not comparative routing performance. The correction passed three
focused adapter tests, 15 snapshot tests, 26 Python tests and Clippy.
The user reported a concurrent benchmark on d, so these timing samples are
not unloaded calibration data.
