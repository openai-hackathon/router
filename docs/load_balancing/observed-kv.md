# Observed KV routing

`prefix_max`, `least_load_kv`, and `kv_batch_ect` run in the regular HTTP Router.
They share request features, an event-derived KV index, compatibility checks, a
per-attempt dispatch ledger and request lifecycle. They do not change vLLM's
scheduler. Initial support is one identifiable DP engine and one full-attention
GPU cache group per URL, with text-only Chat/Completions requests. CLI validation
rejects these policies in PD and IGW modes.

## What the live deployment provides, and what ECT still needs

Probed on 2026-09-12, using the URLs supplied in the conversation:

| Deployment | URL | First probe | Subsequent inference | Model |
| --- | --- | --- | --- | --- |
| a | `https://s9930703--vllm-serve-serve.modal.run` | 200, 1.76 s | 1.28–1.31 s | Qwen/Qwen3-0.6B |
| b | `https://s9930703--vllm-serve-b-serve.modal.run` | 200, 4.16 s | 1.64–1.82 s | Qwen/Qwen3-0.6B |

These are individual small requests, **not a latency benchmark**. Both report
vLLM 0.29.0. The supplied `a` URL repeats the original URL; only two distinct URLs
were available for this verification. The third URL and actual singleton Modal
configuration have not been verified. Different URLs alone do not establish
engine identity or replica count.

| Input | What we obtained | Remaining work |
| --- | --- | --- |
| Prompt tokens, `L` | `/v1/chat/completions/render`; exact parity with inference `return_token_ids` on both URLs: 17 tokens for plain chat, 193 for tool history | Expand golden cases for production traffic; implement and verify a Responses adapter |
| Request-specific reusable prefix, `H_j` | Not available through the public interfaces checked | Collect `BlockStored`, `BlockRemoved`, `AllBlocksCleared`, including parent hashes, token IDs, block size, group and extra keys |
| Event integrity | No HTTP event/snapshot API in deployment OpenAPI | Supply sequence/replay and a complete snapshot or a known-empty engine origin; fail closed on gaps |
| Worker identity | Metrics label `engine="0"` | A stable worker ID plus restart epoch; bind telemetry and inference to the same supervised engine |
| Serving compatibility | `/version`, `/v1/models`: vLLM version, model root, served alias; a reports max context 4096 | Pin model/tokenizer revisions, tokenizer/template digest, block size, hash algorithm, cache groups, dtype, TP/DP and serving settings in a deployment fingerprint |
| Running/waiting/cache occupancy | `/metrics` exposes all three; `/load` returns `server_load` | Use to audit the Router ledger; do not add these values to ledger in-flight counts |
| Per-request service times | Current inference `metrics` is null; `/metrics` has aggregate histograms | Enable `--enable-per-request-metrics` and collect timing samples by `L`, actual cached tokens, output tokens and Router concurrency |
| Actual cache use during execution | Current `usage.prompt_tokens_details` is null; cumulative cache hits exist | Enable the deployed version's prompt-token-details reporting or another attributable measurement for cache-hit validation; cumulative hits do not predict cache locality before dispatch |
| Output prior and ECT coefficients | No calibration supplied | Fit in Router tooling from measured requests; vLLM does not need to return predicted output length, beta, or queue correction |

`/routing/*` are **new bridge APIs in this repository**, not built-in vLLM APIs.
We also checked `/server_info`, `/get_server_info`, `/get_model_info` and
`/tokenizer_info` on a; all returned 404. The actual load API is `/load`, not
`/get_load`. We did not inspect the Modal container's launch flags, local ZMQ
sockets, GPU identity, or deployment configuration. Absence of a public HTTP KV
API does not prove the publisher is disabled inside the container.

The tools fixture uses `tool_choice: "none"` while retaining tool definitions,
assistant calls and tool results. The default auto tool choice was rejected by
the existing deployment. Golden fixtures are synthetic and live under
`tests/fixtures/qwen3_0_6b_vllm_029_tokens.json`. No Responses token parity is claimed.

Reproduce the probes (two short inference calls per URL with `--token-parity`):

```sh
python scripts/routing/probe.py --token-parity \
  --url https://s9930703--vllm-serve-serve.modal.run \
  --url https://s9930703--vllm-serve-b-serve.modal.run
```

## Ranking and fallback

- Prefix Max: highest observed reusable tokens, then lowest in-flight, then URL order.
- Least Load KV: lowest in-flight, then highest reusable tokens, then URL order.
- KV Batch ECT: lowest measured-model ECT, then URL order. Only this policy may
  use session affinity: the successful home must still have the same epoch, be
  within the session TTL, have at least one more reusable block, and satisfy
  `E_home <= 1.015 * E_best + 10 ms`.

All candidates must have usable KV evidence to compare locality. Missing,
stale, unsupported or interrupted evidence causes a common least-load fallback.
Missing/invalid/out-of-range cost predictions also cause common least-load
fallback, preserving candidates. Known incompatible serving fingerprints/models
are excluded. A missing renderer produces an explicit fallback; a new policy
name alone does not mean KV routing is active.

The index matches exact token blocks and follows the opaque hashes/parent links
observed from vLLM. It does not recreate Python pickle/CBOR hashes in Rust.
Consequently, the adapter only indexes unsalted text blocks without LoRA or
extra keys, and refuses unrecognized cache layouts. Prefixes stop at the first
missing block. vLLM must compute the final prompt token for logits, so an exact
full-block-length prompt cannot reuse its final block. Events are observations,
not GPU block reservations; blocks can be evicted between routing and execution.

Inference JSON is preserved for regular Chat/Completions/Responses traffic.
Chat/Completions retain input validation without reserializing typed defaults
into the forwarded request. Unsupported Responses features use least load.
Typed APIs and transparent inference endpoints enter the same selector.

## Per-attempt lifecycle

The selection and load increment share a lock. Rendering and HTTP calls happen
outside that lock. Every policy, including P2 and Cache Aware, uses the ledger.
P2's old regular-router `/get_load` polling is removed; its regular HTTP load is
now immediately updated by reservations.

Unsent reservations and complete HTTP rejection responses release once. Normal
non-streaming completion releases after body consumption. SSE completion is
recognized across HTTP chunk boundaries, including Responses terminal events.
A 200 header is not completion. On transport ambiguity, truncated SSE or client
disconnection, the reservation becomes Unknown and stays counted until verified
engine replacement. There is no guessed timeout that silently treats it as zero.
Unknown attempts without a verified epoch cannot currently be reconciled
automatically. Background Responses also retain their reservation unless a
terminal event is observed; background polling/cancellation reconciliation is
not yet implemented. Use foreground requests for these experiments.

Metrics include `router_routing_decisions_total{policy,fallback}` and dispatch
completion/unknown counters. Decision logs contain candidate `H`, `n`, ECT,
evidence status/age/sequence and selected worker, never prompt text/token arrays.
Backend running/waiting/occupancy are separate gauges. Report fallback fraction
alongside benchmark results; fallback runs are not valid KV-policy comparisons.

## Bridge deployment

`scripts/routing/bridge.py` starts and supervises a fresh vLLM process, connects
to loopback ZMQ before accepting inference, and maintains bounded event history
and an event-derived snapshot. It has no worker selection logic. HTTP inference
streams through it and gets `x-routing-worker-id` and `x-routing-engine-epoch`.
The epoch is unique for each supervised process lifetime. Restart the bridge
and vLLM together; independent internal engine replacement is not supported.
Do not attach this bridge to an already populated, unmanaged engine.

Expose the bridge's port 8001 through one fixed Modal service per replica, with
singleton placement. The vLLM HTTP port and both ZMQ ports remain loopback-only.
The bridge requires `x-routing-token` from `ROUTING_BRIDGE_TOKEN` on every request,
including health. Modal proxy authentication can be layered on top. Use the
same vLLM Python environment; dependencies are FastAPI, httpx, msgspec, pyzmq,
prometheus-client and uvicorn. This adapter targets **vLLM 0.29.0**.

Create a deployment manifest like this, replacing all revision/digest values
with those of the artifacts actually launched. The canonical `serving` JSON is
hashed into the compatibility fingerprint; the bridge validates version, model
root and a single metrics engine, but the operator must ensure the manifest
matches the command and tokenizer artifacts.

```json
{
  "worker_id": "modal-g0",
  "serving": {
    "vllm_version": "0.29.0",
    "model": "local",
    "model_root": "Qwen/Qwen3-0.6B",
    "model_revision": "PINNED_REVISION",
    "tokenizer_revision": "PINNED_REVISION",
    "tokenizer_sha256": "ACTUAL_DIGEST",
    "chat_template_sha256": "ACTUAL_DIGEST",
    "block_size": 16,
    "cache_groups": 1,
    "attention": "full",
    "dp_size": 1,
    "tp_size": 1,
    "dtype": "bfloat16",
    "kv_cache_dtype": "auto",
    "hash_algorithm": "sha256_cbor",
    "max_model_len": 4096,
    "max_num_seqs": 256,
    "max_num_batched_tokens": 4096,
    "scheduling_policy": "fcfs"
  }
}
```

Example launch, after ensuring command/manifest agreement:

```sh
python scripts/routing/bridge.py --config worker-g0.json -- \
  vllm serve Qwen/Qwen3-0.6B --served-model-name local \
  --host 127.0.0.1 --port 8000 --max-model-len 4096 \
  --block-size 16 --dtype bfloat16 --max-num-seqs 256 \
  --max-num-batched-tokens 4096 --enable-prefix-caching \
  --prefix-caching-hash-algo sha256_cbor --enable-per-request-metrics \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://127.0.0.1:5557","replay_endpoint":"tcp://127.0.0.1:5558","topic":"kv-events","buffer_steps":10000}'
```

Pin model and tokenizer revisions in the actual launch command. Do not enable
`--disable-log-stats` with per-request metrics. Streaming calibration requests
need `stream_options.include_usage: true` to receive the final metrics chunk.
The bridge does not enable dev-mode control endpoints or change scheduler code.

Bridge contract:

- `GET /routing/info`: schema 1, worker ID, epoch, model, fingerprint, block size, support status.
- `GET /routing/state`: readiness, optional sampled metrics/age, contiguous sequence, sync state.
- `GET /routing/kv-snapshot`: a complete, consistent inventory derived from events at one sequence.
- `GET /routing/kv-events?epoch=...&after=...`: ordered normalized batches, or 409 when a snapshot is required.
- `POST /routing/render`: `{ "route": "/v1/chat/completions", "body": { ... } }`; returns model renderer token IDs and fingerprint. Unsupported features return 422.

The metric age is the age of the bridge's last successful scrape; it does not
prove that vLLM published new engine metrics at that moment. KV freshness comes
from a live sequence/replay check. Metrics are not used to clear unknown
requests, since the Modal input queue may not be visible to vLLM.

## Router launch and calibration

Create `routing.json` with credentials referenced by environment variable name:

```json
{
  "header_env": { "x-routing-token": "ROUTING_BRIDGE_TOKEN" },
  "poll_interval_ms": 250,
  "telemetry_timeout_ms": 2000,
  "render_timeout_ms": 5000,
  "max_evidence_age_ms": 3000,
  "cost_models": {}
}
```

Optional `renderer_url` points to a trusted `/routing/render` helper. Otherwise
one fresh bridge is used for preprocessing once per request, never all workers.
This first version does not cache rendered requests. Network rendering latency
must be included in client-visible latency measurements. For Modal proxy auth,
add `Modal-Key` and `Modal-Secret` mappings to their respective environment
variable names. Use bridge URLs, not the old direct-vLLM URLs, for KV experiments.

```sh
vllm-router --worker-urls "$G0" "$G1" "$G2" \
  --policy prefix_max --routing-state-config routing.json
# --policy least_load_kv
# --policy kv_batch_ect
```

With `cost_models: {}`, ECT intentionally falls back to least load. Models are
keyed by the bridge's stable worker ID. Each measured model must provide:

```text
fingerprint, calibration_version
prompt_range [min,max], output_range [min,max], concurrency_range [min,max]
output_prior
prefill [a0,a1,a2]
decode [d0,d1,d2]
beta, queue_ms
```

All time coefficients use milliseconds:
`P = a0 + a1*(L-H) + a2*(L²-H²)`;
`D = d0 + d1*O + d2*L*O`;
`E = (P+D)*(1+beta*n) + queue_ms`.
`O` is the measured prior capped by the request's output limit. Coefficients must
be finite/nonnegative and inputs within calibration ranges. These are candidate
features, not an analytic model of vLLM. Calibrate beta and queue correction
jointly to avoid double counting. No simulation coefficients ship as defaults.

For calibration, collect per request: worker/epoch/fingerprint, `L`, observed
prefix and actual cached prompt tokens, pre-dispatch Router `n`, actual output
tokens, scheduled-to-first-token time, generation time, scheduler queue time,
Router first-token and completion timestamps. Separate warm serving runs from
cold starts and transport/platform delays. Fit on a grid of prompt lengths,
cache-hit fractions, output budgets and concurrency; validate on held-out runs.
The current deployment's 4096 context ceiling excludes the 8192-token worked
example from live tests. GPU name/VRAM are useful metadata, not per-request
ranking inputs. Remaining-token prediction and scheduler changes are unnecessary.

## Validation and source contracts

- Rust tests cover atomic reservations, stream completion/disconnection, policy rankings,
  prefix continuity/removal/restart, gaps/replay, invalid costs, fallback and affinity.
- `python -m unittest discover -s scripts/routing -p test_bridge.py` tests normalization.
- `uv run --no-project --with fastapi --with httpx --with msgspec --with pyzmq --with prometheus-client --with uvicorn python scripts/routing/test_bridge_http.py`
  exercises a CPU fake engine through real local HTTP and ZMQ, including replay,
  authentication, rendering, engine headers and unbuffered stream content.
- After `cargo build --bin vllm-router`, run the same Python environment with
  `python -m unittest discover -s scripts/routing -p test_router_e2e.py` to exercise
  the complete path through two bridges and the real Rust Router. Only the worker
  that would lose a deterministic tie is warmed. The test verifies that its four
  cached tokens win selection, the original request survives forwarding, engine
  headers match, and the routing decision reports no fallback.
- Live token fixtures establish parity only for the tested Chat requests. The
  bridge has not been deployed into the supplied Modal containers, and no live
  KV locality or measured ECT benchmark is claimed.

Primary sources for the pinned version:
[vLLM KV event implementation](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/distributed/kv_events.py),
[block event production](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/core/block_pool.py),
[renderer APIs](https://docs.vllm.ai/en/v0.29.0/serving/online_serving/renderer/),
[per-request metrics](https://docs.vllm.ai/en/v0.29.0/features/per_request_metrics/).

## How production-stack obtains cache locality

Inspected production-stack main at commit
`6a33ae48b896361d076a81470dca832ab926cb5d`. Its `KvawareRouter` creates an
`LMCacheControllerManager`, starts its background tasks, tokenizes the prompt,
and submits `LookupMsg(tokens, event_id)`. The returned `layout_info` maps
instance IDs to cache location and matched token position. `QueryInstMsg`
provides the instance-to-endpoint mapping. The example Helm deployment enables
LMCache plus its controller, with controller ports 9000/9001 and a heartbeat
port 9002. The default example also configures a CPU offload buffer.

This path uses **LMCache's own cache inventory updates**, not a subscription by
the production-stack router to vLLM's native KV-events PUB stream. In the current
LMCache source, the CPU storage backend records ADMIT/EVICT operations; workers
send MessagePack messages over ZMQ PUSH to the controller's PULL socket. The
controller maintains a registry and has registration/heartbeat/full-sync paths.
The dependency image and LMCache versions must be pinned together; this source
inspection is not a compatibility test of the example image with vLLM 0.29.0.

LMCache is therefore an alternative to building an inventory collector, if we
choose to add its cache backend to our inference deployment. Its lookup result
must not be treated as a guarantee of GPU-resident KV: CPU/disk/network cache
hits have different restoration costs. ECT would need an explicit cache-tier
and fetch/restore cost model. The current controller lookup also documents
limitations around per-instance prefix continuity and cache locations; audit
these before using it as the exact vector of per-worker `H_j`.

Two additional migration concerns are visible in production-stack's router:
`kvaware` still tokenizes the `prompt` field with a Chat Completions TODO, and
instance mapping is based on endpoint IP extraction. Our tool/Responses request
adapters and Modal HTTPS engine identity cannot be replaced by those assumptions.
Its `loadaware` score is cache-hit fraction minus a normalized in-flight penalty;
it is a different policy from a calibrated expected-completion-time model.

Sources:
[production-stack routing code](https://github.com/vllm-project/production-stack/blob/6a33ae48b896361d076a81470dca832ab926cb5d/src/vllm_router/routers/routing_logic.py),
[deployment example](https://github.com/vllm-project/production-stack/blob/6a33ae48b896361d076a81470dca832ab926cb5d/tutorials/assets/values-17-kv-aware.yaml),
[LMCache worker transport](https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/cache_controller/worker.py),
[LMCache CPU cache operations](https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/storage_backend/local_cpu_backend.py),
[LMCache controller lookup](https://github.com/LMCache/LMCache/blob/dev/lmcache/v1/cache_controller/controllers/kv_controller.py),
[production-stack loadaware design](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/loadaware-routing.html).

## LMCache controller availability probe

On 2026-09-12 (report completed at 03:23 UTC), both supplied URLs returned:

| Request | a | b |
| --- | --- | --- |
| `GET /openapi.json` | 200, vLLM inference API | 200, vLLM inference API |
| `GET /metrics` | 200, no LMCache-named metric samples | 200, no LMCache-named metric samples |
| `POST /lookup` using synthetic rendered tokens | 404 | 404 |
| `GET /controller/workers` | 404 | 404 |
| `GET /controller/key-stats` | 404 | 404 |
| `GET /lookup/info` | 404 | 404 |
| `GET /instances` (MP coordinator) | 404 | 404 |
| `POST /directory/lookup` (MP coordinator) | 404 | 404 |

These results establish that the two tested HTTPS base URLs did not expose the
queried controller interfaces at that time. They do **not** establish that no
controller or LMCache worker exists elsewhere. The deployment owner is bringing
up a cache controller; its actual URL, API version and worker registrations are
still needed. No LMCache data source has been enabled or validated in the Router.

Repeat the read-only test against its explicit URL:

```sh
python scripts/routing/probe_lmcache.py --url https://ACTUAL-CONTROLLER-URL
```

For the in-process controller used by the inspected production-stack design,
`POST /lookup` returns `layout_info: {instance_id: [location, matched_tokens]}`.
An available endpoint with an empty lookup is not proof of empty caches: verify
the registered instances, use a request longer than a complete cache chunk,
render exactly the inference tokens, warm only one worker, then query again.
The expected change must map to that worker. Check freshness, eviction and
restart behavior before accepting this as trusted per-worker prefix evidence.

Keep storage location explicit when adapting the result. A `LocalCPUBackend`
match can provide useful locality, but is not a GPU prefix hit; an ECT model
must include restoration cost for that tier. The three ranking functions and
dispatch lifecycle remain shared regardless of the evidence provider. Serving
fingerprints, stable engine mapping, and measured service-cost calibration are
still necessary even if controller lookup is available.

Contracts:
[controller lookup](https://docs.lmcache.ai/kv_cache_management/lookup.html),
[controller workers and key statistics](https://docs.lmcache.ai/internal_api_server/controller_apis.html),
[MP coordinator APIs](https://docs.lmcache.ai/mp/coordinator.html).
