# Improving Cache-aware ECT with the available telemetry

Date: 2026-09-12. These notes summarize the implementation, repeated lookup queries and Router smoke tests against worker d, and related upstream approaches. **The first stage is now implemented:** a configurable direct completion-time model, background backend metrics collection, and per-attempt completion samples. The partial CPU-prefix adapter correction is also implemented. Phase-based workload tracking, first-output prediction, and reuse-history features remain proposals. See the [current usage guide](completion-time-routing.md) and [three-worker configuration](../../examples/configs/completion_time_routing.json).

Keep three policies and one shared data pipeline. The first two baselines retain their original ranking. The third predicts completion time from cache evidence, load, and measured latency, so **Cache-aware ECT** describes its behavior more accurately; the CLI retains `kv_batch_ect` for compatibility. The original `beta × n` term is a concurrency approximation, without observations of actual batch composition.

## What the measurements changed

### Lookup provides useful evidence, including partial native blocks

Worker d's `POST /lookup` still returns only `event_id` and `layout_info`. The second tuple element in the latter is the matched prefix length, named `cached_tokens` inside the adapter. The HTTP response did not gain a field literally named `cached_tokens`. This matches the deployed LMCache legacy controller contract. [LMCache lookup](https://docs.lmcache.ai/kv_cache_management/lookup.html)

The [two prefix experiments](results/lmcache-d-prefix-semantics-2026-09-12.json) each generated a fresh synthetic prompt and then varied the queried prefix:

| Query | First experiment, L=1045 | Second experiment, L=1043 |
| --- | ---: | ---: |
| Full token sequence before generation | No match | No match |
| Full token sequence after generation | 1045 | 1043 |
| First 512 tokens | 512 | 512 |
| First 1024 tokens | 1024 | 1024 |
| Remove the last token | 1024 | 1024 |
| Append 16 tokens | 1024 | 1024 |
| Change the first token | No match | No match |
| Change the token at index 512 | 512 | 512 |

Six consecutive lookups of the same full token sequence in the second experiment all returned 1043. The observed differences came from warming the cache and changing the queried prefix; no random fluctuation appeared for that fixed query. Results with different prompt lengths cannot be combined into a cache-growth curve for one prompt.

The original adapter incorrectly required raw CPU prefix lengths to be multiples of the vLLM block size. The correction preserves raw lengths and normalizes only the baseline ranking prefix H to complete blocks, leaving the last prompt token to be computed. LMCache supports configurations that store incomplete chunks, but these observations do not establish a specific deployment flag or uniquely determine its chunk size. [LMCache configuration](https://docs.lmcache.ai/api_reference/configurations.html)

```text
cached_tokens = raw CPU matched prefix length
H = floor(min(cached_tokens, L - 1) / native_block_size) × native_block_size

L=1043, cached_tokens=1043, block_size=16 → H=1040
```

### External hits are no longer all zero, but pure restoration cost remains unidentified

The final window in the [d-only Router smoke](results/router-d-partial-prefix-smoke-2026-09-12.json) contained:

| Quantity | Observation |
| --- | ---: |
| Prompt L / output O | 1557 / 32 |
| CPU lookup length used at dispatch | 1557 |
| Increase in each prefill/decode/queue/inference completion count | 1 |
| Native cached-token increase | 1552 |
| External transferred-token increase | 4 |
| Locally computed-token increase | 1 |
| Prefill elapsed increase | 108.213 ms |

The three token sources sum to L, and phase counts are consistent with one completion, making this a useful observation. **It remains an aggregate window without request-ID attribution. The experiment also did not retain a generation-histogram attribution check, so this is not a complete per-request trace.**

The window contains external hits. It does not establish that 108.213 ms was the cost of transferring four tokens, or that all 1557 CPU tokens were transferred. Lookup reports CPU availability; actual use also depends on GPU overlap, final-token computation, and connector behavior. vLLM source accounting distinguishes local cached tokens from external tokens. [vLLM 0.29 stats](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/metrics/stats.py)

The decomposed model's `R = fixed + per_token × cached_tokens` charge can therefore overestimate warm-request cost. Unknown restoration must not be treated as free, but a conservative charge is not a measured transfer. With limited telemetry, learning the **net effect of this cache evidence on completion time** is more practical than forcing an unidentifiable physical decomposition. The new completion-time mode follows this approach and does not add a separate R.

### Background benchmarks make the Router ledger an incomplete view of backend load

The user confirmed that another benchmark was using d. Before all six requests in this experiment, backend metrics reported `running=2, waiting=0`, while this Router's pre-dispatch in-flight count was zero. Some windows contained one submitted request but increases of four, three, or two completions.

A counter from **d's vLLM `/metrics`** includes other backend traffic, so this difference is consistent with background requests. If the counter instead comes from this Router's `router_dispatch_finished_total`, traffic sent directly to d cannot increase it; attempts and retries must then be investigated separately.

Windows with multiple completions are marked `mixed_window_do_not_assign_phase_times`. Even a one-completion window can contain other requests occupying the batch, so all six observations are marked `unloaded_baseline_eligible=false`. They remain useful client-latency observations under background load. They cannot fit an unloaded prefill/decode baseline or assign mixed-window phase sums to this Router's request.

## Direction from related projects

| Project | Published approach | Implication for this Router |
| --- | --- | --- |
| production-stack | `LoadAwareRouter` subtracts a relative-load penalty from the cache-match ratio; load comes from Router prefill/decode request counts | Controller prefix plus load is an established approach. A weighted score can be useful without having units of completion milliseconds. |
| NVIDIA Dynamo | Combines dispatched prompt work, incoming uncached work, active KV blocks, and cache-tier credits | Router workload and phase tracking can improve the approximation that every request contributes one unit of load. |
| llm-d latency scorer | Uses TTFT/TPOT predictions and latency targets, with a composite score when predictions are unavailable | Observable latency can be predicted directly without reconstructing every scheduler phase. |

Sources: [production-stack routing code](https://github.com/vllm-project/production-stack/blob/main/src/vllm_router/routers/routing_logic.py), [Dynamo routing concepts](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/routing-concepts), and [llm-d latency scorer](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/scheduling/scorer/latency/README.md). Reviewed on the date above; main/dev documentation may change.

Their telemetry and deployments differ from ours. Their default weights are not measured c/d coefficients. We can reuse feature and measurement ideas, but combining cache and load alone is not a new research contribution.

## Inputs available without another vLLM API

| Input | Available source | Current implementation or proposed use |
| --- | --- | --- |
| L, CPU prefix C, C/L | Renderer and each worker's lookup | Integrated; preserve raw C, normalized H, tier, and observation age. |
| This Router's unfinished attempts | Shared dispatch ledger | Integrated; all rankings use the count before dispatch. |
| Backend running/waiting and KV usage | Accessible `/metrics` | Background collector now supplies model/engine-validated, expiring snapshots. KV usage is capacity pressure, not prefix evidence. |
| Counter rates and preemptions | Accessible `/metrics` | Potential additional features; the new collector does not ingest them. |
| Per-attempt completion latency | Router dispatch and body termination | Implemented as completion samples for direct latency modeling. |
| Actual O and finish reason | Response usage and termination metadata | Captured when available; missing values remain unknown. |
| TTFT and elapsed time after first output | Streaming body | Future instrumentation; nonstream responses do not expose a first-output timestamp. |
| Prompt work awaiting first output | Store L/C in the ledger and change phase at first model output | Future workload proxy, without claiming GPU prefill progress. |
| Decoding request count and total context length | The same ledger | Future fields; totals would not equal deduplicated GPU KV occupancy. |
| Time and count of recent successful prefix reuse | Router request history | Possible TTL features; they estimate residency likelihood rather than proving GPU hits. |
| Output-length prior | Completed-request usage and finish reason | Offline calibration handles capped output lengths; priors should represent the target workload. |

Global backend hit rate describes recent workload averages and cannot determine the hit for an arbitrary incoming prompt. Under shared traffic, global tokens/s is aggregate throughput rather than an individual request's decode speed.

Pure CPU-to-GPU time, current GPU prefix residency, exact batch composition, and engine epoch still cannot be uniquely inferred. Endpoint identity remains an explicit deployment assumption as requested; obtaining an epoch is not a prerequisite for this experiment. Future LMCache retrieve/transfer metrics could support a finer decomposition, but the captured public metrics contained no `lmcache:` samples. [LMCache metrics](https://docs.lmcache.ai/production/observability/metrics.html)

## Changes to the third policy

### Implemented first stage: predict completion time and learn net cache benefit

Let C be the controller's raw CPU prefix and X the available dispatch-time load features. Each endpoint has its own model:

```text
E_j = completion_model_j(L, C, estimated_output, X)
```

The implemented `ect_model: "completion_time"` uses:

```text
E_ms = intercept_ms
     + prompt_token_ms × L
     + output_token_ms × estimated_output
     + prompt_output_token_ms × L × estimated_output
     - cache_token_ms × C
     + router_inflight_ms × n_router
     + backend_running_ms × running
     + backend_waiting_ms × waiting
     + kv_usage_ms × kv_usage_fraction
```

The training target is **elapsed time from Router dispatch to successful response-body termination**. It includes backend/Modal queueing, inference, and transfer within that boundary. Both nonstream and confirmed streaming terminal paths produce completion samples. First-output timing is not required for this first stage.

The model treats CPU prefix as a predictor whose fitted benefit absorbs GPU overlap and CPU restoration. It neither estimates transferred-token counts nor treats C as an exact GPU-resident H. Cache effects may vary with L, load, and reuse age; a global hit rate or an arbitrary fixed percentage cannot establish them. The implemented linear form is an initial approximation. Better interactions or reuse-history features require additional data and validation.

An offline calibration tool now fits this model from per-attempt samples and validates supported feature ranges. It requires varied, attributable observations and held-out groups. The few d smoke samples are insufficient to fit a full deployment model. See [completion-time configuration and calibration](completion-time-routing.md).

### Future stage: split first output from subsequent decoding

```text
F_j = first_output_model_j(L, C, X)
D_j = effective_time_per_output_token_model_j(L, X)
E_j = F_j + max(estimated_output - 1, 0) × D_j
```

F measures dispatch to the first actual model output; D measures average elapsed time per output token after that. The first token is already included in F. vLLM's prefill/decode timing also splits at the first token, but Router HTTP observations include different transport effects and must remain a separate sample type. [vLLM timing definitions](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/metrics/stats.py#L484)

HTTP 200 headers, role-only SSE events, and empty deltas are not first model output. Content, reasoning, and tool-call output need appropriate handling. An SSE chunk may contain multiple tokens, so chunk counts cannot stand in for O. This would be an observable latency approximation, not exact GPU inter-token timing.

**A model that already captures cache, load, and queue effects must not add full-prefix R or Q again, or multiply by `1+beta*n`.** The existing decomposed model and new empirical model are alternative versions of the third policy. They are never stacked in one decision.

Render/lookup is shared preprocessing already completed before selection. Record its overhead for end-to-end benchmarks rather than adding it independently to each candidate's score. Some lookup calls in this experiment took around a second; that belongs in Router E2E reporting. Lookup RTT is not CPU restoration time.

### Represent background traffic and future phase features in the shared snapshot

Keep `n_router`, `backend_running`, `backend_waiting`, and metrics age as separate features and calibrate them jointly. **Do not use `n_router + running + waiting` as a total request count.** They overlap and are observed across different timing and queue boundaries. An explicitly labeled `max(n_router, running+waiting)` could be a coarse pressure proxy, but it would still not reconstruct an exact total. The implemented model uses separate features.

A future configuration could identify the deployment as `exclusive_router` or `shared_backend`; those names are not current config fields. The implemented background collector invalidates failed or expired samples. A model requiring those inputs triggers common fallback when they are unavailable; it does not substitute idle. Any future change to the baselines' load definition needs a separately named experiment. Currently `least_load_kv` still uses the Router ledger and cannot balance external traffic it cannot observe.

Future phase tracking can retain each request's prompt workload until first output, then move it into decoding-request/context counters. This does not require per-request remaining-token prediction. The model must determine the CPU-prefix discount; `L-C` is not established as remaining GPU computation. Reservation, phase updates, and release-once behavior should use the same lifecycle.

### Match score names to their units and provenance

A manually weighted `cache_credit - load_penalty` heuristic could be called **Cache-aware Cost Routing**, with an explicit `score_units=relative`. Such a mode must not be presented as millisecond ECT or reuse the `+10 ms` affinity margin. A relative-score mode and this field remain proposals; the current config does not support them.

The implemented completion-time model always produces milliseconds and declares its source as `measured`, `estimated`, or `synthetic`. Calling an approximate model Cache-aware ECT does not require identifying every physical cost, but claims of calibrated performance require measured targets, applicable domains, and held-out validation. All candidates must use comparable scores; missing timing models cannot be replaced with unrelated relative scores for only some workers.

## One Router with at least three workers

The intended deployment is **one Router configuration with at least three inference workers**. The adapter queries the configured worker list without hard-coded c/d URLs or a two-worker assumption. Bounded lookup concurrency does not cap the total candidate count at the concurrency limit.

| Policy | Ranking changes for three or more workers | Required work |
| --- | --- | --- |
| `prefix_max` | None: largest H, then smallest n | Obtain each worker's prefix with consistent full-block CPU-inventory semantics. |
| `least_load_kv` | None: smallest n, then largest H | The shared ledger supports multiple workers; direct backend benchmarks remain outside its n. |
| `kv_batch_ect` | Retain minimum predicted completion time; improve the model | Model each endpoint's service capacity, background pressure, and net cache benefit, then compare scores in the same units. |

Adding workers does not supply their ECT coefficients. Identical models can run at different speeds because of GPU hardware, KV capacity, and background traffic. Measurements from d cannot establish the timing curves of two other workers. Model structure can be shared; coefficients need per-worker samples or explicit, validated conditions for sharing.

Each worker needs a correct inference URL → controller → instance binding. The completion-time mode also requires its metrics URL and a model keyed by instance ID. **If one healthy candidate lacks necessary prefix evidence or an applicable cost model, the entire decision falls back.** Report fallback rates, particularly when adding workers. A persistently unconfigured third worker does not produce a valid ECT benchmark. Missing H is not zero, and healthy workers are not silently removed for missing telemetry.

Three-candidate fixtures verify the original G0/G2/G1 ranking example and endpoint identity mode. The new completion-time smoke additionally verifies selection changes caused by external load, KV pressure, and updated lookup results, followed by common fallback when one healthy worker loses metrics. The d-only live smoke validates integration, not multi-worker policy performance. A future live comparison should arrange distinct cache, load, and service-time preferences and verify the chosen worker, reservations, and fallback reasons.

Deploying three **Router processes** is a different problem. The current ledger is shared within each process; reservations do not synchronize across Router processes. That deployment needs coordination or an explicit cross-Router load strategy.

## Implementation and calibration sequence

1. **Implemented: shared completion sampling.** Record per-attempt dispatch/terminal elapsed time, L/C/H, actual usage, finish reason, and pre-dispatch load. Failed attempts are labeled; unconfirmed cancellations are not successful completion samples. Retries have separate attempt IDs, and request E2E time remains a separate quantity. Ordinary logs contain no prompt or token arrays. First-output and phase timing remain future work.
2. **Implemented: background gauge collection.** Poll configured workers with model/engine validation, bounded requests, and freshness checks. Keep backend load separate from the ledger and avoid synchronous scrapes in dispatch. The current collector reads gauges; it does not derive rates from counters. Any future rate collector must handle resets. Use only features known at dispatch, never subsequent hit-counter changes as predictor inputs.
3. **Implemented tooling; deployment calibration still required.** Collect varied L, C/L, O, and load for each worker. Per-request latency can be measured during a background benchmark if load features are recorded. Unattributable phase histograms remain aggregate data. Unloaded P/D measurements require separate isolated windows.
4. **Future: phase workload and streaming prediction.** Add prompt/context counters to the existing ledger and verify first-output/terminal transitions happen once. Keep nonstream observations coarse instead of inventing first-token timestamps. Reuse-history features require separate validation.
5. **Implemented: alternative third-policy model.** `ect_model: "completion_time"` selects direct completion prediction; omitted configuration retains `decomposed`. Models declare provenance, version, and domains; calibration reports include validation results. The CLI and common fallback remain compatible, and the two baselines are unchanged.
6. **Next: evaluate routing on held-out workloads.** Split training and validation by prompt/session group to avoid prefix leakage. Compare client completion p50/p95, TTFT when available, throughput, lookup overhead, fallback rates, and decision errors. New workers and out-of-domain requests require explicit fallback; ignoring data-poor workers cannot substitute for validation.

The [HTTP Router](../../src/routers/http/router.rs) still records its existing `record_generate_duration` when the response object is returned; for streaming, that is not body completion and is not the new training target. The [dispatch ledger](../../src/core/dispatch.rs) now emits separate per-attempt completion samples at confirmed termination. First-output and phase features are not yet recorded.

Historical validation from the partial-prefix change comprised three adapter unit tests, 15 snapshot tests, 26 Python smoke/calibration tests, and Clippy. All six real d-only Router requests succeeded; the two baselines had no fallback, while the original ECT run reported `missing_cost_model` because no model was configured. Those counts describe that earlier experiment, not the expanded current suite. The current [smoke workflow](../../scripts/routing/smoke.sh) additionally covers the completion-time mode. No claim of policy speedup is based on comparing different prompts sent to a single live worker.
