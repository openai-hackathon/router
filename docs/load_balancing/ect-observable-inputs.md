# KV-Batch-ECT: observable and inferred inputs from existing HTTP APIs

Updated: 2026-09-12. This analysis separates direct observations, conditional estimates, and unknown information. It does not claim that production calibration is complete.

**The LMCache adapter is integrated into the Router. For these experiments, the user specified identical model/serving settings and a fixed endpoint-to-engine mapping, so `lmcache.identity_mode: "endpoint"` temporarily skips identity verification. The direct completion-time model is now implemented behind `ect_model: "completion_time"`; it uses CPU lookup inventory and backend load without requiring a separately identifiable CPU restoration curve. Representative per-worker calibration data is still missing.** The default remains the existing `decomposed` model. See the [completion-time routing guide](completion-time-routing.md) for the new mode, model configuration, individual-attempt logging, and calibration workflow.

**Later observation:** the newer [d Router smoke](results/router-d-partial-prefix-smoke-2026-09-12.json) recorded external hits. One window consistent with a single completion had native=1552, external=4, and compute=1. The user confirmed a background benchmark on d; other windows had multiple completions despite only one request from this experiment. These samples are not unloaded calibration runs. The external=0 results in the earlier c/d and six-request c experiments below are historical observations, not a claim that external hits remain zero. See the [Cache-aware ECT design](cache-aware-ect-design.md) for the resulting interpretation and model changes.

Adapter configuration is documented in [LMCache adapter](lmcache-adapter.md). The unknowns below distinguish deployment assumptions from direct observations. Missing epoch/fingerprint verification does not block endpoint mode. Lookup or health failures, expired data, and missing required cost models still trigger common fallback.

## Scope of the recorded experiments

This analysis uses the existing [six-request c results](results/lmcache-c-smoke-2026-09-12.json) and the subsequent [c/d cross-endpoint results](results/lmcache-cd-observations-2026-09-12.json). The latter experiment called only known renderer, chat, metrics, lookup, and health routes on the user-specified c and d endpoints. It did not call a or b, or invoke clear, pin, move, or compress.

The c/d experiment used the same synthetic prompt. Both renderers produced 1042 tokens, and all four generation responses returned exactly the same prompt token IDs. Inference ran sequentially: c first request, c repeat, d first request, d repeat, each with `max_tokens=8`. Both controllers were queried to inspect their respective inventory. Sequential inference simplified metrics attribution.

| Observation point | c `/lookup` | d `/lookup` |
| --- | --- | --- |
| Before sending the new prompt | `layout_info={}` | `layout_info={}` |
| After c first generation | `vllm-c: [LocalCPUBackend, 1024]` | `layout_info={}` |
| After c repeat | `vllm-c: [LocalCPUBackend, 1024]` | `layout_info={}` |
| After d first generation | `vllm-c: [LocalCPUBackend, 1024]` | `vllm-d: [LocalCPUBackend, 1024]` |
| After d repeat | Unchanged | Unchanged |

Each controller's `POST /health` used its own instance ID and returned HTTP 200 with `error_codes={"0":0}`. These observations support querying each controller's local CPU inventory. They do not show that c can enumerate d, or that absent a/b entries in c's response mean those workers have no cache.

## Availability of the four input categories

| ECT input | What existing data provides | What it does not establish | Suggested annotation |
| --- | --- | --- | --- |
| Per-worker prefix evidence | `/lookup` on exact request tokens reports instance, storage tier, and matched prefix length; isolated metrics windows can measure actual native/external token hits afterward | CPU inventory is not GPU-resident prefix; retrospective hit rate cannot predict another prompt's hits | `controller_inventory`, `tier=LocalCPUBackend`, `observed_at`; keep GPU evidence separate |
| Engine identity and invalidation | Configured endpoint-to-instance mapping, health, exporter process start, metric creation, and counter resets can signal invalidation | Instance name, `engine="0"`, and operation IDs are not engine epochs; stable counters do not rule out eviction | `identity=configured`, `engine_epoch=unknown`; record weak invalidation signals separately |
| ECT service/completion time | Individual Router dispatch-to-terminal samples with actual output usage; backend phase `_sum/_count` differences in isolated windows; pre-dispatch Router in-flight and backend running/waiting | A few samples do not calibrate a general model; aggregate backend timings omit parts of network and Modal queuing and cannot identify individual requests under mixed traffic | Individual attempt boundary, background-load observations, attribution checks, model domain, and validation error |
| CPU cache restoration cost | Later external-hit observations permit paired experiments across restored lengths; retrieve/GPU-transfer metrics could separate phases if exposed | A few external tokens and mixed elapsed time do not identify a restoration curve; repeat speedup and lookup RTT are not CPU-to-GPU restoration time | Pure `restore_cost=unknown`; the new direct model fits net cache benefit without claiming an isolated restore cost |

## Prefix and hit rate: preserve storage tier and observation time

LMCache's legacy lookup contract returns `(location, matched_prefix_length)` keyed by instance ID. `event_id` identifies that controller operation. [LMCache lookup documentation](https://docs.lmcache.ai/kv_cache_management/lookup.html)

The available evidence is the CPU prefix reported by the controller at lookup time. It contains no GPU block residency, insertion time, cache epoch, event sequence, or eviction watermark. HTTP receipt time is an observation timestamp, not proof of the inventory's internal freshness. TTL limits reuse of old Router observations but cannot prove that the controller itself is current.

The earlier six-request c window measured:

```text
native queried tokens = 6262
native hit tokens     = 3120
native token hit rate = 3120 / 6262 = 49.8243%

external queried tokens = 3142
external hit tokens     = 0
external token hit rate = 0 / 3142 = 0%
```

External zero here means that the counter was present and its window delta was zero; missing data was not filled with zero. Missing counters, resets, a zero denominator, or unrelated traffic require an unknown result or an aggregate-only report. LMCache lookup hit rate and vLLM's consumed native/external cache hit rates are different metrics and must not share denominators.

The later c/d experiment narrowed attribution to one request per window. Both first generations had native hits=0; both repeats had native hits=1040. Repeat deltas for `prompt_tokens_by_source_total` on each worker were `local_compute=2`, `local_cache_hit=1040`, and `external_kv_transfer=0`. The CPU lookup prefix was 1024. **GPU hits and CPU inventory were therefore observably different.**

Both workers' `cache_config_info` reported `block_size=16`. The 1040-token native hit matches `floor((1042-1)/16)*16`. vLLM 0.29 native prefix lookup uses complete blocks and reuses at most prompt length minus one token so it can produce the final token's logits. [vLLM KV cache manager](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/core/kv_cache_manager.py#L208)

Native metrics can validate what a completed request used; they cannot construct a complete GPU index for arbitrary future requests. Reuse predicted from successful request history is a TTL-limited heuristic, not exact evidence. CPU lookup lengths such as 512, 1024, and 1536 also do not uniquely determine LMCache chunk size. Do not hardcode chunk size from those multiples.

`kv_cache_usage_perc=0` does not imply that no GPU cache can be reused. vLLM's free block queue can retain hashed cached blocks until reuse or eviction. [vLLM block pool](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/core/block_pool.py#L30)

## Identity, configuration, and invalidation: weak detection cannot create an epoch

The c/d experiment saved these directly observable `cache_config_info` fields:

| Field | c | d |
| --- | --- | --- |
| Native block size | 16 | 16 |
| GPU block count | 10720 | 4766 |
| KV token capacity | 171520 | 76256 |
| Prefix caching | Enabled | Enabled |
| Native prefix hash algorithm | `sha256` | `sha256` |
| Cache dtype | `auto` | `auto` |
| Engine label | `0` | `0` |

This provides configured token capacity without inferring it from VRAM. However, `auto` does not identify actual dtype bytes, and these fields do not provide complete model-weight revisions, tokenizer/chat-template revisions, GPU models, or engine epochs. Capacities differ. Matching `model="local"` and token parity for one prompt do not prove identical serving fingerprints or interchangeable KV. ECT coefficients should be calibrated separately for each worker.

Adapter configuration should explicitly bind `worker_url`, `controller_url`, `instance_id`, expected serving fingerprint where verification is enabled, and supported storage tier. c/d are configuration values and must not appear in selection logic. A user-declared identity is configured, not verified. Verification requires an identifier tied to the actual inference engine that changes when the engine changes.

Existing `process_start_time_seconds`, `*_created`, and counters can provide weak exporter incarnation signals:

```text
process start changes OR metric creation changes OR a counter decreases
    -> discard the old metrics-difference window and cached Router lookup results

lookup/health timeout, unsuccessful health, or unexpected instance
    -> mark the source unknown/stale; stop claiming trustworthy prefix evidence
```

These are conservative invalidation suggestions. An exporter can restart without an engine change; an engine can change without immediate exporter replacement. An endpoint can also switch containers. None of these signals proves that old inference attempts have terminated, so they cannot release unknown ledger entries. A per-operation UUID such as `event_id` is not an ordered telemetry sequence.

Without event sequences, rechecking lookup before use, shortening TTL, and invalidating on health failure reduce exposure to stale observations. They cannot detect unreported eviction or guarantee that inventory remains unchanged between selection and dispatch. Strict identity rules remain the default in `verified` mode. The experiments use `endpoint` mode, which is recorded in decisions and metrics while unknown epochs remain null.

## Service time: histogram differences were measurable in isolated windows

Both c and d exposed these `/metrics` series:

```text
vllm:request_prefill_time_seconds_{sum,count}
vllm:request_decode_time_seconds_{sum,count}
vllm:request_queue_time_seconds_{sum,count}
vllm:request_inference_time_seconds_{sum,count}
vllm:time_to_first_token_seconds_{sum,count}
vllm:e2e_request_latency_seconds_{sum,count}
vllm:request_prefill_kv_computed_tokens_{sum,count}
vllm:request_prompt_tokens_{sum,count}
vllm:request_generation_tokens_{sum,count}
```

For a quiet single-request window with the same label set and no reset:

```text
delta(phase_count) = 1
phase_ms = 1000 * (phase_sum_after - phase_sum_before)
```

All phase counts in the four c/d windows increased by one. Prompt histogram deltas were 1042; generation histograms matched usage. Counter creation times did not change, and preemption counters did not increase. Native queries and summed prompt sources also equaled 1042. These checks support window attribution, but they do not prove engine epochs, container identity, or unloaded execution.

| Worker / request | Native H | CPU lookup H | O | Prefill ms | Decode ms | Queue ms | Backend E2E ms | Client HTTP ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c first | 0 | Not reported before; 1024 after | 3 | 86.916 | 62.387 | 0.021 | 157.178 | 1003 |
| c repeat | 1040 | 1024 | 3 | 64.007 | 63.088 | 0.021 | 134.572 | 996 |
| d first | 0 | Not reported before; 1024 after | 8 | 438.838 | 311.470 | 0.021 | 760.811 | 1752 |
| d repeat | 1040 | 1024 | 8 | 98.085 | 295.771 | 0.023 | 402.422 | 1431 |

In vLLM 0.29, prefill runs from first scheduling to the first token, decode runs from the first token to the last token, and queue time runs from enqueueing to first scheduling. These elapsed engine-event intervals can include preemption. Prefill already includes production of the first output token; decode is not the pure GPU-kernel time for all O output tokens again. [vLLM timing definitions](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/metrics/stats.py#L484)

The observed P, D, and Q values are useful measurements, but the four rows are not model coefficients. They cover one L, two H states, and two requests per worker without controlled load variation. d's slower first request may include warm-up or configuration effects. Actual O differs between workers, so total-time ratios do not directly measure relative service capacity. Even one completion does not rule out other unfinished requests sharing the GPU.

For the existing `decomposed` model, a possible calibration procedure is:

1. Fix the model/serving fingerprint and record Router pre-dispatch `n_j`. Confirm that unloaded samples have no other Router or external requests.
2. Vary L, H, and output limits; save actual O and cache source. Scrape before and after each request, wait for the expected logger count, and discard windows that time out.
3. Check label sets, resets, completion reason, prompt/output deltas, and phase counts. Keep mixed-traffic windows as aggregates rather than assigning them to individual requests.
4. Fit baseline P/D on unloaded samples, then jointly calibrate beta and Q across pre-dispatch loads. Separate training and validation groups, and retain domains and error statistics.
5. If a concurrent window only exposes N-request aggregates, `delta(sum)/N` measures the group mean. It does not preserve each request's L/H/n-to-latency mapping and cannot become N independent calibration samples.

For the implemented `completion_time` mode, use the Router's individual dispatch-to-terminal samples instead. Fit CPU lookup C, L, actual O, Router in-flight, and observed backend running/waiting together; KV usage is optional when its coefficient is zero. This supports background traffic without attributing a shared histogram window to one request. It does not require an unloaded phase baseline or a separate restoration curve. The [completion-time routing guide](completion-time-routing.md) documents dataset preparation, explicit provenance, domain checks, and the calibration CLI.

The earlier short replies had only 3 to 8 output tokens and do not represent an agent/tool workload's output prior. Usage can support a workload-specific prior, but max_tokens is a limit rather than expected O. Length-censored training requires an independent output prior in the new calibrator.

Predicting user-visible Router completion time also requires accounting for transport, HTTP preprocessing, and the Modal path. For the rows above, `Client HTTP - backend E2E` is approximately 846 to 1029 ms. This measured difference combines network, connection, platform, and measurement-boundary effects; it is not entirely network RTT or backend queue time. Individual lookup HTTP times were also approximately 0.84 to 1.03 seconds and are not GPU or CPU transfer costs. Measure these overheads separately to avoid double counting. In the new model, render/lookup happen before the dispatch-to-terminal target and remain part of the separately reported total Router E2E latency.

## CPU restoration: external-hit evidence exists, but no isolated timing curve

`/lookup` confirms CPU inventory. The two earlier smoke rounds above had zero external cache-hit deltas, and the c/d windows had `source="external_kv_transfer"` deltas of zero. Those requests therefore provided no observed connector-consumed external cache tokens. A later d window recorded external=4, but its prefill elapsed time includes other work and background load. **A CPU-to-GPU restoration curve still cannot be identified from these samples.**

The new `completion_time` model directly learns the net latency effect of CPU inventory instead of requiring pure transfer time first. This mode is implemented and opt-in; omitting `ect_model` keeps `decomposed`. Neither implementation status nor a handful of smoke requests establishes a measured production calibration.

To estimate restoration separately with existing APIs, collect requests for which lookup shows CPU prefix, actual native GPU H is shorter than that prefix, and the external-transfer token counter increases after execution. Natural GPU eviction or a controlled experiment that retains CPU inventory while reducing GPU residency could provide such samples. Ordinary warm repeats do not establish CPU restoration.

With the same worker/L/O/n, measured native GPU hits, and no other external storage source, one could fit restoration's added critical-path time:

```text
H_usable = max(H_gpu, H_cpu)          # Overlapping prefixes share a start; do not add them.
R_tokens = max(0, H_cpu - H_gpu)      # Additional tokens requiring restoration.

restore_effective_ms
    approximately equals prefill_interval_ms
                         - baseline_prefill_ms(L, H_usable)
```

This is a model-residual estimate. First verify that transfer occurs within the prefill interval. If transfer queues before scheduling or overlaps computation, fit a matching whole-backend latency boundary instead. The residual is not pure PCIe transfer time. A negative residual signals noise, overlap, or a mismatched baseline and should remain diagnostic rather than becoming a claim that restoration is free. When H_gpu exceeds H_cpu, the request is not a restoration sample.

Another option is exposing version-compatible LMCache metrics such as `time_to_retrieve`, `retrieve_to_gpu_time`, `num_hit_tokens`, and `retrieve_speed`. The official legacy metrics documentation lists them, but the recorded public `/metrics` responses contained no `lmcache:` samples. Documentation availability does not establish deployment availability. [LMCache metrics reference](https://docs.lmcache.ai/production/observability/metrics.html)

Once model layout, KV dtype, TP sharding, and effective bandwidth are known, a standard-attention byte model can provide an approximation:

```text
KV_bytes(R) approximately equals
    2 * layers * local_KV_heads * head_dim * dtype_bytes * R

restore_ms approximately equals
    fixed_overhead_ms + 1000 * KV_bytes(R) / effective_bytes_per_second
```

This applies only to the matching KV layout and omits padding, quantization metadata, compression, and overlap. Actual dtype, layout, and effective bandwidth are not yet available, so such numbers would be explicitly assumed priors rather than calibration results. Nominal hardware bandwidth is not end-to-end effective bandwidth.

The `decomposed` model still needs a tier-specific restoration cost when treating CPU prefix as avoided prefill. Substituting CPU prefix into a GPU baseline without restoration would credit saved computation while treating transfer as free. The new direct `completion_time` model instead includes the tier's net effect in its fitted cache coefficient and must not add a separate restoration term. Each model must preserve its own measurement boundary.

## Rules available at integration time

- Supply controller URLs, instance mappings, fingerprints where verified, and supported tiers through configuration. Each worker may use its own controller.
- Preserve raw lookup tier/length/operation ID, observation time, and source status. Explicitly fall back for unsupported tiers or schemas.
- `layout_info={}` only means that this controller did not report this prefix. An unhealthy or unverified source does not establish exact H=0.
- Keep H_gpu, CPU inventory, and retrospective cache-hit counters separate. Do not infer an arbitrary prompt's H from aggregate hit rate.
- Use the implemented individual-attempt samples for new completion-time calibration. Endpoint mode permits an unknown epoch. Missing model or required running/waiting data still causes common fallback. A separate CPU restoration model is required only by the older decomposed path, not the new direct model.

The recorded c/d live experiment finished with 2 renders, 4 inferences, 18 lookups, 8 metrics scrapes, and 2 health calls, all HTTP 200. No request from that experiment remains running. Its JSON artifacts contain neither prompt contents nor token IDs.
