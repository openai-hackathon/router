# Cache-aware ECT: completion-time model

`--policy kv_batch_ect` now supports `ect_model: "completion_time"`, which predicts completion time directly. `prefix_max` and `least_load_kv` retain their original ordering. All three policies still share tokenization, lookup, reservations, and request lifecycle handling.

## Worker selection

For each healthy candidate, the Router queries that worker's LMCache controller using the same request's actual renderer tokens. C is the raw CPU matched prefix, including any tail that is not aligned to a vLLM block. H is the prefix rounded down to whole blocks while reserving the final prompt token. The new model uses C for cache benefit; the baselines and bounded affinity continue to use H.

```text
E_ms = intercept_ms
     + prompt_token_ms * L
     + output_token_ms * estimated_output
     + prompt_output_token_ms * L * estimated_output
     - cache_token_ms * C
     + router_inflight_ms * n_router
     + backend_running_ms * running
     + backend_waiting_ms * waiting
     + kv_usage_ms * kv_usage_fraction
```

Select the smallest E, break ties by stable worker URL order, then apply the existing bounded session affinity. Estimated output is the output prior capped by the request limit. The model currently supports one output choice; `n` or `best_of` values other than 1 cause a common fallback.

`cache_token_ms` represents the calibrated net cache benefit, including GPU overlap, CPU restoration, and avoided prefill computation. It is not a PCIe transfer rate. CPU inventory does not establish GPU-resident prefix length. KV usage measures capacity pressure and cannot replace the request-specific lookup result.

`n_router`, running, and waiting are separate features fitted jointly. The code does not add them into a single request count. Do not copy the same per-request queue penalty into every coefficient. The direct completion model already includes load, transport, and cache effects: **it does not add `restore_models`, add Q, or multiply by beta**.

## Configuration for three or more workers

See [completion_time_routing.json](../../examples/configs/completion_time_routing.json) for a complete parseable example. Its three worker, controller, and metrics URLs are placeholders; no c or d endpoint is hardcoded. All example coefficients are marked `synthetic`. They illustrate the format and are not calibration results for c or d.

The configuration adds three top-level fields:

| Field | Purpose |
| --- | --- |
| `ect_model: "completion_time"` | Use the same new time-model family for every candidate |
| `completion_models` | A separate model for each instance ID |
| `backend_metrics` | Model, worker URL to complete metrics URL mapping, and polling, timeout, and freshness settings |

Keep the existing `lmcache` configuration. Omitting `ect_model` selects `decomposed` for compatibility with existing configurations. The two model families cannot be mixed within one decision.

```bash
cp examples/configs/completion_time_routing.json /tmp/routing.json
# Edit /tmp/routing.json with the three deployed URLs, instance IDs,
# block sizes, and model settings.

./target/debug/vllm-router \
  --worker-urls "$G0" "$G1" "$G2" \
  --policy kv_batch_ect \
  --routing-state-config /tmp/routing.json \
  --intra-node-data-parallel-size 1 \
  --health-check-endpoint /v1/models
```

`source` must explicitly be `measured`, `estimated`, or `synthetic`. For an approximate experiment, supply `estimated` coefficients and their applicable ranges. The Router does not claim those coefficients are calibrated. Every candidate score is in milliseconds; provenance and version are recorded in its `completion_cost`.

`backend_metrics` uses a separate HTTP pool and collects data in the background. Selection reads a local snapshot. Authentication headers come from the shared `header_env`. Every worker must expose `vllm:num_requests_running` and `vllm:num_requests_waiting` for the same model and engine. Gauges from multiple engines behind one endpoint are not summed.

GPU pressure comes from `vllm:kv_cache_usage_perc`, with a range of 0 to 1. When `kv_usage_ms > 0`, a missing gauge causes fallback. A zero coefficient permits an unknown gauge, which remains null in diagnostics. Scrape age starts when the HTTP request begins; it does not guarantee that the vLLM logger updates internally at the same frequency.

## Invalid data and fallback

If any healthy candidate lacks required data, the whole decision falls back to Router least-load. The candidate is not removed or treated as having zero cache or idle load. Existing KV health, schema, TTL, and endpoint/verified identity rules remain in effect. Common new reasons include:

```text
missing_completion_model
missing_backend_metrics
stale_backend_metrics
invalid_backend_metrics_binding
missing_backend_kv_usage
invalid_backend_kv_usage
outside_completion_calibration_range
invalid_completion_prediction
unsupported_completion_choices
unsupported_completion_evidence
```

The model checks the ranges of L, cache fraction, estimated output, Router in-flight, running, waiting, and KV usage, and requires finite nonnegative coefficients. If the cache credit produces E <= 0, selection falls back instead of treating the worker as free. The new model is calibrated against LMCache `LocalCPUBackend` inventory. Native GPU event evidence requires the original model or a separately defined calibration contract.

All three workers need usable data and models for normal ECT comparisons. Configuring only two can cause every request to fall back. When adding a worker, update the CLI URLs, LMCache binding, metrics URL, and completion model together.

## Sampling and calibration

Each attempt using an observed policy can emit a `routing completion sample` log with a `sample` JSON object. It records pre-dispatch L/C/H, load observations, actual `completion_ms`, output usage, finish reason, and attempt ID, without prompt text or token IDs. Timing starts at backend dispatch and ends when the complete nonstreaming body or a confirmed streaming terminal is observed. Rendering and lookup finish before that boundary and must be included separately in overall Router E2E evaluation.

Response headers do not trigger completion samples. Confirmed failures have success=false; unconfirmed disconnects or cancellations do not produce successful samples. Retries are recorded separately: total request retry time is not a single-attempt training target. Missing usage or finish reason remains null; token counts are not guessed. To collect streaming output usage, a client can request the backend's supported usage reporting. The Router does not rewrite the original request options.

Extract the log's `sample` objects into JSONL, select successful samples with complete usable data, and assign `group_id` and `split` (`train` or `validation`). Requests sharing a prefix or session must stay in the same group and cannot span both splits. Samples with missing data or invalid KV evidence are diagnostic only. Individual latency under an external benchmark is still useful when its background-load fields are complete. Do not replace individual timings with differences of global completion `_sum` counters.

```bash
uv run --no-project --with numpy --with scipy \
  python scripts/routing/calibrate_completion.py \
  --input prepared-samples.jsonl \
  --output /tmp/routing-calibrated.json \
  --report /tmp/completion-validation.json \
  --version experiment-01 \
  --base-config /tmp/routing.json
```

With `--base-config`, the calibrator preserves adapter, metrics, authentication, and other existing settings while replacing `ect_model` and the complete `completion_models` map. Without it, the output is a model configuration fragment. Input, output, report, and base config paths must be distinct. A rejected validation writes its report but leaves the base config and any existing output unchanged.

The calibrator uses individual completion times and does not require isolated prefill or CPU restoration timings. It rejects insufficient feature variation, confounded features, and excessive validation error. The small set of d smoke samples does not cover enough conditions to fit the full model. If training samples lack the optional KV usage gauge, that column is disabled and reported; required running and waiting values are never filled with zero.

If training includes `finish_reason="length"`, provide an independently estimated `--output-prior` to avoid treating a truncated length as the natural output prior. Validation checks both known actual output and the capped prior used at dispatch. Each worker needs at least three validation samples and enough training samples with distinguishable features. Meeting the minimum sample count does not establish workload coverage.

```bash
bash scripts/routing/smoke.sh
```

Smoke tests run the real Router binary against three worker fixtures. Changes in external load, KV usage, and lookup results independently change selection; invalid metrics cause common fallback. Existing tests for both baselines, streaming, retries, and reservations continue to run.

## Live integration check

The [2026-09-12 d integration result](results/router-completion-live-2026-09-12.json)
records one nonstreaming request and one streaming repeat through the real Router
to the deployed vLLM/LMCache endpoint. Both returned HTTP 200 with
`ect_model="completion_time"` and no fallback. The raw CPU prefix changed from
0 to 609 tokens; the repeated request used H=608 complete-block routing tokens.
Both requests emitted one successful completion sample with three output tokens,
and the final Router in-flight count was zero.

The model was explicitly synthetic. This validates the deployed HTTP/lookup/
metrics/streaming connection, not the coefficients or multiworker performance.
Three-worker selection, held-stream reservations, retries, and failure fallback
are exercised by the local E2E fixture test.
