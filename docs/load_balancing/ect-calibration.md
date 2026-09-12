# ECT calibration and offline validation

The Router accepts per-worker measured cost models under `cost_models` in
`--routing-state-config`. `scripts/routing/calibrate_ect.py` produces that config
fragment from explicit request timing samples. It does not need a Controller,
scrape metrics, or run inference. A telemetry/measurement adapter can supply the
same samples later.

Each ECT candidate now records the calibration version, uncached tokens,
estimated output tokens, prefill/decode milliseconds, load multiplier, queue
correction and final score. These fields appear in the existing decision log
and offline demo. No prompt text or token array is added to logs.

## Input contract

Supply one JSON object per line, with these exact fields:

```json
{
  "sample_id": "worker-a-request-001",
  "group_id": "context-family-001",
  "worker_id": "worker-a",
  "fingerprint": "model-tokenizer-serving-fingerprint",
  "source": "measured",
  "split": "train",
  "prompt_tokens": 1024,
  "reusable_tokens": 256,
  "output_tokens": 128,
  "max_output_tokens": 256,
  "inflight": 0,
  "prefill_ms": 200.0,
  "decode_ms": 300.0,
  "completion_ms": 520.0
}
```

The numbers above only illustrate the schema. `source` must be `measured` or
`synthetic`, consistently throughout a dataset. The tool records this declared
provenance; it cannot independently establish how input measurements were made.

- `inflight` is the Router count before reserving this request, not backend
  running plus waiting plus Router load.
- `reusable_tokens` is the attributable GPU prefix reuse used for the timing
  sample. CPU/remote hits require a separate restoration model and are outside
  this calibration contract.
- `prefill_ms` and `decode_ms` are non-overlapping service phase durations in
  milliseconds, excluding queue time. They are required for unloaded samples
  (`inflight = 0`). For loaded samples they may be null; loaded phase durations
  are not used to fit unloaded service capability.
- `completion_ms` covers the dispatch-to-completion interval being predicted,
  including the queue/transport overhead that the model absorbs in its fitted
  load and queue terms. Use one consistent measurement definition for all rows.
- Do not substitute TTFT directly for prefill service time or derive individual
  request timings from aggregate histograms. The vLLM metric adapter must first
  establish the timing semantics.
- `output_tokens` is the actual generated length. `max_output_tokens` is the
  request budget and is used to cap the output-length prior during validation.
- `split` is `train` or `validation`. Keep related prompts, shared prefixes and
  repeated sessions together under `group_id`. A group cannot occur in both
  splits, even on different workers. `sample_id` must be unique.

Each worker needs at least six unloaded training rows, at least two distinct
training in-flight counts, and at least three validation rows. These minimum
counts are input guards, not evidence that a real calibration dataset is large
enough. Feature matrices must have full rank and acceptable conditioning. A
worker may have only one serving fingerprint in a calibration run.

## Fit and validate

The tool fits the same model that the Router evaluates:

```text
P = a0 + a1 (L - H) + a2 (L² - H²)
D = d0 + d1 O + d2 L O
E = (P + D) (1 + beta n) + Q
```

It first fits P/D from unloaded phase measurements with nonnegative least
squares. Holding those models fixed, it jointly fits beta and Q against the
completion residual using columns `[(P + D) n, 1]`. It does not add a separately
measured queue term to Q. Column scaling reduces numerical conditioning problems.
The solver is [SciPy NNLS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html).

The output prior is the median actual output length of training requests.
Validation reports two errors separately:

1. ECT evaluated with the **actual** output length, checking the service/load fit.
2. ECT evaluated with the **capped prior**, checking the estimate available at
   routing time.

Both reports include MAE, mean absolute percentage error (MAPE), and the 95th
percentile relative error. Both MAPEs must pass `--max-validation-mape` (default
0.25). That threshold is an initial tooling setting, not a measured quality
guarantee or an application SLO. Validation outside the training domain fails;
the tool does not silently extrapolate or discard those samples.

```sh
uv run --no-project --with numpy --with scipy \
  python scripts/routing/calibrate_ect.py \
  --input /path/to/measured-samples.jsonl \
  --output /path/to/cost-config.json \
  --report /path/to/calibration-report.json \
  --version experiment-001
```

If validation fails, the report is written, the process exits with status 2,
and the cost config is not overwritten. Invalid samples fail before export.
Successful versions are labeled `measured:experiment-001` or
`synthetic:experiment-001`. Merge the resulting `cost_models` field into an
existing Router config to preserve its renderer/credential settings. Runtime
out-of-domain or missing models still cause the common least-load fallback.

## Exercise the toolchain with synthetic data

```sh
python scripts/routing/make_calibration_fixture.py \
  --output /tmp/ect-samples.jsonl

uv run --no-project --with numpy --with scipy \
  python scripts/routing/calibrate_ect.py \
  --input /tmp/ect-samples.jsonl \
  --output /tmp/ect-cost-config.json \
  --report /tmp/ect-calibration-report.json \
  --version fixture-v1

cargo run --example routing_policy_demo -- \
  --config /tmp/ect-cost-config.json
```

The generator produces 504 synthetic rows for three workers, with known
coefficients and distinct context groups for training/validation. The tests
check coefficient recovery, output-prior errors, budget capping, group leakage,
bad units/values, fingerprint mixing, inadequate coverage, and rejected exports.
Coefficient recovery on generated data verifies implementation consistency; it
does not establish prediction accuracy on vLLM.

The demo compares all three policies on the same fixed synthetic cache/load
snapshot. Its predictions and chosen workers are useful for debugging model
changes. It is not a latency benchmark: a different routing decision would
change future load and KV distribution in a live run.
