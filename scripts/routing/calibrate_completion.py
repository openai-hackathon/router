#!/usr/bin/env python3
"""Fit cache-aware dispatch-to-terminal latency from prepared per-attempt JSONL.

Input is individual successful requests, never differences of aggregate metrics.
Assign train/validation groups by shared prompt/session before using this tool.
A length-censored training set requires an explicitly supplied output prior.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.optimize import nnls

from calibrate_ect import CalibrationError, error_summary, write_json


FIELDS = {
    "sample_id",
    "group_id",
    "worker_id",
    "fingerprint",
    "source",
    "split",
    "prompt_tokens",
    "cached_tokens",
    "output_tokens",
    "max_output_tokens",
    "inflight",
    "backend_running",
    "backend_waiting",
    "kv_usage_fraction",
    "completion_ms",
    "finish_reason",
}
OPTIONAL_FIELDS = {
    "attempt_id",
    "request_id",
    "success",
    "worker_url",
    "policy",
    "model_kind",
    "identity_mode",
    "fallback_reason",
    "cache_location",
    "reusable_tokens",
    "backend_sample_age_ms",
    "predicted_completion_ms",
    "num_choices",
}
COEFFICIENTS = (
    "intercept_ms",
    "prompt_token_ms",
    "output_token_ms",
    "prompt_output_token_ms",
    "cache_token_ms",
    "router_inflight_ms",
    "backend_running_ms",
    "backend_waiting_ms",
    "kv_usage_ms",
)
INTEGER_FIELDS = (
    "prompt_tokens",
    "cached_tokens",
    "output_tokens",
    "inflight",
    "backend_running",
    "backend_waiting",
)
# Cost-model failures still yield useful observations for bootstrapping or
# extending calibration. Evidence/identity/metrics failures do not.
CALIBRATABLE_FALLBACKS = {
    None,
    "missing_completion_model",
    "incompatible_completion_model",
    "outside_completion_calibration_range",
    "invalid_completion_prediction",
    "missing_backend_kv_usage",
    "missing_cost_model",
    "incompatible_cost_model",
    "outside_calibration_range",
    "invalid_cost_prediction",
    "missing_restore_model",
    "incompatible_restore_model",
    "outside_restore_calibration_range",
    "invalid_restore_prediction",
}


def validate_samples(samples):
    if not samples:
        raise CalibrationError("empty calibration input")
    ids, groups, sources = set(), {}, set()
    for row in samples:
        if (
            not isinstance(row, dict)
            or not FIELDS <= set(row)
            or set(row) - FIELDS - OPTIONAL_FIELDS
        ):
            raise CalibrationError("sample fields do not match the completion schema")
        for field in ("sample_id", "group_id", "worker_id"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise CalibrationError(f"invalid {field}")
        if row["fingerprint"] is not None and (
            not isinstance(row["fingerprint"], str) or not row["fingerprint"].strip()
        ):
            raise CalibrationError(
                "fingerprint must be nonempty or null for endpoint identity"
            )
        choices = row.get("num_choices", 1)
        if type(choices) is not int or choices != 1:
            raise CalibrationError("completion calibration requires exactly one choice")
        if row.get("success", True) is not True:
            raise CalibrationError("only successful individual attempts can be fitted")
        if "cache_location" in row and row["cache_location"] != "LocalCPUBackend":
            raise CalibrationError(
                "completion calibration requires CPU lookup evidence"
            )
        fallback = row.get("fallback_reason")
        if (
            fallback is not None and not isinstance(fallback, str)
        ) or fallback not in CALIBRATABLE_FALLBACKS:
            raise CalibrationError(
                "cannot calibrate an attempt with invalid routing evidence"
            )
        if row["sample_id"] in ids:
            raise CalibrationError("duplicate sample_id")
        ids.add(row["sample_id"])
        if row["split"] not in ("train", "validation"):
            raise CalibrationError("split must be train or validation")
        if groups.setdefault(row["group_id"], row["split"]) != row["split"]:
            raise CalibrationError("group occurs in both train and validation")
        if row["source"] not in ("measured", "synthetic"):
            raise CalibrationError("sample source must be measured or synthetic")
        sources.add(row["source"])
        for field in INTEGER_FIELDS:
            if type(row[field]) is not int or not 0 <= row[field] <= 10_000_000:
                raise CalibrationError(f"invalid {field}")
        limit = row["max_output_tokens"]
        if limit is not None and (
            type(limit) is not int or not 1 <= limit <= 10_000_000
        ):
            raise CalibrationError("invalid max_output_tokens")
        if (
            row["prompt_tokens"] == 0
            or row["output_tokens"] == 0
            or row["cached_tokens"] > row["prompt_tokens"]
            or (limit is not None and row["output_tokens"] > limit)
        ):
            raise CalibrationError("inconsistent token counts")
        if row["finish_reason"] not in (
            "stop",
            "tool_calls",
            "function_call",
            "length",
        ):
            raise CalibrationError("missing or unsupported successful finish_reason")
        duration = row["completion_ms"]
        if (
            type(duration) not in (int, float)
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise CalibrationError(
                "completion_ms must be a finite positive attempt duration"
            )
        usage = row["kv_usage_fraction"]
        if usage is not None and (
            type(usage) not in (int, float)
            or not math.isfinite(usage)
            or not 0 <= usage <= 1
        ):
            raise CalibrationError("invalid kv_usage_fraction")
    if len(sources) != 1:
        raise CalibrationError("cannot mix synthetic and measured samples")


def features(row, output=None, use_kv_usage=True):
    output = row["output_tokens"] if output is None else output
    return [
        1.0,
        row["prompt_tokens"],
        output,
        row["prompt_tokens"] * output,
        -row["cached_tokens"],
        row["inflight"],
        row["backend_running"],
        row["backend_waiting"],
        row["kv_usage_fraction"] if use_kv_usage else 0.0,
    ]


def fit_coefficients(rows, use_kv_usage):
    matrix = np.asarray(
        [features(row, use_kv_usage=use_kv_usage) for row in rows], dtype=float
    )
    target = np.asarray([row["completion_ms"] for row in rows], dtype=float)
    scale = np.linalg.norm(matrix, axis=0)
    active = scale > 0
    if not np.all(np.isfinite(scale)) or len(rows) < max(8, int(active.sum()) + 1):
        raise CalibrationError(
            "insufficient training samples for active completion features"
        )
    normalized = matrix[:, active] / scale[active]
    if (
        np.linalg.matrix_rank(normalized) != normalized.shape[1]
        or np.linalg.cond(normalized) > 1e8
    ):
        raise CalibrationError("rank-deficient or confounded completion features")
    coefficients = np.zeros(len(COEFFICIENTS))
    try:
        fitted, _ = nnls(normalized, target, maxiter=2000)
    except RuntimeError as error:
        raise CalibrationError("completion fit did not converge") from error
    coefficients[active] = fitted / scale[active]
    if not np.all(np.isfinite(coefficients)):
        raise CalibrationError("invalid fitted completion coefficients")
    return coefficients, [
        name for name, enabled in zip(COEFFICIENTS, active) if not enabled
    ]


def capped_output(prior, limit):
    return prior if limit is None else min(prior, limit)


def fit(samples, version, max_validation_mape=0.25, output_prior=None):
    validate_samples(samples)
    if (
        not isinstance(version, str)
        or not version.strip()
        or not math.isfinite(max_validation_mape)
        or max_validation_mape < 0
    ):
        raise CalibrationError("invalid calibration version or validation threshold")
    if output_prior is not None and (
        type(output_prior) is not int or output_prior <= 0
    ):
        raise CalibrationError("invalid explicit output prior")
    workers = defaultdict(list)
    for row in samples:
        workers[row["worker_id"]].append(row)
    models, reports = {}, {}
    source = samples[0]["source"]
    for worker_id, rows in sorted(workers.items()):
        fingerprints = {row["fingerprint"] for row in rows}
        if len(fingerprints) != 1:
            raise CalibrationError(f"{worker_id}: multiple serving identities")
        train = [row for row in rows if row["split"] == "train"]
        validation = [row for row in rows if row["split"] == "validation"]
        if not train or len(validation) < 3:
            raise CalibrationError(
                f"{worker_id}: require training and at least 3 validation samples"
            )
        censored = sum(row["finish_reason"] == "length" for row in train)
        if censored and output_prior is None:
            raise CalibrationError(
                f"{worker_id}: censored training requires an explicit independent output prior"
            )
        prior = (
            output_prior
            if output_prior is not None
            else int(statistics.median(row["output_tokens"] for row in train))
        )
        # Missing samples disable this optional column, rather than imputing zero
        # pressure. Required running/waiting counters never permit missing data.
        use_kv_usage = all(row["kv_usage_fraction"] is not None for row in train)
        coefficients, disabled = fit_coefficients(train, use_kv_usage)
        ranges = {}
        fields = {
            "prompt_range": "prompt_tokens",
            "output_range": "output_tokens",
            "concurrency_range": "inflight",
            "backend_running_range": "backend_running",
            "backend_waiting_range": "backend_waiting",
        }
        for name, field in fields.items():
            ranges[name] = [
                min(row[field] for row in train),
                max(row[field] for row in train),
            ]
        fractions = [row["cached_tokens"] / row["prompt_tokens"] for row in train]
        ranges["cache_fraction_range"] = [min(fractions), max(fractions)]
        ranges["kv_usage_range"] = (
            [
                min(row["kv_usage_fraction"] for row in train),
                max(row["kv_usage_fraction"] for row in train),
            ]
            if use_kv_usage and coefficients[-1] > 0
            else [0.0, 1.0]
        )
        if not ranges["output_range"][0] <= prior <= ranges["output_range"][1]:
            raise CalibrationError(f"{worker_id}: output prior outside training domain")
        for row in rows:
            checks = [(row[field], ranges[name]) for name, field in fields.items()]
            checks.append(
                (
                    row["cached_tokens"] / row["prompt_tokens"],
                    ranges["cache_fraction_range"],
                )
            )
            if coefficients[-1] > 0:
                if row["kv_usage_fraction"] is None:
                    raise CalibrationError(f"{worker_id}: missing validation KV usage")
                checks.append((row["kv_usage_fraction"], ranges["kv_usage_range"]))
            if any(not bounds[0] <= value <= bounds[1] for value, bounds in checks):
                raise CalibrationError(f"{worker_id}: sample outside training domain")

        def predict(row, output=None):
            # A zero fitted coefficient permits unknown KV usage at inference.
            include_usage = use_kv_usage and row["kv_usage_fraction"] is not None
            result = float(np.dot(features(row, output, include_usage), coefficients))
            if not math.isfinite(result) or result <= 0:
                raise CalibrationError(
                    f"{worker_id}: nonpositive completion prediction"
                )
            return result

        for row in train:
            predict(row)
        actual, predicted, observed = [], [], []
        for row in validation:
            output = capped_output(prior, row["max_output_tokens"])
            if not ranges["output_range"][0] <= output <= ranges["output_range"][1]:
                raise CalibrationError(
                    f"{worker_id}: capped prior outside training domain"
                )
            actual.append(predict(row))
            predicted.append(predict(row, output))
            observed.append(row["completion_ms"])
        actual_error = error_summary(actual, observed)
        prior_error = error_summary(predicted, observed)
        accepted = max(actual_error["mape"], prior_error["mape"]) <= max_validation_mape
        fingerprint = next(iter(fingerprints))
        models[worker_id] = {
            "fingerprint": fingerprint or "",
            "calibration_version": f"{source}:{version}",
            "source": source,
            **ranges,
            "output_prior": prior,
            "coefficients": dict(zip(COEFFICIENTS, coefficients.tolist())),
        }
        reports[worker_id] = {
            "accepted": accepted,
            "train_samples": len(train),
            "train_groups": len({row["group_id"] for row in train}),
            "validation_groups": len({row["group_id"] for row in validation}),
            "identity_assumption": "endpoint" if fingerprint is None else "fingerprint",
            "censored_training_samples": censored,
            "output_prior_source": "explicit"
            if output_prior is not None
            else "uncensored_training_median",
            "disabled_columns": disabled,
            "kv_usage_missing_training_samples": sum(
                row["kv_usage_fraction"] is None for row in train
            ),
            "validation_actual_output": actual_error,
            "validation_output_prior": prior_error,
        }
    return {"ect_model": "completion_time", "completion_models": models}, {
        "schema_version": 1,
        "source": source,
        "calibration_version": version,
        "timing_boundary": "dispatch_to_confirmed_terminal",
        "max_validation_mape": max_validation_mape,
        "accepted": all(report["accepted"] for report in reports.values()),
        "workers": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="Router config, or model fragment without --base-config",
    )
    parser.add_argument(
        "--base-config",
        help="Existing Router JSON config to preserve while replacing completion models",
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--max-validation-mape", type=float, default=0.25)
    parser.add_argument(
        "--output-prior",
        type=int,
        help="Independently chosen prior; required for length-censored training",
    )
    args = parser.parse_args()
    paths = [args.input, args.output, args.report]
    if args.base_config:
        paths.append(args.base_config)
    if len({Path(path).resolve() for path in paths}) != len(paths):
        parser.error("input, output, report and base config must be different paths")
    try:
        base_config = {}
        if args.base_config:
            with open(args.base_config) as base_file:
                base_config = json.load(base_file)
            if not isinstance(base_config, dict):
                raise CalibrationError("base config must be a JSON object")
            try:
                json.dumps(base_config, allow_nan=False)
            except ValueError as error:
                raise CalibrationError(
                    "base config must contain finite JSON values"
                ) from error
        with open(args.input) as input_file:
            samples = [json.loads(line) for line in input_file if line.strip()]
        config, report = fit(
            samples, args.version, args.max_validation_mape, args.output_prior
        )
        write_json(args.report, report)
        if not report["accepted"]:
            parser.exit(
                2,
                "Validation rejected the calibration; config was not written. See report.\n",
            )
        # Replace the model map as a unit so uncalibrated historical workers
        # cannot survive a new calibration through a recursive merge.
        config = {**base_config, **config}
        write_json(args.output, config)
        print(
            json.dumps(
                {
                    "source": report["source"],
                    "workers": len(config["completion_models"]),
                    "accepted": True,
                }
            )
        )
    except (CalibrationError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"Calibration failed: {error}\n")


if __name__ == "__main__":
    main()
