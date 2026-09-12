#!/usr/bin/env python3
"""Fit ECT coefficients from explicit train/validation JSONL timing samples.

Uses unloaded phase timings for P/D, then jointly fits beta and Q against
completion timings. Does not scrape telemetry or infer phase times from TTFT.
"""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

import numpy as np
from scipy.optimize import nnls


class CalibrationError(ValueError):
    pass


FIELDS = {
    "sample_id",
    "group_id",
    "worker_id",
    "fingerprint",
    "source",
    "split",
    "prompt_tokens",
    "reusable_tokens",
    "output_tokens",
    "max_output_tokens",
    "inflight",
    "prefill_ms",
    "decode_ms",
    "completion_ms",
}


def validate_samples(samples):
    if not samples:
        raise CalibrationError("empty calibration input")
    ids, groups, sources = set(), {}, set()
    for row in samples:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise CalibrationError("sample fields do not match the calibration schema")
        for field in ("sample_id", "group_id", "worker_id", "fingerprint"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise CalibrationError(f"invalid {field}")
        if row["sample_id"] in ids:
            raise CalibrationError("duplicate sample_id")
        ids.add(row["sample_id"])
        if row["split"] not in ("train", "validation"):
            raise CalibrationError("split must be train or validation")
        if groups.setdefault(row["group_id"], row["split"]) != row["split"]:
            raise CalibrationError("group occurs in both train and validation")
        if row["source"] not in ("measured", "synthetic"):
            raise CalibrationError("source must be measured or synthetic")
        sources.add(row["source"])
        for field in (
            "prompt_tokens",
            "reusable_tokens",
            "output_tokens",
            "max_output_tokens",
            "inflight",
        ):
            if type(row[field]) is not int or not 0 <= row[field] <= 10_000_000:
                raise CalibrationError(f"invalid {field}")
        if (
            row["prompt_tokens"] == 0
            or row["output_tokens"] == 0
            or row["reusable_tokens"] >= row["prompt_tokens"]
            or row["output_tokens"] > row["max_output_tokens"]
        ):
            raise CalibrationError("inconsistent token counts")
        for field in ("prefill_ms", "decode_ms", "completion_ms"):
            value = row[field]
            if value is None and field != "completion_ms" and row["inflight"] > 0:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise CalibrationError(f"invalid {field}")
        if row["completion_ms"] <= 0:
            raise CalibrationError("completion_ms must be positive")
    if len(sources) != 1:
        raise CalibrationError("cannot mix synthetic and measured samples")


def prefill_features(row):
    length, cached = row["prompt_tokens"], row["reusable_tokens"]
    return [1.0, length - cached, length * length - cached * cached]


def decode_features(row, output=None):
    output = row["output_tokens"] if output is None else output
    return [1.0, output, row["prompt_tokens"] * output]


def fit_nonnegative(matrix, target, name):
    matrix, target = np.asarray(matrix, dtype=float), np.asarray(target, dtype=float)
    scale = np.linalg.norm(matrix, axis=0)
    if np.any(scale == 0) or not np.all(np.isfinite(scale)):
        raise CalibrationError(f"{name}: insufficient feature variation")
    normalized = matrix / scale
    if (
        np.linalg.matrix_rank(normalized) != normalized.shape[1]
        or np.linalg.cond(normalized) > 1e8
    ):
        raise CalibrationError(f"{name}: rank-deficient or ill-conditioned calibration")
    coefficients, _ = nnls(normalized, target, maxiter=1000)
    coefficients /= scale
    if not np.all(np.isfinite(coefficients)):
        raise CalibrationError(f"{name}: invalid fitted coefficients")
    return coefficients.tolist()


def error_summary(predicted, observed):
    errors = np.abs(np.asarray(predicted) - np.asarray(observed))
    relative = errors / observed
    return {
        "samples": len(observed),
        "mae_ms": float(np.mean(errors)),
        "mape": float(np.mean(relative)),
        "p95_relative_error": float(np.quantile(relative, 0.95)),
    }


def fit(samples, version, max_validation_mape=0.25):
    validate_samples(samples)
    if (
        not version.strip()
        or not math.isfinite(max_validation_mape)
        or max_validation_mape < 0
    ):
        raise CalibrationError("invalid calibration version or validation threshold")
    workers = defaultdict(list)
    for row in samples:
        workers[row["worker_id"]].append(row)
    models, reports = {}, {}
    source = samples[0]["source"]
    for worker_id, rows in sorted(workers.items()):
        fingerprints = {row["fingerprint"] for row in rows}
        if len(fingerprints) != 1:
            raise CalibrationError(f"{worker_id}: multiple serving fingerprints")
        train = [row for row in rows if row["split"] == "train"]
        validation = [row for row in rows if row["split"] == "validation"]
        baseline = [row for row in train if row["inflight"] == 0]
        if len(baseline) < 6 or len(validation) < 3:
            raise CalibrationError(
                f"{worker_id}: need at least 6 unloaded training and 3 validation samples"
            )
        if len({row["inflight"] for row in train}) < 2:
            raise CalibrationError(
                f"{worker_id}: need variation in pre-dispatch in-flight counts"
            )
        ranges = {
            "prompt_range": [
                min(row["prompt_tokens"] for row in baseline),
                max(row["prompt_tokens"] for row in baseline),
            ],
            "output_range": [
                min(row["output_tokens"] for row in baseline),
                max(row["output_tokens"] for row in baseline),
            ],
            "concurrency_range": [
                min(row["inflight"] for row in train),
                max(row["inflight"] for row in train),
            ],
        }
        for row in rows:
            for field, bounds in [
                ("prompt_tokens", ranges["prompt_range"]),
                ("output_tokens", ranges["output_range"]),
                ("inflight", ranges["concurrency_range"]),
            ]:
                if not bounds[0] <= row[field] <= bounds[1]:
                    raise CalibrationError(
                        f"{worker_id}: {field} outside calibration domain"
                    )
        prefill = fit_nonnegative(
            [prefill_features(row) for row in baseline],
            [row["prefill_ms"] for row in baseline],
            "prefill",
        )
        decode = fit_nonnegative(
            [decode_features(row) for row in baseline],
            [row["decode_ms"] for row in baseline],
            "decode",
        )

        def service(row, output=None):
            return float(
                np.dot(prefill_features(row), prefill)
                + np.dot(decode_features(row, output), decode)
            )

        # P/D are fixed by unloaded phase measurements. Fit beta and Q together
        # against residual completion time; never add a separately fitted queue.
        beta, queue = fit_nonnegative(
            [[service(row) * row["inflight"], 1.0] for row in train],
            [row["completion_ms"] - service(row) for row in train],
            "load/queue",
        )
        prior = int(statistics.median(row["output_tokens"] for row in train))
        actual_predictions, prior_predictions, observed = [], [], []
        for row in validation:
            output = min(prior, row["max_output_tokens"])
            if not ranges["output_range"][0] <= output <= ranges["output_range"][1]:
                raise CalibrationError(
                    f"{worker_id}: capped prior outside calibration domain"
                )
            actual_predictions.append(
                service(row) * (1 + beta * row["inflight"]) + queue
            )
            prior_predictions.append(
                service(row, output) * (1 + beta * row["inflight"]) + queue
            )
            observed.append(row["completion_ms"])
        actual_error = error_summary(actual_predictions, observed)
        prior_error = error_summary(prior_predictions, observed)
        accepted = max(actual_error["mape"], prior_error["mape"]) <= max_validation_mape
        models[worker_id] = {
            "fingerprint": next(iter(fingerprints)),
            "calibration_version": f"{source}:{version}",
            **ranges,
            "output_prior": prior,
            "prefill": prefill,
            "decode": decode,
            "beta": beta,
            "queue_ms": queue,
        }
        reports[worker_id] = {
            "accepted": accepted,
            "train_samples": len(train),
            "unloaded_phase_samples": len(baseline),
            "train_groups": len({row["group_id"] for row in train}),
            "validation_groups": len({row["group_id"] for row in validation}),
            "validation_actual_output": actual_error,
            "validation_output_prior": prior_error,
        }
    return {"cost_models": models}, {
        "schema_version": 1,
        "source": source,
        "calibration_version": version,
        "max_validation_mape": max_validation_mape,
        "accepted": all(report["accepted"] for report in reports.values()),
        "workers": reports,
    }


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            json.dump(value, output, indent=2, allow_nan=False)
            output.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output", required=True, help="Router config fragment containing cost_models"
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--max-validation-mape", type=float, default=0.25)
    args = parser.parse_args()
    if (
        Path(args.output).resolve()
        in {Path(args.input).resolve(), Path(args.report).resolve()}
        or Path(args.input).resolve() == Path(args.report).resolve()
    ):
        parser.error("input, output and report must be different paths")
    try:
        with open(args.input) as input_file:
            samples = [json.loads(line) for line in input_file if line.strip()]
        config, report = fit(samples, args.version, args.max_validation_mape)
        write_json(args.report, report)
        if not report["accepted"]:
            parser.exit(
                2,
                "Validation rejected the calibration; config was not written. See report.\n",
            )
        write_json(args.output, config)
        print(
            json.dumps(
                {
                    "source": report["source"],
                    "workers": len(config["cost_models"]),
                    "accepted": True,
                }
            )
        )
    except (CalibrationError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"Calibration failed: {error}\n")


if __name__ == "__main__":
    main()
