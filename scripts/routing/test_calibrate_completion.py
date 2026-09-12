import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from calibrate_completion import CalibrationError, COEFFICIENTS, features, fit


TRUTH = [500.0, 0.8, 6.0, 0.003, 0.5, 30.0, 50.0, 90.0, 200.0]


def duration(row):
    factor = {"g0": 1.0, "g1": 1.4, "g2": 0.7}[row["worker_id"]]
    return factor * sum(
        a * b
        for a, b in zip(
            features(row, use_kv_usage=row["kv_usage_fraction"] is not None), TRUTH
        )
    )


def samples():
    rng = random.Random(719)
    rows = []
    for worker in ("g0", "g1", "g2"):
        for split, count in (("train", 81), ("validation", 12)):
            for index in range(count):
                prompt = rng.choice((128, 256, 512))
                row = {
                    "sample_id": f"{worker}-{split}-{index}",
                    "group_id": f"{split}-prefix-{index}",
                    "worker_id": worker,
                    "fingerprint": "same-model",
                    "source": "synthetic",
                    "split": split,
                    "prompt_tokens": prompt,
                    "cached_tokens": int(prompt * rng.choice((0.0, 0.5, 1.0))),
                    "output_tokens": (8, 16, 32)[index % 3] if split == "train" else 16,
                    "max_output_tokens": 32,
                    "inflight": rng.choice((0, 1, 3)),
                    "backend_running": rng.choice((0, 2, 5)),
                    "backend_waiting": rng.choice((0, 1, 2)),
                    "kv_usage_fraction": rng.choice((0.0, 0.5, 1.0)),
                    "finish_reason": "stop",
                }
                row["completion_ms"] = duration(row)
                rows.append(row)
    return rows


class CompletionCalibrationTest(unittest.TestCase):
    def test_recovers_three_worker_models_with_cache_credit_and_external_load(self):
        config, report = fit(samples(), "fixture", max_validation_mape=1e-8)
        self.assertTrue(report["accepted"])
        self.assertEqual(config["ect_model"], "completion_time")
        for worker, model in config["completion_models"].items():
            factor = {"g0": 1.0, "g1": 1.4, "g2": 0.7}[worker]
            for name, truth in zip(COEFFICIENTS, TRUTH):
                self.assertAlmostEqual(
                    model["coefficients"][name], truth * factor, places=6
                )
            self.assertEqual(model["source"], "synthetic")
            self.assertEqual(model["output_prior"], 16)
            self.assertEqual(model["cache_fraction_range"], [0.0, 1.0])
            self.assertLess(
                report["workers"][worker]["validation_output_prior"]["mape"], 1e-9
            )

    def test_zero_columns_are_disabled_but_correlated_features_are_rejected(self):
        rows = samples()
        for row in rows:
            row["inflight"] = row["backend_running"] = row["backend_waiting"] = 0
            row["kv_usage_fraction"] = None
            row["completion_ms"] = duration(row)
        config, report = fit(rows, "idle")
        self.assertTrue(report["accepted"])
        for worker, model in config["completion_models"].items():
            self.assertEqual(model["coefficients"]["kv_usage_ms"], 0.0)
            self.assertIn(
                "backend_running_ms", report["workers"][worker]["disabled_columns"]
            )
        rows = samples()
        for row in rows:
            row["backend_running"] = row["inflight"]
        with self.assertRaisesRegex(CalibrationError, "confounded"):
            fit(rows, "confounded")

    def test_censoring_requires_explicit_prior_and_caps_validation_output(self):
        rows = samples()
        for row in rows:
            row["finish_reason"] = "length"
            row["max_output_tokens"] = row["output_tokens"]
            if row["split"] == "validation":
                row["output_tokens"] = row["max_output_tokens"] = 8
                row["completion_ms"] = duration(row)
        with self.assertRaisesRegex(CalibrationError, "censored training"):
            fit(rows, "censored")
        config, report = fit(rows, "censored", output_prior=16)
        self.assertTrue(report["accepted"])
        self.assertEqual(config["completion_models"]["g0"]["output_prior"], 16)
        self.assertEqual(report["workers"]["g0"]["output_prior_source"], "explicit")

    def test_endpoint_identity_remains_an_explicit_assumption(self):
        rows = samples()
        for row in rows:
            row["fingerprint"] = None
            row["attempt_id"] = row["sample_id"]
            row["request_id"] = "request"
            row["success"] = True
        config, report = fit(rows, "endpoint")
        self.assertEqual(config["completion_models"]["g0"]["fingerprint"], "")
        self.assertEqual(report["workers"]["g0"]["identity_assumption"], "endpoint")

    def test_bootstrap_samples_require_valid_cpu_evidence_and_one_choice(self):
        rows = samples()
        for row in rows:
            row["cache_location"] = "LocalCPUBackend"
            row["num_choices"] = 1
            row["fallback_reason"] = "missing_completion_model"
        self.assertTrue(fit(rows, "bootstrap")[1]["accepted"])
        for field, value in (
            ("cache_location", "native_gpu"),
            ("cache_location", None),
            ("fallback_reason", "invalid_lmcache_evidence"),
            ("fallback_reason", "invalid_backend_metrics_binding"),
            ("fallback_reason", {}),
            ("num_choices", 2),
            ("num_choices", True),
            ("num_choices", 1.0),
        ):
            invalid = copy.deepcopy(rows)
            invalid[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(
                CalibrationError
            ):
                fit(invalid, "invalid")

    def test_invalid_data_split_leaks_and_unobserved_domains_are_rejected(self):
        base = samples()
        for kind in (
            "duplicate",
            "group_leak",
            "identity",
            "source",
            "nan",
            "bool",
            "cache_overlong",
            "unknown_output",
            "failure",
            "extra_field",
            "domain",
            "unknown_kv_validation",
            "unknown_load",
            "prior_domain",
        ):
            rows = copy.deepcopy(base)
            validation = next(row for row in rows if row["split"] == "validation")
            if kind == "duplicate":
                rows[1]["sample_id"] = rows[0]["sample_id"]
            elif kind == "group_leak":
                validation["group_id"] = rows[0]["group_id"]
            elif kind == "identity":
                rows[0]["fingerprint"] = None
            elif kind == "source":
                rows[0]["source"] = "measured"
            elif kind == "nan":
                rows[0]["completion_ms"] = float("nan")
            elif kind == "bool":
                rows[0]["backend_running"] = True
            elif kind == "cache_overlong":
                rows[0]["cached_tokens"] = rows[0]["prompt_tokens"] + 1
            elif kind == "unknown_output":
                rows[0]["output_tokens"] = None
            elif kind == "failure":
                rows[0]["success"] = False
            elif kind == "extra_field":
                rows[0]["aggregate_prefill_ms"] = 200
            elif kind == "domain":
                validation["backend_running"] = 99
            elif kind == "unknown_kv_validation":
                validation["kv_usage_fraction"] = None
            elif kind == "unknown_load":
                rows[0]["backend_waiting"] = None
            else:
                validation["max_output_tokens"] = validation["output_tokens"] = 1
            with self.subTest(kind=kind), self.assertRaises(CalibrationError):
                fit(rows, "invalid")

    def test_validation_uses_real_output_and_routing_prior_as_separate_checks(self):
        rows = samples()
        for row in rows:
            if row["split"] == "validation":
                row["output_tokens"] = 32
                row["completion_ms"] = duration(row)
        _, report = fit(rows, "prior", max_validation_mape=1e-8)
        self.assertFalse(report["accepted"])
        for worker in report["workers"].values():
            self.assertLess(worker["validation_actual_output"]["mape"], 1e-9)
            self.assertGreater(worker["validation_output_prior"]["mape"], 0.01)

    def test_cli_does_not_replace_config_when_holdout_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            data, output, report = [
                Path(directory) / name
                for name in ("samples.jsonl", "config.json", "report.json")
            ]
            rows = samples()
            for row in rows:
                if row["split"] == "validation":
                    row["completion_ms"] *= 2
            data.write_text("".join(json.dumps(row) + "\n" for row in rows))
            output.write_text("existing calibration\n")
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("calibrate_completion.py")),
                    "--input",
                    str(data),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                    "--version",
                    "test",
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(json.loads(report.read_text())["accepted"])
            self.assertEqual(output.read_text(), "existing calibration\n")

    def test_cli_merges_base_config_and_protects_all_input_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            data, base, output, report = [
                Path(directory) / name
                for name in ("samples.jsonl", "base.json", "output.json", "report.json")
            ]
            base_config = {
                "lmcache": {
                    "identity_mode": "endpoint",
                    "workers": {"worker-url": {"instance_id": "g0"}},
                },
                "backend_metrics": {
                    "model": "local",
                    "urls": {"worker-url": "metrics-url"},
                },
                "header_env": {"Authorization": "ROUTING_TOKEN"},
                "ect_model": "decomposed",
                "completion_models": {"stale-worker": {"old": True}},
            }
            base.write_text(json.dumps(base_config, indent=2) + "\n")
            original_base = base.read_bytes()
            rows = samples()
            data.write_text("".join(json.dumps(row) + "\n" for row in rows))
            command = [
                sys.executable,
                str(Path(__file__).with_name("calibrate_completion.py")),
                "--input",
                str(data),
                "--base-config",
                str(base),
                "--output",
                str(output),
                "--report",
                str(report),
                "--version",
                "base-test",
            ]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            merged = json.loads(output.read_text())
            for field in ("lmcache", "backend_metrics", "header_env"):
                self.assertEqual(merged[field], base_config[field])
            self.assertEqual(merged["ect_model"], "completion_time")
            self.assertEqual(set(merged["completion_models"]), {"g0", "g1", "g2"})
            self.assertEqual(base.read_bytes(), original_base)
            original_output = output.read_bytes()
            for field in ("--input", "--output", "--report"):
                conflicting = command.copy()
                conflicting[conflicting.index(field) + 1] = str(base)
                rejected = subprocess.run(conflicting, text=True, capture_output=True)
                self.assertEqual(rejected.returncode, 2, rejected.stderr)
                self.assertIn("different paths", rejected.stderr)
                self.assertEqual(base.read_bytes(), original_base)
            for row in rows:
                if row["split"] == "validation":
                    row["completion_ms"] *= 2
            data.write_text("".join(json.dumps(row) + "\n" for row in rows))
            rejected = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(rejected.returncode, 2, rejected.stderr)
            self.assertFalse(json.loads(report.read_text())["accepted"])
            self.assertEqual(base.read_bytes(), original_base)
            self.assertEqual(output.read_bytes(), original_output)
            for invalid_base in ([], {"poll_interval_ms": float("inf")}):
                base.write_text(json.dumps(invalid_base))
                rejected = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(rejected.returncode, 2, rejected.stderr)
                self.assertIn("base config", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
