import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from calibrate_ect import CalibrationError, fit
from make_calibration_fixture import TRUTH, samples, timings


class CalibrationTest(unittest.TestCase):
    def test_recovers_phase_load_and_queue_coefficients(self):
        config, report = fit(samples(), "unit-test")
        self.assertTrue(report["accepted"])
        self.assertEqual(report["source"], "synthetic")
        for worker, model in config["cost_models"].items():
            for field in ("prefill", "decode"):
                for actual, expected in zip(model[field], TRUTH[worker][field]):
                    self.assertAlmostEqual(actual, expected, places=7)
            self.assertAlmostEqual(model["beta"], TRUTH[worker]["beta"], places=7)
            self.assertAlmostEqual(
                model["queue_ms"], TRUTH[worker]["queue_ms"], places=7
            )
            self.assertEqual(model["calibration_version"], "synthetic:unit-test")
            self.assertEqual(model["output_prior"], 128)
            self.assertLess(
                report["workers"][worker]["validation_output_prior"]["mape"], 1e-9
            )

    def test_leakage_duplicates_and_mixed_provenance_are_rejected(self):
        for kind in ("leakage", "duplicate", "fingerprint", "source"):
            rows = samples()
            if kind == "leakage":
                validation = next(row for row in rows if row["split"] == "validation")
                validation["group_id"] = rows[0]["group_id"]
            elif kind == "duplicate":
                rows[1]["sample_id"] = rows[0]["sample_id"]
            elif kind == "fingerprint":
                rows[0]["fingerprint"] = "different"
            else:
                rows[0]["source"] = "measured"
            with self.subTest(kind=kind), self.assertRaises(CalibrationError):
                fit(rows, "test")

    def test_bad_values_and_insufficient_coverage_are_rejected(self):
        for kind in (
            "nan",
            "negative",
            "no_phase",
            "tokens",
            "no_validation",
            "no_load",
            "rank",
            "domain",
            "unknown_field",
        ):
            rows = samples()
            if kind == "nan":
                rows[0]["completion_ms"] = float("nan")
            elif kind == "negative":
                rows[0]["decode_ms"] = -1
            elif kind == "no_phase":
                rows[0]["prefill_ms"] = None
            elif kind == "tokens":
                rows[0]["reusable_tokens"] = rows[0]["prompt_tokens"] + 1
            elif kind == "no_validation":
                rows = [row for row in rows if row["split"] == "train"]
            elif kind == "no_load":
                rows = [row for row in rows if row["inflight"] == 0]
            elif kind == "rank":
                for row in rows:
                    row["prompt_tokens"], row["reusable_tokens"] = 1024, 0
            elif kind == "domain":
                next(row for row in rows if row["split"] == "validation")[
                    "prompt_tokens"
                ] = 16384
            else:
                rows[0]["duration_seconds"] = 1
            with self.subTest(kind=kind), self.assertRaises(CalibrationError):
                fit(rows, "test")

    def test_output_prior_error_is_separate_from_service_fit_error(self):
        rows = samples()
        for row in rows:
            if row["split"] == "validation":
                row["output_tokens"] = 256
                row["prefill_ms"], row["decode_ms"], row["completion_ms"] = timings(row)
        _, report = fit(rows, "test", max_validation_mape=1e-6)
        self.assertFalse(report["accepted"])
        for worker in report["workers"].values():
            self.assertLess(worker["validation_actual_output"]["mape"], 1e-9)
            self.assertGreater(worker["validation_output_prior"]["mape"], 0.01)

    def test_output_limit_caps_the_routing_prior(self):
        rows = samples()
        for row in rows:
            if row["split"] == "validation":
                row["output_tokens"] = row["max_output_tokens"] = 64
                row["prefill_ms"], row["decode_ms"], row["completion_ms"] = timings(row)
        _, report = fit(rows, "test", max_validation_mape=1e-6)
        self.assertTrue(report["accepted"])

    def test_cli_writes_report_but_preserves_config_when_validation_fails(self):
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
                    str(Path(__file__).with_name("calibrate_ect.py")),
                    "--input",
                    str(data),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                    "--version",
                    "test",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(json.loads(report.read_text())["accepted"])
            self.assertEqual(output.read_text(), "existing calibration\n")


if __name__ == "__main__":
    unittest.main()
