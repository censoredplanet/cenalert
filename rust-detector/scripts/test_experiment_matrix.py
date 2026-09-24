"""Regression checks for experimental isolation, resume, and CLI integration."""

import argparse
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import batch_ensemble_tuning as batch
import run_experiment_matrix as matrix


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.series = self.root / "series"
        self.series.mkdir()
        self.labels = self.root / "labels.json"
        self.labels.write_text(json.dumps([
            {"scope": "country", "country_code": "RU", "date": "2025-07-20"}
        ]))
        self.binary = Path(__file__).resolve().parents[1] / "target/release/cenalert-detector"
        self.args = argparse.Namespace(
            series_directory=self.series, countries=None, labels=self.labels,
            rust_binary=self.binary, output_directory=self.root / "output", seeds=[42],
            evaluations=20, population_size=10, proximity_days=3, threads_per_tuning=1,
            selections=["knee", "leftmost"], resume=False,
        )
        self.config = batch.Config(self.labels, self.binary, self.root, 20, 10, 3, 1, 1, 42)

    def add_series(self, country="RU", zero=False):
        rows = [{"country": country, "date": str(date(2025, 1, 1) + timedelta(days=i)),
                 "rate": 0.0 if zero else (100.0 if i == 200 else 10.0 + i % 3)}
                for i in range(240)]
        (self.series / f"{country}.json").write_text(json.dumps(rows))

    def test_constraints_only_reach_labeled_variants(self):
        for variant, (algorithm, constrained) in matrix.VARIANTS.items():
            with self.subTest(variant=variant), patch.object(batch, "run_command") as run:
                config = matrix.replace(self.config, algorithm=algorithm)
                batch.tune(config, Path("input.json"), Path("front.json"), "RU", constrained)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--algorithm") + 1], algorithm)
                self.assertEqual("--labels" in command, constrained)
                self.assertEqual("--proximity-days" in command, constrained)
                self.assertEqual(command[command.index("--seed") + 1],
                                 str(batch.country_seed(42, "RU")))

    def test_resume_rejects_changed_settings_labels_and_series(self):
        self.add_series()
        matrix.prepare_inputs(self.args)
        self.args.resume = True
        matrix.prepare_inputs(self.args)
        self.args.evaluations = 30
        with self.assertRaisesRegex(ValueError, "settings changed"):
            matrix.prepare_inputs(self.args)
        self.args.evaluations = 20
        original = self.labels.read_text()
        self.labels.write_text('[{"scope":"global","date":"2025-01-01"}]')
        with self.assertRaisesRegex(ValueError, "labels changed"):
            matrix.prepare_inputs(self.args)
        self.labels.write_text(original)
        self.add_series(zero=True)
        with self.assertRaisesRegex(ValueError, "series changed"):
            matrix.prepare_inputs(self.args)

    def test_zero_anomaly_front_is_a_valid_result(self):
        front, parameters = self.root / "front.json", self.root / "params.json"
        front.write_text('[[[30, 3, 5, 0.1, 0.05], [0, 0]]]')
        message = matrix.select_parameters(matrix.replace(self.config, algorithm="chebyshev"),
                                           front, parameters, "knee")
        self.assertIn("zero-anomaly", message)
        self.assertEqual(json.loads(parameters.read_text())["min_residual_ratio"], 0.1)

    def test_real_four_group_workflow_and_resume(self):
        self.add_series()
        self.add_series("SS", zero=True)
        countries, details = matrix.prepare_inputs(self.args)
        output = self.args.output_directory
        config = matrix.replace(self.config, work_directory=output,
                                labels=output / "inputs/labels.json",
                                rust_binary=output / "inputs/cenalert-detector")
        results = [matrix.run_variant(config, country, variant, details[country],
                                     self.args.selections, False)
                   for country in countries for variant in matrix.VARIANTS]
        for result in results:
            with self.subTest(country=result["country"], variant=result["variant"]):
                self.assertEqual(result["status"], "skipped_all_zero"
                                 if result["country"] == "SS" else "completed", result)
                if result["country"] == "RU":
                    self.assertEqual(len(result["selections"]), 2)
                    for selected in result["selections"]:
                        self.assertGreaterEqual(selected["anomaly_count"], 1)
                        self.assertEqual(selected["country_labels_covered"], 1)
        matrix.export_results(output, results, 3)
        for name in ("summary.csv", "anomalies.csv", "label-coverage.csv",
                     "detector-differences.csv", "report.md"):
            self.assertTrue((output / name).is_file())
        with patch.object(matrix, "tune", side_effect=AssertionError("unexpected retuning")):
            resumed = matrix.run_variant(config, "RU", "original", details["RU"],
                                         self.args.selections, True)
        self.assertEqual(resumed["status"], "completed")

    def test_infeasible_is_recorded_without_detection(self):
        with patch.object(matrix, "tune", side_effect=RuntimeError(
                "tuning found no parameter set that detects every label")), \
                patch.object(matrix, "detect") as detect:
            result = matrix.run_variant(self.config, "RU", "labels",
                                        dict(observations=2, zero_observations=0),
                                        ["knee"], False)
        self.assertEqual(result["status"], "no_feasible_solution")
        detect.assert_not_called()

    def test_partial_failed_runs_are_excluded_from_comparisons(self):
        run = dict(country="RU", seed=42, effective_seed=297, variant="original",
                   algorithm="chebyshev", constrained=False, status="error",
                   detail="selection failed", selections=[{"selection": "knee"}])
        matrix.export_results(self.root, [run], 3)
        self.assertEqual(len((self.root / "anomalies.csv").read_text().splitlines()), 1)
        self.assertIn("selection failed", (self.root / "summary.csv").read_text())


if __name__ == "__main__":
    unittest.main()
