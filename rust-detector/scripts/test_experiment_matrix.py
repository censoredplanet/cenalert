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
            {"scope": "country", "country_code": "RU", "date": "2024-07-19"},
            {"scope": "country", "country_code": "RU", "date": "2025-07-20"}
        ]))
        self.binary = Path(__file__).resolve().parents[1] / "target/release/cenalert-detector"
        self.args = argparse.Namespace(
            series_directory=self.series, countries=None, labels=self.labels,
            rust_binary=self.binary, output_directory=self.root / "output", seeds=[42],
            evaluations=20, population_size=10, proximity_days=3, threads_per_tuning=1,
            selections=["knee", "leftmost"], resume=False,
            train_end=date(2024, 12, 31), test_end=date(2026, 12, 31), baseline_window=60,
        )
        self.config = batch.Config(self.labels, self.binary, self.root, 20, 10, 3, 1, 1, 42)

    def add_series(self, country="RU", zero=False):
        rows = [{"country": country, "date": str(date(2024, 1, 1) + timedelta(days=i)),
                 "rate": 0.0 if zero else (100.0 if i in (200, 566) else 10.0 + i % 3)}
                for i in range(606)]
        (self.series / f"{country}.json").write_text(json.dumps(rows))

    def test_constraints_only_reach_labeled_variants(self):
        for variant, (algorithm, constrained) in matrix.VARIANTS.items():
            if variant == "fixed_3sigma":
                continue
            with self.subTest(variant=variant), patch.object(batch, "run_command") as run:
                config = matrix.replace(self.config, algorithm=algorithm)
                batch.tune(config, Path("input.json"), Path("front.json"), "RU", constrained)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--algorithm") + 1], algorithm)
                self.assertEqual("--labels" in command, constrained)
                self.assertEqual("--proximity-days" in command, constrained)
                self.assertEqual(command[command.index("--seed") + 1],
                                 str(batch.country_seed(42, "RU")))

    def test_selected_variants_are_snapshotted(self):
        self.add_series()
        self.args.variants = ["labels", "labels_ensemble"]
        matrix.prepare_inputs(self.args)
        manifest = json.loads((self.args.output_directory / "experiment.json").read_text())
        self.assertEqual(set(manifest["variants"]), {"labels", "labels_ensemble"})
        self.assertEqual(manifest["ensemble_member_balance"], {
            "unit": "independent_dense_anomaly_intervals", "period": "training",
            "maximum_share": 0.5, "zero_total_feasible": True,
        })
        self.args.resume = True
        matrix.prepare_inputs(self.args)
        self.args.variants = ["labels"]
        with self.assertRaisesRegex(ValueError, "settings changed"):
            matrix.prepare_inputs(self.args)

    def test_warm_start_is_scoped_snapshotted_and_forwarded(self):
        self.add_series()
        initial = self.root / "initial.json"
        initial.write_text('[[37, 0.002, 0.07, 3.1, 15, 0.75, 1.3]]')
        self.args.initial_parameters = initial
        with self.assertRaisesRegex(ValueError, "one country and one ensemble variant"):
            matrix.prepare_inputs(self.args)
        self.args.variants = ["labels_ensemble"]
        matrix.prepare_inputs(self.args)
        saved = self.args.output_directory / "inputs/initial-parameters.json"
        self.assertEqual(json.loads(saved.read_text()), json.loads(initial.read_text()))
        self.args.resume = True
        matrix.prepare_inputs(self.args)
        with patch.object(batch, "run_command") as run:
            config = matrix.replace(self.config, initial_parameters=saved)
            batch.tune(config, Path("input.json"), Path("front.json"), "RU", True)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--initial-parameters") + 1], str(saved))
        initial.write_text('[[38, 0.002, 0.07, 3.1, 15, 0.75, 1.3]]')
        with self.assertRaisesRegex(ValueError, "settings changed"):
            matrix.prepare_inputs(self.args)

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
                    self.assertEqual(len(result["selections"]),
                                     1 if result["variant"] == "fixed_3sigma" else 2)
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
        (self.root / "inputs").mkdir()
        (self.root / "inputs/training-labels.json").write_text(self.labels.read_text())
        with patch.object(matrix, "tune", side_effect=RuntimeError(
                "tuning found no parameter set that detects every label")), \
                patch.object(matrix, "detect") as detect:
            result = matrix.run_variant(self.config, "RU", "labels",
                                        dict(observations=2, zero_observations=0,
                                             training_observations=1, test_observations=1,
                                             training_maximum=10),
                                        ["knee"], False)
        self.assertEqual(result["status"], "no_feasible_solution")
        detect.assert_not_called()

    def test_training_inputs_exclude_future_labels_and_values(self):
        self.add_series()
        matrix.prepare_inputs(self.args)
        inputs = self.args.output_directory / "inputs"
        training_labels = json.loads((inputs / "training-labels.json").read_text())
        self.assertEqual([label["date"] for label in training_labels], ["2024-07-19"])
        training = json.loads((inputs / "RU-train.json").read_text())
        self.assertEqual(training[-1]["date"], "2024-12-31")

    def test_member_balance_failure_is_infeasible_for_both_ensemble_variants(self):
        (self.root / "inputs").mkdir()
        (self.root / "inputs/training-labels.json").write_text(self.labels.read_text())
        for variant in ("ensemble", "labels_ensemble"):
            with self.subTest(variant=variant), patch.object(matrix, "tune", side_effect=RuntimeError(
                    "tuning found no ensemble parameter set satisfying member-balance and applicable label constraints")), \
                    patch.object(matrix, "detect") as detect:
                result = matrix.run_variant(self.config, "RU", variant,
                    dict(observations=2, zero_observations=0, training_observations=1,
                         test_observations=1, training_maximum=10), ["knee"], False)
            self.assertEqual(result["status"], "no_feasible_solution")
            detect.assert_not_called()

    def test_balance_failure_in_diagnostic_search_is_reported(self):
        with patch.object(batch, "tune", side_effect=RuntimeError(
                "tuning found no ensemble parameter set satisfying member-balance and applicable label constraints")):
            result = batch.diagnose_infeasible(self.config, "RU", Path("input.json"), [], self.root)
        self.assertEqual(result["diagnostic_status"], "no_feasible_solution")
        self.assertNotIn("labels_never_detected", result)

    def test_fixed_baseline_uses_only_previous_values(self):
        rows = [{"date": str(date(2025, 1, 1) + timedelta(days=i)), "rate": value}
                for i, value in enumerate([0, 0, 0, 10, 10000])]
        first = matrix.fixed_three_sigma(rows[:4], 3, 100)["anomalies"][0]
        later = matrix.fixed_three_sigma(rows, 3, 100)["anomalies"][0]
        self.assertEqual(first["start"], later["start"])
        self.assertIsNone(first["score"])
        self.assertEqual(first["impact"], 10)

    def test_metrics_exclude_training_starts_and_training_labels(self):
        anomalies = [dict(start="2024-12-31", impact=10),
                     dict(start="2025-01-02", impact=20)]
        labels = [dict(scope="country", country_code="RU", date="2024-12-31"),
                  dict(scope="country", country_code="RU", date="2025-01-01")]
        result = matrix.evaluate_period(anomalies, labels, "RU", "2025-01-01", "2025-12-31", 3)
        self.assertEqual(result["anomaly_count"], 1)
        self.assertEqual(result["country_labels_total"], 1)
        self.assertEqual(result["country_labels_covered"], 1)
        self.assertEqual(result["total_impact"], 20)

    def test_partial_failed_runs_are_excluded_from_comparisons(self):
        run = dict(country="RU", seed=42, effective_seed=297, variant="original",
                   algorithm="chebyshev", constrained=False, status="error",
                   detail="selection failed", selections=[{"selection": "knee"}])
        matrix.export_results(self.root, [run], 3)
        self.assertEqual(len((self.root / "anomalies.csv").read_text().splitlines()), 1)
        self.assertIn("selection failed", (self.root / "summary.csv").read_text())


if __name__ == "__main__":
    unittest.main()
