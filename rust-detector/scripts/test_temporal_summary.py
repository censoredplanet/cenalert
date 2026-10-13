"""Checks for manuscript summaries of saved temporal experiments."""
import json
import tempfile
import unittest
from pathlib import Path

from summarize_temporal_results import selected_point, write_reports


class TemporalSummaryTests(unittest.TestCase):
    def test_selected_point_uses_saved_parameters_and_rounded_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parameters = dict(window=31, z=3.2, k=6, min_residual_ratio=.1, efficiency=.05)
            (root / "knee-parameters.json").write_text(json.dumps(parameters))
            front = [([30.6, 3.2, 6, .1, .05], [7, -90])]
            self.assertEqual(selected_point(root, "knee", "original", front), [7, 90])
            with self.assertRaises(ValueError):
                selected_point(root, "knee", "original", [])

    def test_report_uses_supplied_counts_and_valid_latex(self):
        matrix = []
        for variant, selection in [("ensemble", "knee"), ("labels_ensemble", "knee"),
                                   ("labels_ensemble", "leftmost"), ("fixed_3sigma", "fixed")]:
            matrix.append(dict(variant=variant, selection=selection, anomaly_count=17,
                               country_labels_covered=4, country_labels_total=9,
                               label_matched_anomalies=6, unmatched_anomalies=11,
                               training=dict(anomaly_count=31, country_labels_covered=8,
                                             country_labels_total=8)))
        metrics = dict(configuration=dict(train_end="2020-12-31", evaluations=100,
                                          population_size=10, proximity_days=3),
                       common_country_seed_pairs=[("RU", 1)], matrix=matrix,
                       votes=[dict(variant="labels_ensemble", selection="knee", dense=10,
                                   croston=7)], failures=[], missed_test_labels=[],
                       test_start="2021-01-01", test_end="2021-12-31",
                       labels=dict(experiment=dict(total=19, scope={"country": 16, "global": 3}),
                                   training=dict(total=10)), review_queue=[])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_reports(root, metrics, {}, 1)
            latex = (root / "results.tex").read_text()
            self.assertIn("4/9", latex)
            self.assertIn("1 common countries", latex)
            self.assertIn(r"Table~\ref{tab:cenalert-matrix}", latex)
            self.assertNotIn("\r", latex)
            self.assertNotIn("\n\n", latex)
            self.assertIn("2021-12-31", (root / "results-zh.md").read_text())


if __name__ == "__main__":
    unittest.main()
