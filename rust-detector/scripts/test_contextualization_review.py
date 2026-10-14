"""Tests for held-out contextualization analysis."""
import unittest
import csv
import tempfile
from pathlib import Path

from review_contextualization import (
    coverage, evaluation_labels, notification_matches, ordering_cases, retain_by_impact,
    unmatched_notification_rows, write_unmatched_notifications,
    calibration_evidence, common_completed_countries, low_impact_rows,
    transport_scores, restore_null_scores,
    write_low_impact_review, write_paper_table,
    write_configuration_summary,
)


class ContextualizationReviewTests(unittest.TestCase):
    def test_summary_csv_reports_notifications_and_retained_labels(self):
        summary = {"original/knee": dict(variant="original", selection="knee", countries=2,
                   raw=10, coalesced=9, retained=7, raw_covered=3, retained_covered=2,
                   labels_total=4, low_impact_retained=1,
                   notification_matching={"5": dict(total=6, matched=4)})}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            write_configuration_summary(path, summary)
            with path.open() as stream:
                row, = csv.DictReader(stream)
        self.assertEqual(row["unmatched_notifications"], "2")
        self.assertEqual(row["country_labels_after"], "2")
        self.assertEqual(row["raw_anomalies"], "10")

    def test_low_impact_export_keeps_calibration_and_test_matches_separate(self):
        calibration_label = dict(date="2024-01-01", scope="global", description="Outage")
        test_label = dict(date="2026-01-01", scope="country", country_code="RU", description="Block")
        record = dict(variant="labels_ensemble", selection="knee", country="RU",
                      anomaly=dict(start="2026-01-01", end="2026-01-02", impact=.5),
                      threshold=.1, reason="above_calibrated_threshold",
                      threshold_sources=[dict(anomaly=dict(start="2024-01-01"), labels=[calibration_label])])
        matching = dict(individuals=[dict(country="RU", start="2026-01-01", labels=[test_label])], groups=[])
        analysis = dict(all_configurations=True, low_impact_retained=[record],
                        available_grouping={"labels_ensemble/knee": {"5": {"label_matching": matching}}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "low.csv"
            write_low_impact_review(path, analysis)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["calibration_events"], "Outage")
        self.assertEqual(rows[0]["test_label_matches"], "2026-01-01 Block")

    def test_paper_table_uses_post_filter_coverage(self):
        summary = {"fixed_3sigma/fixed": dict(variant="fixed_3sigma", selection="fixed", raw=8,
                   retained_covered=2, labels_total=3,
                   notification_matching={"5": dict(total=4, matched=3)})}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.tex"
            write_paper_table(path, summary)
            text = path.read_text()
        self.assertIn(r"Fixed $3\sigma$ & -- & 8 & 4 & 3 & 2/3", text)
        self.assertIn(r"\label{tab:contextualization}", text)

    def test_null_scores_round_trip_without_changing_measurements(self):
        original = [dict(country="RU", start="2024-01-01", score=None, impact=2),
                    dict(country="TR", start="2024-01-01", score=3, impact=1)]
        sent = transport_scores(original)
        self.assertEqual(sent[0]["score"], 0)
        self.assertIsNone(original[0]["score"])
        restore_null_scores(sent, original)
        self.assertEqual(sent, original)

    def test_common_countries_require_every_configuration(self):
        runs = {("RU", "a"): {"status": "completed"}, ("RU", "b"): {"status": "completed"},
                ("IN", "a"): {"status": "no_feasible_solution"}, ("IN", "b"): {"status": "completed"}}
        self.assertEqual(common_completed_countries(runs, ["RU", "IN"], ["a", "b"]), ["RU"])
        self.assertEqual(common_completed_countries(runs, ["RU", "IN"], ["b"]), ["RU", "IN"])

    def test_calibration_preserves_all_minimum_sources(self):
        country = dict(scope="country", country_code="RU", date="2024-01-05")
        global_ = dict(scope="global", date="2024-02-01", match_mode="exact")
        a = dict(start="2024-01-01", component_starts=["2024-01-01", "2024-01-05"], impact=.1)
        b = dict(start="2024-02-01", impact=.1)
        evidence = calibration_evidence([a, b], [country, global_], "RU", 0, .1)
        self.assertEqual(evidence["matched_training_anomalies"], 2)
        self.assertEqual([s["labels"] for s in evidence["threshold_sources"]], [[country], [global_]])
        empty = calibration_evidence([a], [country], "IN", 3, None)
        self.assertEqual(empty["threshold_sources"], [])
        with self.assertRaises(ValueError):
            calibration_evidence([a], [country], "RU", 0, None)

    def test_low_impact_rows_distinguish_missing_calibration(self):
        record = dict(country="IN", variant="labels_ensemble", selection="knee", threshold=None,
                      calibration=dict(matched_training_anomalies=0, threshold_sources=[]),
                      retained_anomalies=[dict(impact=.9), dict(impact=1)])
        rows = low_impact_rows([record], ["RU"])
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["in_common_countries"])
        self.assertEqual(rows[0]["reason"], "no_matched_training_anomaly")
        record["threshold"] = .5
        self.assertEqual(low_impact_rows([record], ["IN"])[0]["reason"], "above_calibrated_threshold")

    def test_unmatched_export_keeps_group_constituents(self):
        def anomaly(country, day):
            return dict(country=country, start=day, end=day, peak=day, impact=5)
        a = anomaly("RU", "2026-01-01")
        b = anomaly("TR", "2026-01-01")
        c = anomaly("IN", "2026-02-01")
        group = dict(date=a["start"], countries=["RU", "TR"], anomalies=[a, b])
        grouping = dict(individual_notifications=[c], grouped_notifications=[group])
        grouping["label_matching"] = notification_matches(grouping, {c: [] for c in ["RU", "TR", "IN"]}, 3)
        rows = unmatched_notification_rows(grouping)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({r["notification_id"] for r in rows}), 2)
        self.assertEqual([r["country"] for r in rows[:2]], ["RU", "TR"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unmatched.csv"
            write_unmatched_notifications(path, grouping)
            with path.open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
        grouping["label_matching"]["groups"][0]["members"][0]["labels"] = [{}]
        self.assertEqual(len(unmatched_notification_rows(grouping)), 1)

    def test_evaluation_updates_preserve_training(self):
        train = {"date": "2024-01-01", "scope": "global"}
        test = {"date": "2026-06-16", "scope": "country", "country_code": "IN"}
        self.assertEqual(evaluation_labels([train], [train, test], "2024-12-31"), [train, test])
        with self.assertRaises(ValueError):
            evaluation_labels([train], [test], "2024-12-31")

    def test_notification_matching_distinguishes_partial_groups(self):
        local = {"date": "2026-01-05", "scope": "country", "country_code": "RU"}
        anomaly = {"country": "RU", "start": "2026-01-01",
                   "component_starts": ["2026-01-01", "2026-01-05"]}
        other = {"country": "TR", "start": "2026-01-01"}
        grouping = {"individual_notifications": [anomaly],
                    "grouped_notifications": [{"date": "2026-01-01", "anomalies": [anomaly, other]}]}
        labels = {"RU": [local], "TR": [local]}
        result = notification_matches(grouping, labels, 0)
        self.assertEqual(result["matched"], 2)
        self.assertEqual(result["groups_any_matched"], 1)
        self.assertEqual(result["groups_all_members_matched"], 0)
        global_ = {"date": "2026-01-01", "scope": "global"}
        result = notification_matches(grouping, {c: [global_] for c in labels}, 0)
        self.assertEqual(result["groups_all_members_matched"], 1)
        result = notification_matches(grouping, {c: [] for c in labels}, 0)
        self.assertEqual(result["matched"], 0)

    def test_coverage_uses_component_starts(self):
        label = {"date": "2026-01-05", "scope": "country", "country_code": "RU"}
        merged = {"start": "2026-01-01", "component_starts": ["2026-01-01", "2026-01-05"]}
        self.assertEqual(coverage([merged], [label], 0), [label])

    def test_impact_boundary_and_missing_calibration(self):
        anomalies = [{"impact": n} for n in [1, 2, 3]]
        self.assertEqual(retain_by_impact(anomalies, 2), anomalies[1:])
        self.assertEqual(retain_by_impact(anomalies, None), anomalies)
        self.assertEqual(retain_by_impact(anomalies, 0), anomalies)

    def test_order_conflict_requires_strictly_closer_global_label(self):
        anomaly = {"start": "2026-07-14"}
        country = {"date": "2026-07-16", "scope": "country", "country_code": "IN"}
        global_ = {"date": "2026-07-14", "scope": "global"}
        self.assertEqual(len(ordering_cases([anomaly], [country, global_], "IN", 3)), 1)
        self.assertEqual(ordering_cases([anomaly], [country, global_], "RU", 3), [])
        global_["date"] = "2026-07-12"
        self.assertEqual(ordering_cases([anomaly], [country, global_], "IN", 3), [])


if __name__ == "__main__":
    unittest.main()
