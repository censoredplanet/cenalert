"""Tests for held-out contextualization analysis."""
import unittest
import csv
import tempfile
from pathlib import Path

from review_contextualization import (
    coverage, evaluation_labels, notification_matches, ordering_cases, retain_by_impact,
    unmatched_notification_rows, write_unmatched_notifications,
)


class ContextualizationReviewTests(unittest.TestCase):
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
