"""Tests for held-out contextualization analysis."""
import unittest

from review_contextualization import (
    coverage, evaluation_labels, notification_matches, ordering_cases, retain_by_impact,
)


class ContextualizationReviewTests(unittest.TestCase):
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
