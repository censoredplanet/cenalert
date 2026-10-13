"""Tests for held-out contextualization analysis."""
import unittest

from review_contextualization import coverage, ordering_cases, retain_by_impact


class ContextualizationReviewTests(unittest.TestCase):
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
