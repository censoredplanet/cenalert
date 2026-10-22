import unittest

from review_event_outcomes import distribution, event_outcomes


class EventOutcomeTests(unittest.TestCase):
    def test_quantiles_include_empty_singleton_and_interpolation(self):
        self.assertIsNone(distribution([])["median"])
        self.assertEqual(distribution([2])["q75"], 2)
        self.assertEqual(distribution([4, 1, 3, 2])["q25"], 1.75)

    def test_coverage_and_percentiles_use_separate_inputs(self):
        labels = [dict(date=d, scope="country") for d in
                  ["2025-01-01", "2025-02-01", "2025-03-01"]]
        raw = [dict(start=l["date"], end=l["date"], impact=v)
               for l, v in zip(labels, [2, 4])]
        rows = event_outcomes(labels, raw, raw[1:], [dict(impact=2), dict(impact=3)], 3)
        self.assertEqual([e["status"] for e in rows], ["filtered", "retained", "missed"])
        self.assertEqual(rows[0]["matches"][0]["training_percentile"], .5)
        self.assertEqual(rows[1]["matches"][0]["training_percentile"], 1)

    def test_coalesced_components_preserve_event(self):
        label = dict(date="2025-01-10", scope="country")
        raw = dict(start="2025-01-10", end="2025-01-10", impact=2)
        retained = dict(start="2025-01-01", component_starts=["2025-01-01", "2025-01-10"])
        row, = event_outcomes([label], [raw], [retained], [], 0)
        self.assertEqual(row["status"], "retained")
        self.assertIsNone(row["matches"][0]["training_percentile"])


if __name__ == "__main__":
    unittest.main()
