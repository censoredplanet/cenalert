import unittest

from batch_ensemble_tuning import label_applies, label_country_code, label_source_summary, required_labels


class LabelCompatibilityTests(unittest.TestCase):
    def test_source_summary_accepts_both_formats(self):
        self.assertEqual(label_source_summary({"metadata": {"source": "Other"}}), "Other")
        self.assertEqual(label_source_summary({"metadata": {"sources": [
            {"name": "Example News", "url": "https://example.com"},
            {"name": "Example News", "url": "https://example.com/2"},
            {"url": "https://example.org"}]} }), "Example News; https://example.org")

    def test_nested_country_only_applies_to_its_owner(self):
        label = {"date": "2025-01-10", "scope": "country",
                 "metadata": {"country_code": " aa ", "sources": []}}
        self.assertEqual(label_country_code(label), "AA")
        self.assertTrue(label_applies(label, "AA"))
        self.assertFalse(label_applies(label, "BB"))
        self.assertEqual(required_labels([label], "AA", "2025-01-01", "2025-02-01", 3), [label])
        self.assertEqual(required_labels([label], "BB", "2025-01-01", "2025-02-01", 3), [])

    def test_conflicting_and_invalid_codes_fail(self):
        for value in ("BB", "", []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                label_country_code({"country_code": "AA", "metadata": {"country_code": value}})

    def test_agreeing_codes_and_optional_metadata(self):
        self.assertEqual(label_country_code({"country_code": "AA", "metadata": {"country_code": "aa"}}), "AA")
        self.assertEqual(label_country_code({"country_code": "AA"}), "AA")
        self.assertIsNone(label_country_code({}))


if __name__ == "__main__":
    unittest.main()
