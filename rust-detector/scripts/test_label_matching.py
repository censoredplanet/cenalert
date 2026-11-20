from datetime import date
import unittest

from label_matching import label_applies, label_country_code, label_matches, matched_label


class LabelMatchingTests(unittest.TestCase):
    def test_country_scope(self):
        label = {"scope": "country", "metadata": {"country_code": " aa "}}
        self.assertEqual(label_country_code(label), "AA")
        self.assertTrue(label_applies(label, "AA"))
        self.assertFalse(label_applies(label, "BB"))
        label["additional_matching_countries"] = ["BB"]
        self.assertTrue(label_applies(label, "BB"))
        with self.assertRaises(ValueError):
            label_country_code({"country_code": "AA", "metadata": {"country_code": "BB"}})

    def test_date_windows_and_tie_breaking(self):
        country = {"date": "2025-01-01", "scope": "country", "country_code": "AA"}
        global_event = {"date": "2025-01-01", "scope": "global"}
        self.assertTrue(label_matches(country, date(2025, 1, 4), 3))
        self.assertFalse(label_matches(global_event, date(2025, 1, 4), 3))
        self.assertTrue(label_matches(global_event, date(2025, 1, 2), 3))
        result = matched_label({"start": "2025-01-02"}, [global_event, country], "AA", 3)
        self.assertEqual(result["label"], country)
        self.assertIsNone(matched_label({"start": "2025-02-01"}, [country], "AA", 3))


if __name__ == "__main__":
    unittest.main()
