import csv
import tempfile
import unittest
from pathlib import Path

from export_unmatched_training import unmatched
from merge_label_updates import merge_updates


class LabelUpdateTests(unittest.TestCase):
    def test_incremental_import_preserves_metadata_and_is_idempotent(self):
        original = [{"scope": "country", "country_code": "RU", "date": "2020-01-01",
                     "description": "Existing", "metadata": {"title": "Keep this"}}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "update.csv"
            with path.open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow([""] * 8)
                writer.writerow(["Kenya", "KE", "", "2024-11-07", "", "", "Telegram", "link"])
            merged, added, duplicates = merge_updates(original, [path])
            self.assertEqual(merged[:1], original)
            self.assertEqual(len(added), 1)
            self.assertEqual(duplicates, 0)
            self.assertEqual(added[0]["metadata"]["source"], "update.csv")
            again, added, duplicates = merge_updates(merged, [path])
            self.assertEqual(again, merged)
            self.assertEqual((len(added), duplicates), (0, 1))
            conflicting = [dict(merged[1], description="Another event")]
            with self.assertRaises(ValueError):
                merge_updates(conflicting, [path])

    def test_training_review_uses_country_and_global_windows(self):
        anomaly = {"start": "2024-01-03"}
        global_label = {"scope": "global", "date": "2024-01-01"}
        country_label = dict(global_label, scope="country", country_code="KE")
        self.assertEqual(unmatched([anomaly], [global_label], "KE", 3), [anomaly])
        self.assertEqual(unmatched([anomaly], [country_label], "KE", 3), [])
        self.assertEqual(unmatched([anomaly], [country_label], "RU", 3), [anomaly])


if __name__ == "__main__":
    unittest.main()
