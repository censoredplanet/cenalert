import tempfile
import unittest
from datetime import date
from pathlib import Path

from prepare_sample_archive import prepare


class ArchiveTests(unittest.TestCase):
    def test_selects_latest_complete_matching_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for batch, names in [("2026-01-10", ["2025-12", "2025-12"]),
                                 ("2026-01-11", ["2025-12", "2026-01"]),
                                 ("2026-01-12", ["2025-12", "2025-12"])]:
                for sample, name in enumerate(names):
                    path = root / f"output_{batch}" / f"sample{sample}" / "PK"
                    path.mkdir(parents=True)
                    (path / f"{name}_multiTimeline.csv").write_text("value,date\n5,2026-01-01\n")
            result = prepare(root, "PK", date(2026, 1, 11), 2)
            self.assertEqual(len(result["windows"]), 1)
            self.assertEqual(result["windows"][0]["collected_on"], "2026-01-10")
            self.assertEqual(len(result["windows"][0]["samples"]), 2)
            result = prepare(root, "PK", date(2026, 1, 12), 2)
            self.assertEqual(result["windows"][0]["collected_on"], "2026-01-12")

    def test_no_complete_windows_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no complete"):
                prepare(Path(directory), "PK", date(2026, 1, 12), 2)

    def test_coarse_samples_must_be_complete_and_from_same_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for batch in ["2026-01-10", "2026-01-11"]:
                for sample in range(2):
                    folder = root / f"output_{batch}" / f"sample{sample}" / "PK"
                    folder.mkdir(parents=True)
                    (folder / "2025-12_multiTimeline.csv").write_text("value,date\n5,2026-01-01\n")
                    if batch == "2026-01-10" or sample == 0:
                        (folder / "2025-12_coarseMultiTimeline.csv").write_text("value,date\n10,2026-01-01\n")
            early = prepare(root, "PK", date(2026, 1, 10), 2)["windows"][0]
            self.assertEqual(len(early["coarse"]["samples"]), 2)
            latest = prepare(root, "PK", date(2026, 1, 11), 2)["windows"][0]
            self.assertNotIn("coarse", latest)


if __name__ == "__main__":
    unittest.main()
