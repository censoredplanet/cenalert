import json
import unittest
from datetime import date

from preview_notifications import event_status, metric_text, preview
import test_snapshot


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_snapshot.SnapshotTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_preview_records_matching_files_without_sending(self):
        with self.fixture.run_detector():
            from build_snapshot import build_snapshot
            build_snapshot(self.fixture.args)
        output = self.fixture.root / "preview"
        preview(self.fixture.args.output, output, date(2025, 1, 2))
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(len(manifest["messages"]), 1)
        message = manifest["messages"][0]
        self.assertEqual(message["status"], "started")
        self.assertIn("Ensemble votes at onset: 3/3", (output / message["text"]).read_text())
        self.assertIsNone(message["image"])
        with self.assertRaises(ValueError):
            preview(self.fixture.args.output, output, date(2025, 1, 2))

    def test_stale_snapshot_has_no_messages(self):
        from build_snapshot import build_snapshot
        with self.fixture.run_detector():
            build_snapshot(self.fixture.args)
        output = self.fixture.root / "stale"
        preview(self.fixture.args.output, output, date(2025, 1, 3))
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["messages"], [])
        self.assertEqual(manifest["stale_countries"], ["AA"])


class NotificationLogicTests(unittest.TestCase):
    def test_status_does_not_duplicate_new_and_ongoing(self):
        anomaly = {"start": "2025-01-01", "end": "2025-01-02"}
        self.assertEqual(event_status(anomaly, date(2025, 1, 2), date(2025, 1, 2)), "started")
        self.assertEqual(event_status(anomaly, date(2025, 1, 3), date(2025, 1, 2)), "ongoing")
        self.assertEqual(event_status(anomaly, date(2025, 1, 4), date(2025, 1, 3)), "ended")
        self.assertIsNone(event_status(anomaly, date(2025, 1, 4), date(2025, 1, 2)))

    def test_sparse_and_non_chebyshev_scores(self):
        self.assertEqual(metric_text({"sparse": True, "residual": 5}, "ensemble"), "Croston residual: 5.00")
        self.assertNotIn("probability", metric_text({"score": 2}, "lof"))

    def test_actual_member_votes_are_displayed(self):
        anomaly = {"ensemble_vote": {"vote_count": 2,
            "chebyshev": {"voted": True}, "isolation_forest": {"voted": False},
            "lof": {"voted": True}}}
        self.assertEqual(metric_text(anomaly, "ensemble"),
                         "Ensemble votes at onset: 2/3 (chebyshev, lof)")


if __name__ == "__main__":
    unittest.main()
