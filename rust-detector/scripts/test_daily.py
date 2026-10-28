import fcntl
import json
import os
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch, Mock

import test_snapshot
from run_daily import arguments, normalize_series, run, stitch


class DailyTests(unittest.TestCase):
    def test_stitching_aligns_overlap_by_date(self):
        import pandas as pd
        from lib.stitching import stitch_two_windows_ratio_coarse
        old = pd.DataFrame({'date': ['2025-01-01', '2025-01-02'], 'value': [20., 40.]})
        new = pd.DataFrame({'date': ['2025-01-02', '2025-01-01', '2025-01-03'],
                            'value': [4., 2., 3.]})
        result = stitch_two_windows_ratio_coarse(old, new, None, None, set(old.date))
        self.assertEqual(result.value.to_list(), [20., 40., 30.])

    def setUp(self):
        self.fixture = test_snapshot.SnapshotTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        binary = self.root / "detector"
        binary.touch()
        binary.chmod(0o700)
        self.args = arguments(["--series-directory", str(self.root / "series"),
            "--parameters-directory", str(self.root / "params"),
            "--labels", str(self.root / "labels.json"), "--countries", "AA",
            "--rust-binary", str(binary), "--output-root", str(self.root / "runs"),
            "--through-date", "2025-01-01"])

    def test_publish_and_failed_rerun_preserves_current(self):
        with self.fixture.run_detector():
            run(self.args)
        current = self.args.output_root / "current"
        previous = current.resolve()
        self.assertTrue((current / "cenalert.sqlite").is_file())
        self.assertTrue((current / "detection/AA/anomalies.csv").is_file())
        self.assertTrue((current / "preview/manifest.json").is_file())
        manifest = json.loads((current / "run.json").read_text())
        self.assertEqual(manifest["coverage"]["AA"]["end"], "2025-01-01")
        self.assertFalse(manifest["notifications_sent"])
        with patch("run_daily.export_snapshot", side_effect=RuntimeError("export failed")), self.fixture.run_detector():
            with self.assertRaises(RuntimeError):
                run(self.args)
        self.assertEqual(current.resolve(), previous)
        failed = list(self.args.output_root.glob("failed-*"))
        self.assertEqual(len(failed), 1)
        self.assertTrue((failed[0] / "stitched/AA_stitched.csv").is_file())
        self.assertTrue((failed[0] / "cenalert.sqlite").is_file())
        with self.fixture.run_detector():
            run(self.args)
        self.assertNotEqual(current.resolve(), previous)
        self.assertTrue(previous.is_dir())

    def test_plan_does_not_create_output(self):
        self.args.plan = True
        run(self.args)
        self.assertFalse(self.args.output_root.exists())

    def test_concurrent_run_is_rejected(self):
        self.args.output_root.mkdir()
        with (self.args.output_root / ".daily.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "another daily run"):
                run(self.args)
        self.assertFalse((self.args.output_root / "current").exists())

    def test_existing_current_directory_is_not_replaced(self):
        (self.args.output_root / "current").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            run(self.args)
        self.assertTrue((self.args.output_root / "current").is_dir())

    def test_stale_input_does_not_publish(self):
        self.args.through_date = date(2025, 1, 2)
        with self.assertRaisesRegex(ValueError, "stale"):
            run(self.args)
        self.assertFalse((self.args.output_root / "current").exists())

    def test_missing_dates_are_rejected(self):
        (self.root / "series/AA_stitched.csv").write_text("date,value\n2025-01-01,1\n2025-01-03,2\n")
        self.args.through_date = date(2025, 1, 3)
        with self.assertRaisesRegex(ValueError, "missing dates"):
            normalize_series(self.args, self.args.series_directory, self.root / "normalized")

    def test_download_failure_does_not_publish(self):
        self.args.series_directory = None
        self.args.download = True
        def fail_after_one_sample(args, output):
            output.mkdir()
            (output / "sample.txt").write_text("downloaded sample")
            raise RuntimeError("download failed")
        with patch.dict(os.environ, {"GOOGLE_TRENDS_API_KEY": "test-only"}), patch(
                "run_daily.download", side_effect=fail_after_one_sample):
            with self.assertRaises(RuntimeError):
                run(self.args)
        self.assertFalse((self.args.output_root / "current").exists())
        retained = list(self.args.output_root.glob("failed-*/samples/sample.txt"))
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].read_text(), "downloaded sample")

    def test_bootstrap_multiple_overlapping_windows(self):
        self.args.sample_count = 2
        samples = self.root / "samples"
        for index in range(2):
            directory = samples / f"sample{index}/AA"
            directory.mkdir(parents=True)
            for suffix in ("multiTimeline", "coarseMultiTimeline"):
                (directory / f"2025-01_{suffix}.csv").write_text(
                    "date,value\n2025-01-01,2\n2025-01-02,4\n")
                (directory / f"2025-02_{suffix}.csv").write_text(
                    "date,value\n2025-01-02,2\n2025-01-03,4\n")
        output = self.root / "stitched"
        stitch(self.args, samples, output)
        from build_snapshot import read_series
        rows = read_series(output / "AA_stitched.csv", "AA", "vpn")
        self.assertEqual([row["rate"] for row in rows], [2, 4, 8])

    def test_bootstrap_and_incremental_stitch(self):
        self.args.sample_count = 2
        samples = self.root / "samples"
        for index in range(2):
            country = samples / f"sample{index}/AA"
            country.mkdir(parents=True)
            for kind in ("multiTimeline", "coarseMultiTimeline"):
                (country / f"2025-01_{kind}.csv").write_text("date,value\n2025-01-01,2\n2025-01-02,4\n")
        initial = self.root / "initial"
        stitch(self.args, samples, initial)
        self.assertTrue((initial / "AA_coarse_stitched.csv").is_file())
        self.args.previous_stitched = initial
        updated = self.root / "updated"
        stitch(self.args, samples, updated)
        self.assertEqual((initial / "AA_stitched.csv").read_text(), (updated / "AA_stitched.csv").read_text())
        (samples / "sample1/AA/2025-01_coarseMultiTimeline.csv").unlink()
        with self.assertRaisesRegex(ValueError, "unpaired"):
            stitch(self.args, samples, self.root / "invalid")

    def test_strict_download_errors_do_not_include_credentials(self):
        from lib.google_trends_utils import get_multi_timeline
        service = Mock()
        service.getGraph.side_effect = RuntimeError("secret-in-request-url")
        logger = Mock()
        with self.assertRaisesRegex(RuntimeError, "download failed") as raised:
            get_multi_timeline(service, logger, str(self.root), "test.csv",
                               "2025-01", "2025-02", "AA", attempts=1, strict=True)
        self.assertNotIn("secret", str(raised.exception))
        self.assertNotIn("secret", str(logger.mock_calls))

    def test_samples_to_real_detector(self):
        binary = Path(__file__).resolve().parents[1] / "target/release/cenalert-detector"
        if not binary.is_file():
            self.skipTest("cargo build --release is required for the local E2E test")
        self.args.rust_binary = binary
        self.args.series_directory = None
        self.args.samples_directory = self.root / "samples"
        self.args.sample_count = 2
        self.args.through_date = date(2025, 1, 2)
        for index in range(2):
            directory = self.args.samples_directory / f"sample{index}/AA"
            directory.mkdir(parents=True)
            for suffix in ("multiTimeline", "coarseMultiTimeline"):
                (directory / f"2025-01_{suffix}.csv").write_text("date,value\n2025-01-01,2\n2025-01-02,4\n")
        (self.root / "params/AA.json").write_text(json.dumps({"window": 10,
            "min_residual_ratio": 0.05, "efficiency": 0.05, "chebyshev_z": 3,
            "chebyshev_k": 5, "isolation_forest_min_score": 0.6, "lof_min_score": 1.5}))
        run(self.args)
        current = self.args.output_root / "current"
        self.assertTrue((current / "cenalert.sqlite").is_file())
        # The next daily update can consume both historical stitching files.
        self.args.previous_stitched = current / "stitched"
        for index in range(2):
            directory = self.args.samples_directory / f"sample{index}/AA"
            for suffix in ("multiTimeline", "coarseMultiTimeline"):
                (directory / f"2025-01_{suffix}.csv").write_text(
                    "date,value\n2025-01-01,2\n2025-01-02,4\n2025-01-03,8\n")
        self.args.through_date = date(2025, 1, 3)
        run(self.args)
        self.assertTrue((current / "stitched/AA_coarse_stitched.csv").is_file())
        self.assertEqual(json.loads((current / "run.json").read_text())["coverage"]["AA"]["observations"], 3)


if __name__ == "__main__":
    unittest.main()
