import argparse
import csv
import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_snapshot import build_snapshot, display_fields, read_series
from export_snapshot import export_snapshot


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "series").mkdir()
        (self.root / "params").mkdir()
        (self.root / "series/AA_stitched.csv").write_text("date,value\n2025-01-01,200\n")
        (self.root / "params/AA.json").write_text("{}")
        (self.root / "labels.json").write_text(json.dumps([
            {"date": "2025-01-01", "scope": "country", "country_code": "AA",
             "id": "example", "title": "Example", "metadata": {
                 "affected_services": ["Chat"], "sources": [{"name": "News", "url": "https://example.org"}]}}
        ]))
        self.args = argparse.Namespace(series_directory=self.root / "series",
            parameters_directory=self.root / "params", labels=self.root / "labels.json",
            rust_binary=self.root / "detector", output=self.root / "output.sqlite",
            countries=["AA"], topic="vpn", algorithm="ensemble", proximity_days=3)
        self.result = {"scale_factor": 2, "parameters": {},
            "annotated": [{"date": "2025-01-01", "value": 100, "anomaly": True}], "anomalies": [
            {"start": "2025-01-01", "end": "2025-01-01", "peak": "2025-01-01",
             "impact": 30, "score": 3, "residual": 0}]}

    def run_detector(self):
        return patch("build_snapshot.subprocess.run", return_value=argparse.Namespace(stdout=json.dumps(self.result)))

    def test_snapshot_and_complete_replacement(self):
        with self.run_detector():
            build_snapshot(self.args)
        with sqlite3.connect(self.args.output) as db:
            self.assertEqual(db.execute("SELECT value FROM series").fetchone(), (100,))
            self.assertEqual(db.execute("SELECT title, label_index FROM events").fetchone(), ("Example", 0))
            self.assertEqual(json.loads(db.execute("SELECT affected_services FROM events").fetchone()[0]), ["Chat"])
            self.assertEqual(db.execute("SELECT stable_id FROM labels").fetchone(), ("example",))
        self.result["anomalies"] = []
        with self.run_detector():
            build_snapshot(self.args)
        with sqlite3.connect(self.args.output) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM events").fetchone(), (0,))

    def test_detector_failure_preserves_previous_file(self):
        self.args.output.write_bytes(b"previous snapshot")
        with patch("build_snapshot.subprocess.run", side_effect=subprocess.CalledProcessError(1, "detect", stderr="invalid window")):
            with self.assertRaisesRegex(RuntimeError, "AA: detection failed .*invalid window"):
                build_snapshot(self.args)
        self.assertEqual(self.args.output.read_bytes(), b"previous snapshot")

    def test_csv_export(self):
        with self.run_detector():
            build_snapshot(self.args)
        output = self.root / "csv"
        export_snapshot(self.args.output, output)
        with (output / "AA/annotated.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["value"], "100")
        self.assertEqual(rows[0]["anomaly"], "true")
        self.assertEqual(rows[0]["threshold"], "")
        with (output / "AA/anomalies.csv").open() as stream:
            anomalies = list(csv.DictReader(stream))
        self.assertEqual(anomalies[0]["quartile"], "0")
        self.assertEqual(json.loads((output / "AA/detector.json").read_text())["algorithm"], "ensemble")
        with self.assertRaises(ValueError):
            export_snapshot(self.args.output, output)

    def test_invalid_annotated_data_does_not_replace_snapshot(self):
        self.args.output.write_bytes(b"previous snapshot")
        self.result["annotated"][0]["value"] = 999
        with self.run_detector(), self.assertRaises(ValueError):
            build_snapshot(self.args)
        self.assertEqual(self.args.output.read_bytes(), b"previous snapshot")

    def test_missing_parameters_preserves_previous_file(self):
        self.args.output.write_bytes(b"previous snapshot")
        self.args.countries = ["BB"]
        with self.assertRaises(ValueError):
            build_snapshot(self.args)
        self.assertEqual(self.args.output.read_bytes(), b"previous snapshot")

    def test_legacy_description_not_split_into_services(self):
        fields = display_fields({"metadata": {"affected_services": "News coverage, followed by debate"}})
        self.assertEqual(json.loads(fields[-2]), ["News coverage, followed by debate"])

    def test_duplicate_dates_rejected(self):
        path = self.root / "bad.csv"
        path.write_text("date,value\n2025-01-01,1\n2025-01-01,2\n")
        with self.assertRaises(ValueError):
            read_series(path, "AA", "vpn")


if __name__ == "__main__":
    unittest.main()
