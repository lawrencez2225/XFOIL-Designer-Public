"""Progress reporting that reads artefacts and never touches the solver."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import watch_batch


def write_record(batch, name, status, *, age_seconds=0.0):
    run_dir = Path(batch) / "M0.1_Re500000" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "run.json"
    manifest.write_text(json.dumps({"status": status, "airfoil": name}))
    stamp = time.time() - age_seconds
    os.utime(manifest, (stamp, stamp))
    return manifest


class LatestBatchTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_picks_the_most_recently_touched_batch(self):
        old = self.root / "uiuc_old"
        new = self.root / "uiuc_new"
        old.mkdir()
        new.mkdir()
        stamp = time.time() - 500
        os.utime(old, (stamp, stamp))
        self.assertEqual(watch_batch.latest_batch(self.root), new)

    def test_ignores_files_that_are_not_directories(self):
        (self.root / "uiuc_file").write_text("x")
        (self.root / "uiuc_dir").mkdir()
        self.assertEqual(
            watch_batch.latest_batch(self.root), self.root / "uiuc_dir"
        )

    def test_missing_batches_are_reported(self):
        with self.assertRaises(SystemExit):
            watch_batch.latest_batch(self.root)


class ReportTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.batch = Path(self._tmp.name) / "uiuc_20260101_000000"
        self.batch.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_counts_each_status(self):
        write_record(self.batch, "a", "ok", age_seconds=30)
        write_record(self.batch, "b", "ok", age_seconds=20)
        write_record(self.batch, "c", "partial", age_seconds=10)
        write_record(self.batch, "d", "empty", age_seconds=0)
        summary = watch_batch.report(self.batch, expected=100)
        self.assertEqual(summary["done"], 4)
        self.assertEqual(
            summary["statuses"], {"ok": 2, "partial": 1, "empty": 1}
        )

    def test_usable_share_excludes_empty(self):
        for name, status in (
            ("a", "ok"),
            ("b", "partial"),
            ("c", "empty"),
            ("d", "empty"),
        ):
            write_record(self.batch, name, status)
        summary = watch_batch.report(self.batch)
        self.assertAlmostEqual(summary["usable_share"], 0.5)

    def test_remaining_is_measured_against_the_expected_total(self):
        write_record(self.batch, "a", "ok", age_seconds=60)
        write_record(self.batch, "b", "ok", age_seconds=0)
        summary = watch_batch.report(self.batch, expected=10)
        self.assertEqual(summary["remaining"], 8)
        self.assertIsNotNone(summary["eta_minutes"])

    def test_eta_is_absent_without_an_expected_total(self):
        write_record(self.batch, "a", "ok")
        summary = watch_batch.report(self.batch)
        self.assertIsNone(summary["remaining"])
        self.assertIsNone(summary["eta_minutes"])

    def test_an_empty_batch_reports_zero(self):
        summary = watch_batch.report(self.batch)
        self.assertEqual(summary["done"], 0)

    def test_an_unreadable_record_does_not_stop_the_report(self):
        path = write_record(self.batch, "broken", "ok")
        path.write_text("{ not json")
        summary = watch_batch.report(self.batch)
        self.assertEqual(summary["done"], 1)
        self.assertEqual(summary["statuses"], {"unreadable": 1})


class FormatTest(unittest.TestCase):
    def test_output_names_every_reported_quantity(self):
        summary = {
            "batch": "x",
            "done": 2,
            "expected": 10,
            "elapsed_minutes": 1.0,
            "rate_per_minute": 2.0,
            "remaining": 8,
            "eta_minutes": 4.0,
            "statuses": {"ok": 2},
            "usable_share": 1.0,
            "last_write": time.time(),
        }
        text = watch_batch.format_report(summary)
        for fragment in ("进度", "速率", "预计", "状态", "可用率"):
            self.assertIn(fragment, text)

    def test_a_stalled_batch_is_called_out(self):
        summary = {
            "batch": "x",
            "done": 2,
            "expected": 10,
            "elapsed_minutes": 1.0,
            "rate_per_minute": 2.0,
            "remaining": 8,
            "eta_minutes": 4.0,
            "statuses": {"ok": 2},
            "usable_share": 1.0,
            "last_write": time.time() - 600,
        }
        self.assertIn("可能已停止", watch_batch.format_report(summary))

    def test_an_empty_batch_still_formats(self):
        text = watch_batch.format_report(
            {"batch": "x", "done": 0, "expected": 10}
        )
        self.assertIn("进度", text)


if __name__ == "__main__":
    unittest.main()
