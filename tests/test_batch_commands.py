import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xfoil_mac import workflows as x

from tests.helpers import GEOMETRY, fake_solver


class BatchCommandTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        binary = self.root / "xfoil"
        binary.write_text("fake executable fingerprint")
        self.app = x.AppPaths(
            self.root,
            self.root,
            binary,
            self.root,
            self.root / "runs",
            self.root / "single",
            self.root / "batch",
        )
        self.database = self.root / "database"
        self.database.mkdir()
        for name in ("a", "b"):
            (self.database / f"{name}.dat").write_text(GEOMETRY)
        for target in ("write_geometry_plot", "write_comparison"):
            patcher = patch.object(x, target)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(
            x,
            "write_cp_polar_outputs",
            return_value={"result_plots": [], "pressure_vectors": []},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_batch(self, **overrides):
        options = dict(
            app=self.app,
            folder=self.database,
            reynolds=1e6,
            mach=0.1,
            iterations=100,
            alpha_start=0,
            alpha_end=1,
            alpha_step=1,
            out_dir=self.root / "output",
            force=False,
            headless=True,
            retries=0,
        )
        options.update(overrides)
        return x.run_batch(**options)

    def test_crashed_case_does_not_stop_next_case_and_is_checkpointed(self):
        def execute(*args, **kwargs):
            if args[2].parent.parent.name == "a":
                return "solver_error", 1
            self.assertTrue((self.root / "output/a/run.json").exists())
            self.assertTrue((self.root / "output/summary.csv").exists())
            return fake_solver(*args, **kwargs)

        with (
            patch("xfoil_mac.execution.execute", side_effect=execute),
            patch.object(x, "start_xquartz") as quartz,
        ):
            summary = self.run_batch()
        quartz.assert_not_called()
        with summary.open() as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual(
            [r["status"] for r in records], ["solver_error", "ok"]
        )
        self.assertTrue((self.root / "output/b/airfoil_input.dat").is_file())
        self.assertTrue((self.root / "output/b/cp_alpha_1.txt").is_file())

    def test_invalid_geometry_is_recorded_and_remaining_case_runs(self):
        (self.database / "a.dat").write_text("broken\n1 0\n")
        with patch("xfoil_mac.execution.execute", side_effect=fake_solver):
            summary = self.run_batch()
        with summary.open() as handle:
            statuses = [r["status"] for r in csv.DictReader(handle)]
        self.assertEqual(statuses, ["failed", "ok"])
        self.assertTrue((self.root / "output/a.failure.json").is_file())

    def test_resume_skips_complete_cases_and_avoids_summary_duplicates(
        self,
    ):
        with patch("xfoil_mac.execution.execute", side_effect=fake_solver):
            self.run_batch()
        with patch("xfoil_mac.execution.execute") as execute:
            summary = self.run_batch(resume=True)
        execute.assert_not_called()
        with summary.open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_existing_results_are_not_overwritten_without_force(self):
        with patch("xfoil_mac.execution.execute", side_effect=fake_solver):
            self.run_batch()
        original = (self.root / "output/a/polar.txt").read_bytes()
        with self.assertRaises(ValueError):
            self.run_batch()
        self.assertEqual(
            original, (self.root / "output/a/polar.txt").read_bytes()
        )

    def test_resume_rejects_changed_input_without_modifying_saved_polar(self):
        with patch("xfoil_mac.execution.execute", side_effect=fake_solver):
            self.run_batch()
        polar = self.root / "output/a/polar.txt"
        original = polar.read_bytes()
        (self.database / "a.dat").write_text(GEOMETRY.replace("0.08", "0.09"))
        with patch("xfoil_mac.execution.execute") as execute:
            self.run_batch(resume=True)
        execute.assert_not_called()
        self.assertEqual(original, polar.read_bytes())
        failure = json.loads((self.root / "output/a.failure.json").read_text())
        self.assertIn("Cannot resume", failure["error"])

    def test_colliding_file_stems_have_separate_output_directories(self):
        (self.database / "a.txt").write_text(GEOMETRY)
        with patch("xfoil_mac.execution.execute", side_effect=fake_solver):
            self.run_batch()
        manifest = json.loads((self.root / "output/batch.json").read_text())
        self.assertEqual(len(set(manifest["cases"])), 3)
        self.assertTrue(
            all(
                (self.root / "output" / name / "polar.txt").exists()
                for name in manifest["cases"]
            )
        )

    def test_interrupted_case_keeps_data_and_resume_continues_remaining_cases(
        self,
    ):
        def interrupt(*args, **kwargs):
            fake_solver(*args, **kwargs)
            return "interrupted", -9

        with (
            patch("xfoil_mac.execution.execute", side_effect=interrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.run_batch()
        saved = json.loads((self.root / "output/a/run.json").read_text())
        self.assertEqual(saved["stage"], "interrupted")
        self.assertTrue(
            (self.root / "output/a/attempts/0001/polar.txt").exists()
        )
        with patch(
            "xfoil_mac.execution.execute", side_effect=fake_solver
        ) as runner:
            self.run_batch(resume=True)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(runner.call_args.args[2].parent.parent.name, "b")
        recovered = json.loads((self.root / "output/a/run.json").read_text())
        self.assertEqual(recovered["status"], "ok")

    def test_resume_summary_matches_current_input_inventory(self):
        (self.database / "a.dat").write_text("broken\n")
        with patch("xfoil_mac.execution.execute", side_effect=fake_solver):
            self.run_batch()
        (self.database / "a.dat").unlink()
        with patch("xfoil_mac.execution.execute") as runner:
            summary = self.run_batch(resume=True)
        runner.assert_not_called()
        with summary.open() as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual([record["airfoil"] for record in records], ["b"])


if __name__ == "__main__":
    unittest.main()
