"""Reject incomplete XFOIL DUMP outputs without shifting surface identities."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from xfoil_mac.analysis.boundary_layer import (
    read_boundary_layer,
    write_boundary_reports,
)
from xfoil_mac.analysis.lift import run_lift
from xfoil_mac.data import POLAR_FIELDS, sha256_file
from xfoil_mac.runtime import discover_app
from xfoil_mac.workflows import run_polar

GEOMETRY = [(1.0, 0.0), (0.5, 0.08), (0.0, 0.0), (0.5, -0.05), (1.0, 0.0)]
DUMP_LINES = [
    "# s x y Ue/Vinf Dstar Theta Cf H",
    *(
        f"{i} {x} {y} 1 .01 .005 .001 2"
        for i, (x, y) in enumerate(GEOMETRY + [(1.1, 0.0)])
    ),
]
VALID_DUMP = "\n".join(DUMP_LINES) + "\n"


class BoundaryIntegrityTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.app = replace(discover_app(), run_root=self.root / "runs")
        self.dump = VALID_DUMP

    def fake_execute(self, binary, env, folder, commands, timeout):
        """Write a converged point; vary only DUMP integrity across tests."""
        (folder / "geometry.dat").write_text(
            "\n".join(f"{x} {y}" for x, y in GEOMETRY)
        )
        target = next(
            (
                float(c.split()[1])
                for c in reversed(commands)
                if c.startswith("CL ")
            ),
            0.2,
        )
        (folder / "polar.txt").write_text(
            " ".join(POLAR_FIELDS)
            + f"\n0 {target} .01 .003 -.02 .5 .6 50 60\n"
        )
        (folder / "xfoil.log").write_text("Converged fixture\n")
        for command in commands:
            if command.startswith("CPWR "):
                (folder / command.split()[1]).write_text("1 0\n.5 -1\n0 0\n")
            elif command.startswith("DUMP "):
                (folder / command.split()[1]).write_text(self.dump)
        return "finished", 0

    def polar_options(self):
        return dict(
            app=self.app,
            reynolds=1e6,
            mach=0.05,
            alpha_start=0,
            alpha_end=0,
            retries=0,
            boundary_layer=True,
            out_file=self.root / "polar" / "polar.txt",
            show_xfoil_geometry=False,
            save_pressure_vectors=False,
            quiet=True,
        )

    def lift_options(self):
        return dict(
            app=self.app,
            targets=[0.2],
            destination=self.root / "lift",
            reynolds=1e6,
            mach=0.05,
            retries=0,
            boundary_layer=True,
        )

    def test_valid_dump_preserves_upper_lower_and_wake(self):
        path = self.root / "dump.txt"
        path.write_text(VALID_DUMP)
        records = read_boundary_layer(path, GEOMETRY)
        self.assertEqual(
            [row["surface"] for row in records],
            ["upper", "upper", "upper", "lower", "lower", "wake"],
        )
        self.assertEqual(records[3]["y"], -0.05)
        self.assertEqual(records[-1]["x"], 1.1)

    def test_malformed_rows_are_errors_not_omitted_records(self):
        path = self.root / "dump.txt"
        for malformed in (
            "bad data",
            "1 .5 .08 1 .01",
            "1 .5 .08 nan .01 .005 .001 2",
        ):
            with self.subTest(malformed=malformed):
                lines = DUMP_LINES[:]
                lines[2] = malformed
                path.write_text("\n".join(lines))
                with self.assertRaisesRegex(ValueError, r"dump.txt:3"):
                    read_boundary_layer(path, GEOMETRY)

    def test_removed_surface_row_cannot_turn_wake_into_lower_surface(self):
        path = self.root / "dump.txt"
        path.write_text("\n".join(DUMP_LINES[:4] + DUMP_LINES[5:]))
        with self.assertRaisesRegex(ValueError, "does not match geometry"):
            read_boundary_layer(path, GEOMETRY)

    def test_empty_truncated_and_wrong_geometry_are_rejected(self):
        path = self.root / "dump.txt"
        for contents in ("", DUMP_LINES[0], "\n".join(DUMP_LINES[:4])):
            with self.subTest(contents=contents):
                path.write_text(contents)
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    read_boundary_layer(path, GEOMETRY)
        path.write_text(VALID_DUMP)
        changed = GEOMETRY[:]
        changed[1] = (0.5, 0.09)
        with self.assertRaisesRegex(ValueError, "does not match geometry"):
            read_boundary_layer(path, changed)

    def test_native_rounding_and_fortran_exponents_are_accepted(self):
        path = self.root / "dump.txt"
        path.write_text(VALID_DUMP.replace(".01", "1.0D-02"))
        geometry = [(x + 1e-7, y - 1e-7) for x, y in GEOMETRY]
        self.assertEqual(len(read_boundary_layer(path, geometry)), 6)

    def test_invalid_report_records_error_and_removes_stale_plot(self):
        folder = self.root / "boundary_layer"
        folder.mkdir()
        (self.root / "geometry.dat").write_text(
            "\n".join(f"{x} {y}" for x, y in GEOMETRY)
        )
        (folder / "bl_alpha_0.txt").write_text("")
        (folder / "bl_alpha_0.png").write_text("stale")
        (folder / "bl_alpha_0.csv").write_text("stale")
        self.assertEqual(write_boundary_reports(self.root, [{"alpha": 0}]), [])
        report = json.loads((folder / "index.json").read_text())
        self.assertEqual(report["errors"][0]["status"], "invalid_output")
        self.assertFalse((folder / "bl_alpha_0.png").exists())
        self.assertFalse((folder / "bl_alpha_0.csv").exists())

    def test_polar_empty_dump_is_incomplete_and_resume_retries(self):
        self.dump = ""
        with patch(
            "xfoil_mac.execution.execute", side_effect=self.fake_execute
        ):
            run_polar(**self.polar_options())
        manifest = self.root / "polar" / "run.json"
        saved = json.loads(manifest.read_text())
        self.assertEqual(saved["status"], "incomplete_outputs")
        self.assertEqual(saved["missing_boundary_layer_alphas"], [0])
        self.assertIn(
            "Incomplete",
            saved["attempts"][0]["boundary_layer_errors"][0]["error"],
        )
        self.dump = VALID_DUMP
        with patch(
            "xfoil_mac.execution.execute", side_effect=self.fake_execute
        ) as runner:
            run_polar(**self.polar_options(), resume=True)
        self.assertEqual(runner.call_count, 1)
        saved = json.loads(manifest.read_text())
        self.assertEqual(saved["status"], "ok")
        self.assertEqual(saved["boundary_layer_errors"], [])
        self.assertEqual(len(saved["attempts"]), 2)

    def test_complete_polar_resume_checks_dump_content(self):
        with patch(
            "xfoil_mac.execution.execute", side_effect=self.fake_execute
        ):
            run_polar(**self.polar_options())
        root = self.root / "polar"
        for path in (
            root / "boundary_layer" / "bl_alpha_0.txt",
            root / "attempts" / "0001" / "bl_sequence_0001.txt",
        ):
            path.write_text("")
        with patch(
            "xfoil_mac.execution.execute", side_effect=self.fake_execute
        ) as runner:
            run_polar(**self.polar_options(), resume=True)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(
            json.loads((root / "run.json").read_text())["status"], "ok"
        )

    def test_lift_empty_dump_is_not_marked_successful(self):
        self.dump = ""
        with patch(
            "xfoil_mac.analysis.lift.execute", side_effect=self.fake_execute
        ):
            manifest = run_lift(**self.lift_options())
        saved = json.loads(manifest.read_text())
        self.assertEqual(saved["status"], "needs_attention")
        point = saved["points"][0]
        self.assertEqual(point["status"], "incomplete_outputs")
        self.assertNotIn("result", point)
        self.assertIn("Incomplete", point["attempts"][0]["output_errors"][0])

    def test_lift_resume_rejects_invalid_dump_even_with_matching_hash(self):
        with patch(
            "xfoil_mac.analysis.lift.execute", side_effect=self.fake_execute
        ):
            manifest = run_lift(**self.lift_options())
        saved = json.loads(manifest.read_text())
        point = saved["points"][0]
        raw_path = point["attempts"][0]["directory"] + "/boundary_layer.txt"
        dump = manifest.parent / raw_path
        dump.write_text("")
        point["raw_hashes"][raw_path] = sha256_file(dump)
        manifest.write_text(json.dumps(saved))
        with patch(
            "xfoil_mac.analysis.lift.execute", side_effect=self.fake_execute
        ) as runner:
            run_lift(**self.lift_options(), resume=True)
        self.assertEqual(runner.call_count, 1)
        saved = json.loads(manifest.read_text())
        self.assertEqual(saved["status"], "ok")
        self.assertEqual(len(saved["points"][0]["attempts"]), 2)
        self.assertIn(
            "Incomplete", saved["points"][0]["resume_output_errors"][0]
        )


if __name__ == "__main__":
    unittest.main()
