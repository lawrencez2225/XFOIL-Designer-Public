import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xfoil_mac import cli as x
from xfoil_mac import workflows
from tests.helpers import fake_solver


class CLITest(unittest.TestCase):
    def test_cartesian_sweep_and_comparison_use_four_distinct_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "xfoil"
            binary.write_text("fake")
            app = x.AppPaths(
                root,
                root,
                binary,
                root,
                root / "runs",
                root / "single",
                root / "batch",
            )
            with (
                patch.object(x, "discover_app", return_value=app),
                patch("xfoil_mac.execution.execute", side_effect=fake_solver),
                patch.object(workflows, "write_geometry_plot"),
                patch.object(
                    workflows,
                    "write_cp_polar_outputs",
                    return_value={"result_plots": [], "pressure_vectors": []},
                ),
                patch.object(x, "write_comparison") as compare,
            ):
                status = x.main(
                    [
                        "--polar",
                        "--nacas",
                        "0012",
                        "2412",
                        "--re-list",
                        "500000",
                        "1000000",
                        "--mach",
                        ".1",
                        "--aseq",
                        "0",
                        "1",
                        "1",
                        "--headless",
                        "--outdir",
                        str(root / "study"),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(len(compare.call_args.args[0]), 4)
            manifests = [
                json.loads(p.read_text())
                for p in (root / "study").rglob("run.json")
            ]
            self.assertEqual(
                {
                    (m["config"]["input"]["naca"], m["config"]["re"])
                    for m in manifests
                },
                {
                    ("0012", 500000),
                    ("0012", 1000000),
                    ("2412", 500000),
                    ("2412", 1000000),
                },
            )

    def test_resume_requires_stable_output_directory(self):
        with (
            patch.object(x, "discover_app", return_value=x.discover_app()),
            self.assertRaises(SystemExit),
        ):
            x.main(["--polar", "--resume", "--headless"])

    def test_conflicting_options_rejected(self):
        for args in (
            ["--force", "--resume"],
            ["--re", "1e6", "--re-list", "1e6"],
            ["--nacas", "0012", "--airfoil", "a.dat"],
        ):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                x.build_parser().parse_args(args)

    def test_physical_options_are_not_silently_ignored_by_other_modes(self):
        for args in (
            ["--wing", "example.json"],
            ["--compare", "results"],
            ["--install-avl"],
        ):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                x.main([*args, "--chord", ".3"])


if __name__ == "__main__":
    unittest.main()
