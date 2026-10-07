"""Opt-in checks against the actual local XFOIL and AVL executables."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xfoil_mac.avl import find_avl, run_wing
from xfoil_mac.data import sha256_file
from xfoil_mac.runtime import discover_app
from xfoil_mac.workflows import run_polar
from xfoil_mac.storage.dataset import collect_rows
from xfoil_mac.flow import actual_conditions

ROOT = Path(__file__).resolve().parents[1]


def avl_installed() -> bool:
    """Whether an AVL build is available.

    AVL is downloaded rather than committed, so a fresh clone has none
    until `--install-avl` is run. Two tests below need it; without this
    they error out on an otherwise healthy checkout.
    """
    try:
        find_avl(ROOT)
    except ValueError:
        return False
    return True


AVL_MISSING = "AVL is not installed; run python -m xfoil_mac --install-avl"


@unittest.skipUnless(
    os.environ.get("XFOIL_NATIVE_TESTS") == "1",
    "Set XFOIL_NATIVE_TESTS=1 for local solver tests",
)
class NativeIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for target, value in [
            ("write_geometry_plot", None),
            (
                "write_cp_polar_outputs",
                {"result_plots": [], "pressure_vectors": []},
            ),
        ]:
            p = patch("xfoil_mac.workflows." + target, return_value=value)
            p.start()
            self.addCleanup(p.stop)

    def test_real_type_two_adaptive_and_complete_resume(self):
        options = dict(
            app=discover_app(),
            naca="2412",
            reynolds=1e6,
            mach=0.1,
            flow_type=2,
            reference_cl=0.5,
            alpha_start=0,
            alpha_end=8,
            alpha_step=2,
            out_file=self.root / "polar.txt",
            adaptive_rounds=2,
            adaptive_max_points=15,
            show_xfoil_geometry=False,
            timeout=40,
            quiet=True,
        )
        run_polar(**options)
        saved = json.loads((self.root / "run.json").read_text())
        self.assertEqual(saved["status"], "ok")
        self.assertGreater(saved["summary"]["points"], 5)
        self.assertEqual(saved["config"]["alphas"], [0, 2, 4, 6, 8])
        self.assertIn(
            "2 2 Reynolds number ~ 1/sqrt(CL)",
            (self.root / "attempts/0001/polar.txt").read_text(),
        )
        dataset, _ = collect_rows(self.root)
        for row in dataset:
            expected = actual_conditions(saved["config"], row["CL"])
            self.assertAlmostEqual(row["re"], expected[0])
            self.assertAlmostEqual(row["mach"], expected[1])
        before = {
            str(p): sha256_file(p)
            for p in self.root.glob("attempts/*/*")
            if p.is_file()
        }
        with patch(
            "xfoil_mac.execution.execute",
            side_effect=AssertionError("Resume invoked XFOIL"),
        ):
            run_polar(**options, resume=True)
        self.assertEqual(before, {p: sha256_file(Path(p)) for p in before})

    def test_real_type_three_exports_actual_conditions(self):
        run_polar(
            discover_app(),
            naca="2412",
            reynolds=1e6,
            mach=0.1,
            flow_type=3,
            reference_cl=0.5,
            alpha_start=0,
            alpha_end=4,
            alpha_step=2,
            out_file=self.root / "polar.txt",
            show_xfoil_geometry=False,
            timeout=40,
            quiet=True,
        )
        self.assertIn(
            "3 1 Reynolds number ~ 1/CL",
            (self.root / "attempts/0001/polar.txt").read_text(),
        )
        with (self.root / "operating_points.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertAlmostEqual(float(row["Re"]) * float(row["CL"]), 500000)
            self.assertEqual(float(row["Mach"]), 0.1)
        dataset, _ = collect_rows(self.root)
        for row in dataset:
            self.assertAlmostEqual(row["re"] * row["CL"], 500000)
            self.assertEqual(row["mach"], 0.1)

    @unittest.skipUnless(avl_installed(), AVL_MISSING)
    def test_real_avl_target_lift_mach_derivatives_and_resume(self):
        binary = find_avl(ROOT)
        source = self.root / "wing.json"
        config = json.loads((ROOT / "examples/wing.json").read_text())
        config["cases"] = [{"cl": 0.5}]
        source.write_text(json.dumps(config))
        with patch("xfoil_mac.avl._plot_wing"):
            result = run_wing(source, self.root / "wing", binary, timeout=30)
            report = json.loads(result.read_text())
            case = report["cases"][0]
            self.assertEqual(case["status"], "ok")
            self.assertAlmostEqual(case["Mach"], 0.1, places=3)
            self.assertAlmostEqual(case["CLtot"], 0.5, places=4)
            self.assertGreater(case["CDind"], 0)
            self.assertTrue(3 < case["CLa"] < 6.5)
            self.assertLess(abs(case["Cnb"]), 0.02)
            with patch(
                "xfoil_mac.execution.execute",
                side_effect=AssertionError("Resume invoked AVL"),
            ):
                run_wing(
                    source, self.root / "wing", binary, timeout=30, resume=True
                )

    @unittest.skipUnless(avl_installed(), AVL_MISSING)
    def test_real_avl_higher_aspect_ratio_reduces_induced_drag_at_equal_lift(
        self,
    ):
        binary = find_avl(ROOT)
        drag = []
        with patch("xfoil_mac.avl._plot_wing"):
            for span in (8, 12):
                config = json.loads((ROOT / "examples/wing.json").read_text())
                config["cases"] = [{"cl": 0.5}]
                config["reference"].update(area=span, span=span)
                config["surfaces"][0]["sections"][-1].update(
                    y=span / 2, twist=0
                )
                source = self.root / f"wing{span}.json"
                source.write_text(json.dumps(config))
                report = json.loads(
                    run_wing(
                        source, self.root / f"run{span}", binary, 30
                    ).read_text()
                )
                self.assertEqual(report["status"], "ok")
                drag.append(report["cases"][0]["CDind"])
        self.assertLess(drag[1], drag[0] * 0.8)


if __name__ == "__main__":
    unittest.main()
