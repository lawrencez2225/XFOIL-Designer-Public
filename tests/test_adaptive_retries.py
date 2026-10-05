"""Regression checks for missing points introduced after the retry loop."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xfoil_mac.data import POLAR_FIELDS, cp_name, parse_polar_file
from xfoil_mac.execution import solve
from xfoil_mac.runtime import FlowAssumptions, discover_app
from xfoil_mac.workflows import run_polar
from tests.helpers import fake_solver


class AdaptiveRetryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            "input": {"kind": "naca", "naca": "0012"},
            "re": 1e6,
            "mach": 0.1,
            "iterations": 100,
            "timeout": 5,
            "retries": 2,
            "retry_step": 0.5,
            "alphas": [0.0, 1.0, 2.0],
            "adaptive": {"rounds": 1, "max_points": 10, "min_step": 0.125},
        }

    def solve(self):
        return solve(
            Path("fake"),
            {},
            self.root,
            self.root / "polar.txt",
            self.config,
            [],
            [],
            lambda *a: None,
        )

    def test_failed_refinement_retries_from_other_side_and_keeps_raw_failure(
        self,
    ):
        def partial(*args, **kwargs):
            result = fake_solver(*args, **kwargs)
            if args[2].name == "0002":
                (args[2] / "polar.txt").write_text(" ".join(POLAR_FIELDS))
            return result

        with (
            patch("xfoil_mac.execution.execute", side_effect=partial),
            patch(
                "xfoil_mac.execution.refinement_candidates", return_value=[0.5]
            ),
        ):
            rows, attempts, status = self.solve()
        self.assertEqual(status, "finished")
        self.assertEqual([r["alpha"] for r in rows], [0, 0.5, 1, 2])
        self.assertEqual(len(attempts), 3)
        self.assertEqual([r["seed_alpha"] for r in attempts[1:]], [0, 1])
        self.assertEqual([r["retry_round"] for r in attempts[1:]], [1, 2])
        self.assertEqual([r["adaptive_round"] for r in attempts[1:]], [1, 1])
        self.assertEqual(
            parse_polar_file(self.root / "attempts/0002/polar.txt"), []
        )
        self.assertTrue(
            (self.root / "attempts/0002/cp_sequence_0001.txt").exists()
        )
        self.assertTrue((self.root / cp_name(0.5)).exists())

    def test_refinement_retry_limit_keeps_unconverged_point_missing(self):
        for retries in (0, 2):
            with self.subTest(retries=retries):
                self.root = Path(self.temp.name) / str(retries)
                self.root.mkdir()
                self.config["retries"] = retries
                self.config["alphas"] = [0.0, 1.0, 2.0]

                def failed(*args, **kwargs):
                    result = fake_solver(*args, **kwargs)
                    if args[2].name != "0001":
                        (args[2] / "polar.txt").write_text(
                            " ".join(POLAR_FIELDS)
                        )
                    return result

                with (
                    patch("xfoil_mac.execution.execute", side_effect=failed),
                    patch(
                        "xfoil_mac.execution.refinement_candidates",
                        return_value=[0.5],
                    ),
                ):
                    rows, attempts, _ = self.solve()
                self.assertEqual(
                    len(attempts), 2 + retries + (1 if retries else 0)
                )
                self.assertEqual([r["alpha"] for r in rows], [0, 1, 2])
                self.assertIn(0.5, self.config["alphas"])
                self.assertFalse((self.root / cp_name(0.5)).exists())

    def test_refinement_retry_stays_within_remaining_budget(
        self,
    ):
        clock = [0.0]
        self.config["timeout"] = 0.5

        def slow_failure(*args, **kwargs):
            result = fake_solver(*args, **kwargs)
            if args[2].name == "0001":
                clock[0] += 0.1
            else:
                clock[0] += 0.5
                (args[2] / "polar.txt").write_text(" ".join(POLAR_FIELDS))
            return result

        with (
            patch(
                "xfoil_mac.execution.time.monotonic",
                side_effect=lambda: clock[0],
            ),
            patch(
                "xfoil_mac.execution.execute", side_effect=slow_failure
            ) as runner,
            patch(
                "xfoil_mac.execution.refinement_candidates", return_value=[0.5]
            ),
        ):
            _, attempts, status = self.solve()
        self.assertEqual(status, "timeout")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(runner.call_count, 2)
        self.assertAlmostEqual(runner.call_args_list[1].args[-1], 0.4)


@unittest.skipUnless(
    os.environ.get("XFOIL_NATIVE_TESTS") == "1",
    "Requires the local XFOIL executable",
)
class NativeAdaptiveRetryTest(unittest.TestCase):
    def test_naca2412_sweep_recovers_four_point_two_five_degrees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("xfoil_mac.workflows.write_geometry_plot"),
                patch(
                    "xfoil_mac.workflows.write_cp_polar_outputs",
                    return_value={"result_plots": [], "pressure_vectors": []},
                ),
            ):
                run_polar(
                    discover_app(),
                    naca="2412",
                    reynolds=1747237,
                    mach=0.25,
                    assumptions=FlowAssumptions(chord_m=0.3),
                    alpha_start=-4,
                    alpha_end=12,
                    alpha_step=1,
                    adaptive_rounds=2,
                    out_file=root / "polar.txt",
                    show_xfoil_geometry=False,
                    timeout=30,
                    quiet=True,
                )
            manifest = json.loads((root / "run.json").read_text())
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(manifest["summary"]["failed_alphas"], [])
            point = next(
                r
                for r in parse_polar_file(root / "polar.txt")
                if r["alpha"] == 4.25
            )
            self.assertAlmostEqual(point["CL"], 0.7617, delta=0.0002)
            self.assertTrue((root / cp_name(4.25)).exists())
            attempts = [
                r for r in manifest["attempts"] if r["alphas"] == [4.25]
            ]
            self.assertEqual(attempts[-1]["converged_alphas"], [4.25])
            if len(attempts) > 1:
                self.assertNotEqual(
                    attempts[0]["seed_alpha"], attempts[-1]["seed_alpha"]
                )


if __name__ == "__main__":
    unittest.main()
