import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xfoil_mac import plotting as x
from xfoil_mac import data, runtime
from xfoil_mac.data import (
    convergence_summary,
    gapped_rows,
    performance_metrics,
    POLAR_FIELDS,
)
from xfoil_mac.execution import execute, solve
from tests.helpers import fake_solver


def row(alpha, cl=None, cd=0.01):
    return dict(
        zip(
            POLAR_FIELDS,
            [
                alpha,
                alpha if cl is None else cl,
                cd,
                0.003,
                -0.02,
                0.5,
                0.6,
                50,
                60,
            ],
        )
    )


class ValidationTest(unittest.TestCase):
    def test_invalid_inputs_fail_before_starting_xfoil(self):
        for value in (-1, 0, math.nan, math.inf):
            with self.subTest(re=value), self.assertRaises(ValueError):
                runtime.resolve_flow_inputs(value, 0.1)
        for args in (
            (math.nan, 1, 1),
            (0, math.inf, 1),
            (0, 1, 0),
            (0, 1, -1),
            (0, 10, 0.00001),
            (0.0005, 0.0015, 0.001),
        ):
            with self.subTest(alpha=args), self.assertRaises(ValueError):
                data.alpha_values(*args)
        self.assertEqual(data.alpha_values(2, -2, -1), [2, 1, 0, -1, -2])

    def test_partial_convergence_and_missing_point_break(self):
        rows = [row(0), row(2)]
        summary = convergence_summary(rows, [0, 1, 2])
        self.assertEqual(summary["status"], "partial_convergence")
        self.assertEqual(summary["failed_alphas"], [1])
        self.assertAlmostEqual(summary["convergence_rate"], 2 / 3)
        self.assertTrue(math.isnan(gapped_rows(rows, [0, 1, 2])[1]["CD"]))
        self.assertEqual(x.polar_row_for_alpha(rows, 1), {"alpha": 1})

    def test_parser_rejects_nonfinite_and_negative_drag(self):
        text = (
            "alpha CL CD CDp CM Top_Xtr Bot_Xtr\n0 nan .1 0 0 .5 .5\n"
            "1 .1 -.01 0 0 .5 .5\n2 .2 .01 0 0 .5 .5\n"
        )
        self.assertEqual(
            [r["alpha"] for r in data.parse_polar_text(text)], [2]
        )

    def test_target_cl_interpolation_does_not_bridge_gaps_or_ambiguity(
        self,
    ):
        result = performance_metrics(
            [row(0, 0, 0.01), row(1, 1, 0.02)], [0, 1], 0.5
        )
        self.assertAlmostEqual(result["cd_at_target_cl"], 0.015)
        self.assertEqual(result["target_cl_status"], "interpolated")
        result = performance_metrics([row(0, 0), row(2, 1)], [0, 1, 2], 0.5)
        self.assertIsNone(result["cd_at_target_cl"])
        result = performance_metrics(
            [row(0, 0), row(1, 1), row(2, 0)], [0, 1, 2], 0.5
        )
        self.assertEqual(
            result["target_cl_status"], "ambiguous_multiple_crossings"
        )

    def test_cp_without_matching_converged_polar_row_is_not_plotted(self):
        with tempfile.TemporaryDirectory() as folder:
            with (
                patch.object(x, "write_cp_distribution_plot") as cp_plot,
                patch.object(x, "write_polar_result_plots", return_value=[]),
                patch.object(x, "write_pressure_vector_plot") as vectors,
            ):
                x.write_cp_polar_outputs(
                    [(0, [(0, 1)]), (1, [(0, 9)])],
                    [],
                    [row(0)],
                    Path(folder),
                    "test",
                )
        self.assertEqual([a for a, _ in cp_plot.call_args.args[0]], [0])
        self.assertEqual(vectors.call_count, 1)


class SolverTest(unittest.TestCase):
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
        }

    def test_retry_recovers_missing_angle_and_keeps_raw_failed_output(self):
        def partial(*args, **kwargs):
            result = fake_solver(*args, **kwargs)
            if args[2].name == "0001":
                path = args[2] / "polar.txt"
                path.write_text(
                    "\n".join(
                        line
                        for line in path.read_text().splitlines()
                        if not line.startswith("1.0 ")
                    )
                )
            return result

        with patch("xfoil_mac.execution.execute", side_effect=partial):
            rows, attempts, status = solve(
                Path("fake"),
                {},
                self.root,
                self.root / "polar.txt",
                self.config,
                [],
                [],
                lambda *a: None,
            )
        self.assertEqual([r["alpha"] for r in rows], [0, 1, 2])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[-1]["alphas"], [1])
        self.assertIn(attempts[-1]["seed_alpha"], (0, 2))
        self.assertTrue(
            (self.root / "attempts/0001/cp_sequence_0002.txt").exists()
        )
        self.assertTrue((self.root / "cp_alpha_1.txt").exists())

    def test_nonconverged_cp_remains_raw_and_is_not_promoted(self):
        self.config["retries"] = 0

        def failed(*args, **kwargs):
            fake_solver(*args, **kwargs)
            (args[2] / "polar.txt").write_text(" ".join(POLAR_FIELDS))
            return "finished", 0

        with patch("xfoil_mac.execution.execute", side_effect=failed):
            rows, _, _ = solve(
                Path("fake"),
                {},
                self.root,
                self.root / "polar.txt",
                self.config,
                [],
                [],
                lambda *a: None,
            )
        self.assertEqual(rows, [])
        self.assertFalse((self.root / "cp_alpha_0.txt").exists())
        self.assertTrue(
            (self.root / "attempts/0001/cp_sequence_0001.txt").exists()
        )

    def test_resume_recovers_uncheckpointed_attempt_without_recalculating(
        self,
    ):
        directory = self.root / "attempts/0001"
        directory.mkdir(parents=True)
        fake_solver(None, None, directory, ["ASEQ 0 2 1"], 1)
        previous = [
            {
                "directory": "attempts/0001",
                "alphas": [0, 1, 2],
                "status": "running",
            }
        ]
        with patch("xfoil_mac.execution.execute") as runner:
            rows, attempts, _ = solve(
                Path("fake"),
                {},
                self.root,
                self.root / "polar.txt",
                self.config,
                [],
                previous,
                lambda *a: None,
            )
        runner.assert_not_called()
        self.assertEqual(len(rows), 3)
        self.assertEqual(attempts[0]["status"], "interrupted")

    def test_timeout_reaps_a_real_process_and_keeps_log_and_commands(self):
        executable = self.root / "sleeper"
        executable.write_text(
            "#!/bin/sh\nprintf 'started\\n'\nexec /bin/sleep 30\n"
        )
        executable.chmod(0o755)
        # The assertion below reads what the child managed to write before
        # the timeout killed it, so the timeout has to be comfortably longer
        # than it takes to schedule a shell on a busy machine. At one second
        # this raced: the kill could land before the shell had printed, and
        # the log came back empty.
        status, code = execute(
            executable, os.environ.copy(), self.root, ["QUIT"], 5.0
        )
        self.assertEqual(status, "timeout")
        self.assertLess(code, 0)
        self.assertTrue((self.root / "commands.txt").is_file())
        self.assertIn("started", (self.root / "xfoil.log").read_text())

    def test_keyboard_interrupt_kills_and_reaps_solver(self):
        with patch("xfoil_mac.execution.subprocess.Popen") as popen:
            process = popen.return_value
            process.communicate.side_effect = [KeyboardInterrupt(), None]
            process.returncode = -9
            status, code = execute(Path("fake"), {}, self.root, ["QUIT"], 1)
        self.assertEqual(status, "interrupted")
        process.kill.assert_called_once()
        self.assertEqual(process.communicate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
