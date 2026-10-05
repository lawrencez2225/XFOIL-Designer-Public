"""Recover stalled BL iterations without accepting incomplete solver output."""

import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from tests.helpers import fake_solver
from tests.test_boundary_integrity import VALID_DUMP
from xfoil_mac.analysis.boundary_layer import boundary_layer_error
from xfoil_mac.analysis.diagnostics import diagnose, log_evidence
from xfoil_mac.data import POLAR_FIELDS, cp_name, parse_polar_file, read_pairs
from xfoil_mac.execution import execute, solve, solver_commands
from xfoil_mac.flow import actual_conditions
from xfoil_mac.runtime import discover_app, xfoil_env


def recovery_config():
    return {
        "input": {"kind": "naca", "naca": "0012"},
        "re": 1e6,
        "mach": 0.1,
        "iterations": 100,
        "timeout": 10,
        "retries": 2,
        "retry_step": 0.5,
        "alphas": [0.0, 1.0, 2.0],
    }


class SolverRecoveryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = recovery_config()

    def solve(self):
        return solve(
            Path("fake"),
            {},
            self.root,
            self.root / "polar.txt",
            self.config,
            [],
            [],
            lambda *args: None,
        )

    def diagnose(self, attempts):
        config = dict(self.config, polar_filename="polar.txt")
        (self.root / "run.json").write_text(
            json.dumps({"config": config, "attempts": attempts})
        )
        with patch("xfoil_mac.analysis.diagnostics.report_page"):
            return diagnose(self.root)["points"]

    def stalled(self, binary, env, folder, commands, timeout):
        result = fake_solver(binary, env, folder, commands, timeout)
        if "INIT" not in commands:
            path = folder / "polar.txt"
            path.write_text(
                "\n".join(
                    line
                    for line in path.read_text().splitlines()
                    if not line.startswith("1.0 ")
                )
            )
        return result

    def test_reset_recovers_only_the_missing_point_after_continuation(self):
        original = deepcopy(self.config)
        with patch("xfoil_mac.execution.execute", side_effect=self.stalled):
            rows, attempts, status = self.solve()
        self.assertEqual(status, "finished")
        self.assertEqual([r["alpha"] for r in rows], [0, 1, 2])
        self.assertEqual(self.config, original)
        self.assertEqual(
            [a["strategy"] for a in attempts],
            [
                "initial_sweep",
                "continuation",
                "continuation",
                "boundary_layer_reset",
            ],
        )
        self.assertEqual(attempts[-1]["initialization"], "INIT")
        self.assertEqual(attempts[-1]["valid_output_alphas"], [1])
        self.assertEqual([a["alphas"] for a in attempts[1:]], [[1]] * 3)
        initial = parse_polar_file(self.root / "attempts/0001/polar.txt")
        self.assertEqual([r for r in rows if r["alpha"] != 1], initial)
        self.assertEqual(
            parse_polar_file(self.root / "attempts/0003/polar.txt"), []
        )

    def test_zero_retries_does_not_reset(self):
        self.config["retries"] = 0
        with patch("xfoil_mac.execution.execute", side_effect=self.stalled):
            rows, attempts, _ = self.solve()
        self.assertEqual(len(attempts), 1)
        self.assertEqual([r["alpha"] for r in rows], [0, 2])

    def test_permanently_failed_point_is_not_filled_or_retried_forever(self):
        def failed(*args):
            result = self.stalled(*args)
            (args[2] / "polar.txt").write_text(" ".join(POLAR_FIELDS))
            return result

        self.config["alphas"] = [1.0]
        with patch("xfoil_mac.execution.execute", side_effect=failed):
            rows, attempts, _ = self.solve()
        self.assertEqual(rows, [])
        self.assertEqual(len(attempts), 4)
        self.assertEqual(attempts[-1]["valid_output_alphas"], [])
        self.assertFalse((self.root / cp_name(1)).exists())

    def test_reset_without_cp_is_not_complete(self):
        def missing_cp(*args):
            result = self.stalled(*args)
            if "INIT" in args[3]:
                (args[2] / "cp_sequence_0001.txt").unlink()
            return result

        with patch("xfoil_mac.execution.execute", side_effect=missing_cp):
            _, attempts, _ = self.solve()
        self.assertEqual(attempts[-1]["converged_alphas"], [1])
        self.assertEqual(attempts[-1]["valid_output_alphas"], [])
        self.assertFalse((self.root / cp_name(1)).exists())

    def test_reset_with_corrupt_boundary_layer_is_not_complete(self):
        self.config["alphas"] = [1.0]
        self.config["boundary_layer"] = True

        def corrupt_bl(*args):
            result = self.stalled(*args)
            (args[2] / "bl_sequence_0001.txt").write_text("invalid dump")
            return result

        with patch("xfoil_mac.execution.execute", side_effect=corrupt_bl):
            _, attempts, _ = self.solve()
        self.assertEqual(attempts[-1]["valid_output_alphas"], [])
        self.assertTrue(attempts[-1]["boundary_layer_errors"])
        self.assertFalse(
            (self.root / "boundary_layer/bl_alpha_1.txt").exists()
        )

    def test_valid_boundary_layer_is_required_and_promoted(self):
        self.config["alphas"] = [1.0]
        self.config["boundary_layer"] = True

        def valid_bl(*args):
            result = self.stalled(*args)
            (args[2] / "bl_sequence_0001.txt").write_text(VALID_DUMP)
            return result

        with patch("xfoil_mac.execution.execute", side_effect=valid_bl):
            _, attempts, _ = self.solve()
        self.assertEqual(attempts[-1]["valid_output_alphas"], [1])
        self.assertTrue((self.root / "boundary_layer/bl_alpha_1.txt").exists())

    def test_failed_reset_cannot_claim_an_older_polar_row_as_recovery(self):
        self.config["alphas"] = [1.0]
        self.config["boundary_layer"] = True

        def invalid_initial_bl_then_failed_retries(*args):
            result = fake_solver(*args)
            (args[2] / "bl_sequence_0001.txt").write_text("invalid dump")
            if args[2].name != "0001":
                (args[2] / "polar.txt").write_text(" ".join(POLAR_FIELDS))
            return result

        with patch(
            "xfoil_mac.execution.execute",
            side_effect=invalid_initial_bl_then_failed_retries,
        ):
            rows, attempts, _ = self.solve()
        self.assertEqual(len(rows), 1)
        self.assertEqual(attempts[-1]["initialization"], "INIT")
        self.assertEqual(attempts[-1]["converged_alphas"], [])
        self.assertEqual(attempts[-1]["valid_output_alphas"], [])
        point = self.diagnose(attempts)[0]
        self.assertEqual(point["status"], "invalid_boundary_layer")
        self.assertIsNotNone(point["boundary_layer_error"])
        self.assertFalse(point["recovered_by_init"])

    def test_recovery_attribution_requires_this_attempts_output_metadata(self):
        with patch("xfoil_mac.execution.execute", side_effect=self.stalled):
            _, attempts, _ = self.solve()
        point = self.diagnose(attempts)[1]
        self.assertEqual(point["status"], "converged")
        self.assertTrue(point["recovered_by_init"])
        for field in ("converged_alphas", "valid_output_alphas"):
            with self.subTest(missing_field=field):
                legacy = deepcopy(attempts)
                del legacy[-1][field]
                point = self.diagnose(legacy)[1]
                self.assertEqual(point["status"], "converged")
                self.assertFalse(point["recovered_by_init"])

    def test_reset_with_new_polar_cannot_reuse_an_older_pressure_distribution(
        self,
    ):
        self.config["alphas"] = [1.0]
        self.config["boundary_layer"] = True

        def reset_without_new_cp(*args):
            result = fake_solver(*args)
            folder, commands = args[2:4]
            polar = folder / "polar.txt"
            boundary = folder / "bl_sequence_0001.txt"
            if "INIT" in commands:
                polar.write_text(
                    " ".join(POLAR_FIELDS)
                    + "\n1 .9 .01 .003 -.02 .5 .6 50 60\n"
                )
                boundary.write_text(VALID_DUMP)
                (folder / "cp_sequence_0001.txt").unlink()
            else:
                boundary.write_text("invalid dump")
                if folder.name != "0001":
                    polar.write_text(" ".join(POLAR_FIELDS))
            return result

        with patch(
            "xfoil_mac.execution.execute", side_effect=reset_without_new_cp
        ):
            rows, attempts, _ = self.solve()
        self.assertEqual(rows[0]["CL"], 0.9)
        self.assertEqual(attempts[-1]["converged_alphas"], [1])
        self.assertEqual(attempts[-1]["valid_output_alphas"], [])
        self.assertFalse((self.root / cp_name(1)).exists())
        self.assertTrue(
            (self.root / "attempts/0001/cp_sequence_0001.txt").exists()
        )
        point = self.diagnose(attempts)[0]
        self.assertEqual(point["status"], "missing_cp")
        self.assertFalse(point["recovered_by_init"])

    def test_reset_obeys_remaining_timeout(self):
        self.config["timeout"] = 3.5
        clock = [0.0]

        def slow(*args):
            if "INIT" in args[3]:
                self.assertEqual(args[-1], 0.5)
                clock[0] += 0.5
                return "timeout", -9
            clock[0] += 1
            return self.stalled(*args)

        with (
            patch("xfoil_mac.execution.execute", side_effect=slow),
            patch(
                "xfoil_mac.execution.time.monotonic",
                side_effect=lambda: clock[0],
            ),
        ):
            rows, attempts, status = self.solve()
        self.assertEqual(status, "timeout")
        self.assertEqual(len(attempts), 4)
        self.assertEqual([r["alpha"] for r in rows], [0, 2])

    def test_new_neighbor_can_recover_old_gap_after_reset_failed(self):
        self.config["adaptive"] = {
            "rounds": 1,
            "max_points": 10,
            "min_step": 0.125,
        }

        def closer_seed(*args):
            result = fake_solver(*args)
            commands = args[3]
            if args[2].name == "0006":
                self.assertIn("ALFA 0.5", commands)
            else:
                path = args[2] / "polar.txt"
                path.write_text(
                    "\n".join(
                        line
                        for line in path.read_text().splitlines()
                        if not line.startswith("1.0 ")
                    )
                )
            return result

        with (
            patch("xfoil_mac.execution.execute", side_effect=closer_seed),
            patch(
                "xfoil_mac.execution.refinement_candidates", return_value=[0.5]
            ),
        ):
            rows, attempts, _ = self.solve()
        self.assertEqual([r["alpha"] for r in rows], [0, 0.5, 1, 2])
        self.assertEqual(attempts[-1]["seed_alpha"], 0.5)
        self.assertEqual(attempts[-1]["adaptive_round"], 1)
        self.assertEqual(
            sum(a["initialization"] == "INIT" for a in attempts), 1
        )

    def test_reset_preserves_type_two_and_three_initialization(self):
        for flow_type in (1, 2, 3):
            with self.subTest(flow_type=flow_type):
                self.config.update(flow_type=flow_type, reference_cl=0.5)
                commands = solver_commands(
                    self.config,
                    [2],
                    seed=1,
                    retry_round=2,
                    reset_boundary_layer=True,
                )
                index = commands.index("INIT")
                self.assertTrue(
                    any(c.startswith("ALFA ") for c in commands[:index])
                )
                self.assertEqual(commands[index + 1], "PACC")
                self.assertEqual(commands.count("INIT"), 1)
                if flow_type != 1:
                    self.assertLess(commands.index("ALFA 2"), index)
                    self.assertIn(
                        ["TYPE", str(flow_type)],
                        [commands[i : i + 2] for i in range(index)],
                    )


class TargetLogEvidenceTest(unittest.TestCase):
    def test_residuals_do_not_leak_from_warmup_or_other_sweep_points(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "xfoil.log"
            path.write_text(
                "Solving BL system ...\n"
                "1 rms: 0.1E-02\n a = -3.000 CL = -.1\n"
                "Convergence failed\n"
                "Solving BL system ...\n"
                "1 rms: 0.2E-05\n a = -2.000 CL = .1\n"
            )
            failed = log_evidence(path, "finished", -3)
            following = log_evidence(path, "finished", -2)
            self.assertEqual(failed["last_residual"], 0.001)
            self.assertTrue(failed["explicit_failure_in_log"])
            self.assertEqual(following["last_residual"], 0.000002)
            self.assertFalse(following["explicit_failure_in_log"])
            self.assertIsNone(
                log_evidence(path, "finished", 9)["last_residual"]
            )


@unittest.skipUnless(
    os.environ.get("XFOIL_NATIVE_TESTS") == "1",
    "Requires the local XFOIL executable",
)
class NativeSolverRecoveryTest(unittest.TestCase):
    def test_naca2412_minus_three_degrees_recovers_with_same_physics(self):
        config = recovery_config()
        config.update(
            input={"kind": "naca", "naca": "2412"},
            re=367500.0,
            mach=20 / 340.3,
            iterations=200,
            timeout=30,
            alphas=[-4.0, -3.0, -2.0],
            boundary_layer=True,
            solver_settings={
                "ncrit": 9,
                "xtr_top": 1,
                "xtr_bottom": 1,
                "panels": 240,
                "panel_bunching": 1,
                "te_le_ratio": 0.15,
            },
        )
        app = discover_app()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows, attempts, status = solve(
                app.xfoil_bin,
                xfoil_env(app),
                root,
                root / "polar.txt",
                config,
                [],
                [],
                lambda *args: None,
            )
            self.assertEqual(status, "finished")
            self.assertEqual([r["alpha"] for r in rows], [-4, -3, -2])
            reset = attempts[-1]
            self.assertEqual(reset["strategy"], "boundary_layer_reset")
            self.assertEqual(reset["valid_output_alphas"], [-3])
            self.assertEqual(rows[1]["CL"], -0.0788)
            self.assertEqual(rows[1]["CD"], 0.00951)
            initial = parse_polar_file(root / "attempts/0001/polar.txt")
            self.assertEqual([r for r in rows if r["alpha"] != -3], initial)
            log = root / reset["directory"] / "xfoil.log"
            self.assertIn(
                "BLs will be initialized on next point", log.read_text()
            )
            self.assertLess(
                log_evidence(log, "finished", -3)["last_residual"], 1e-4
            )

    def test_native_reset_keeps_cl_dependent_conditions_valid(self):
        app = discover_app()
        for flow_type in (2, 3):
            with (
                self.subTest(flow_type=flow_type),
                tempfile.TemporaryDirectory() as folder,
            ):
                root = Path(folder)
                config = recovery_config()
                config.update(
                    input={"kind": "naca", "naca": "2412"},
                    flow_type=flow_type,
                    reference_cl=0.5,
                    boundary_layer=True,
                )
                commands = solver_commands(
                    config,
                    [2],
                    seed=1,
                    retry_round=2,
                    reset_boundary_layer=True,
                )
                status, code = execute(
                    app.xfoil_bin, xfoil_env(app), root, commands, 15
                )
                self.assertEqual((status, code), ("finished", 0))
                rows = parse_polar_file(root / "polar.txt")
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["alpha"], 2)
                self.assertIn(
                    (
                        "2 2 Reynolds number ~ 1/sqrt(CL)"
                        if flow_type == 2
                        else "3 1 Reynolds number ~ 1/CL"
                    ),
                    (root / "polar.txt").read_text(),
                )
                self.assertIsNotNone(actual_conditions(config, rows[0]["CL"]))
                self.assertGreaterEqual(
                    len(read_pairs(root / "cp_sequence_0001.txt")), 3
                )
                self.assertIsNone(
                    boundary_layer_error(
                        root / "bl_sequence_0001.txt",
                        read_pairs(root / "geometry.dat"),
                    )
                )
                self.assertIn(
                    "BLs will be initialized on next point",
                    (root / "xfoil.log").read_text(),
                )


if __name__ == "__main__":
    unittest.main()
