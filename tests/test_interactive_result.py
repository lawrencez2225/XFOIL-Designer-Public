import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from xfoil_mac import preview


class RecordingStdin(io.StringIO):
    def close(self) -> None:
        pass


class FakeProcess:
    def __init__(self) -> None:
        self.stdin = RecordingStdin()

    def wait(self, timeout=None) -> int:
        return 0

    def poll(self):
        return 0


class InteractiveResultTest(unittest.TestCase):
    def test_replays_colored_cp_and_pressure_vectors_for_every_alpha(
        self,
    ) -> None:
        process = FakeProcess()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app = preview.AppPaths(
                app_root=root,
                xfoil_home=root,
                xfoil_bin=root / "xfoil",
                runtime_lib=root,
                run_root=root,
                single_root=root,
                batch_root=root,
            )
            with (
                patch.object(
                    preview.subprocess, "Popen", return_value=process
                ),
                patch.object(preview, "xfoil_env", return_value={}),
                patch.object(preview.time, "sleep"),
                patch(
                    "builtins.input", side_effect=["", "", ""]
                ) as user_input,
            ):
                result = preview.run_xfoil_interactive_result(
                    app,
                    root,
                    ["OPER", "ASEQ -2 2 2", "PACC"],
                    [-2.0, 0.0, 2.0],
                )

        self.assertEqual(result, 0)
        self.assertEqual(
            process.stdin.getvalue().splitlines(),
            [
                "OPER",
                "ASEQ -2 2 2",
                "PACC",
                "ALFA -2",
                "ALFA 0",
                "ALFA 2",
                "ALFA -2",
                "CPV",
                "ALFA 0",
                "CPV",
                "ALFA 2",
                "CPV",
                "",
                "QUIT",
            ],
        )
        self.assertEqual(user_input.call_count, 3)
        self.assertIn("replay", user_input.call_args_list[0].args[0].lower())
        self.assertIn(
            "pressure vectors", user_input.call_args_list[1].args[0].lower()
        )
        self.assertEqual(
            user_input.call_args_list[2],
            call(
                "Pressure-vector replay is complete. "
                "Press Return to return to the menu..."
            ),
        )

    def test_single_target_warm_start_is_excluded_from_saved_polar(
        self,
    ) -> None:
        from xfoil_mac.execution import solver_commands

        config = {
            "input": {"kind": "naca", "naca": "0012"},
            "re": 1e6,
            "mach": 0.1,
            "iterations": 100,
            "retry_step": 0.5,
        }
        for alpha in (5.0, -3.5):
            with self.subTest(alpha=alpha):
                commands = solver_commands(config, [alpha])
                before_save = commands[: commands.index("PACC")]
                self.assertIn("ALFA 0", before_save)
                self.assertFalse(
                    any(
                        c.startswith("ALFA ")
                        for c in commands[commands.index("PACC") :]
                    )
                )
                self.assertIn("CSAV cp_sequence_", commands)
                self.assertIn(f"ASEQ {alpha:g} {alpha:g} 1", commands)
                self.assertEqual(commands[:3], ["PLOP", "G", ""])


if __name__ == "__main__":
    unittest.main()
