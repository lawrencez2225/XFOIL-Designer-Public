"""Optional human-paced XQuartz replay, separate from durable calculations."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Sequence

from .runtime import AppPaths, die, live_cpx_pause_seconds, xfoil_env


def run_xfoil_interactive_result(
    app: AppPaths,
    work_dir: Path,
    commands_before_result: Sequence[str],
    alphas: Sequence[float],
) -> int:
    work_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [str(app.xfoil_bin)],
        cwd=str(work_dir),
        env=xfoil_env(app),
        stdin=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdin is not None
        for command in commands_before_result:
            process.stdin.write(command + "\n")
            process.stdin.flush()
            if command.startswith("ASEQ "):
                time.sleep(live_cpx_pause_seconds())

        print("")
        print(
            "Multi-angle Cp sequence plot is displayed in one Xplot11 window."
        )
        print(
            "The curves, coefficient table, and airfoil geometry remain "
            "visible."
        )
        input(
            "Press Return to replay blue/red upper/lower Cp curves for "
            "every alpha..."
        )

        for alpha in alphas:
            process.stdin.write(f"ALFA {alpha:g}\n")
            process.stdin.flush()
            time.sleep(live_cpx_pause_seconds())
        input(
            "Blue/red Cp replay is complete. Press Return to replay "
            "pressure vectors for every alpha..."
        )

        for alpha in alphas:
            process.stdin.write(f"ALFA {alpha:g}\n")
            process.stdin.write("CPV\n")
            process.stdin.flush()
            time.sleep(live_cpx_pause_seconds())
        input(
            "Pressure-vector replay is complete. Press Return to return "
            "to the menu..."
        )

        for command in ["", "QUIT"]:
            process.stdin.write(command + "\n")
            process.stdin.flush()
        process.stdin.close()
        return process.wait(timeout=120)
    finally:
        if process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def build_operating_state_commands(
    re_value: float,
    mach: float,
    iterations: int,
    reynolds_command: str = "VISC",
) -> list[str]:
    reynolds_command = reynolds_command.upper()
    if reynolds_command not in {"VISC", "RE"}:
        die(f"internal error: unsupported Reynolds command {reynolds_command}")
    return [
        "OPER",
        reynolds_command,
        f"{re_value:g}",
        "MACH",
        f"{mach:g}",
        "TYPE",
        "1",
        "ITER",
        f"{iterations}",
    ]


def build_geometry_commands(
    load_command: str, show_xfoil_geometry: bool = True
) -> list[str]:
    commands = [
        load_command,
        "PANE",
        "PSAV geometry.dat",
    ]
    if show_xfoil_geometry:
        commands.extend(["GDES", ""])
    return commands
