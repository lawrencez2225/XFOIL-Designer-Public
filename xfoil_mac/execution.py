"""Bounded XFOIL runs with retained raw outputs and per-attempt checkpoints."""

from __future__ import annotations

import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from .data import (
    cp_name,
    read_pairs,
    aligned_rows,
    atomic_text,
    parse_polar_text,
    row_for_alpha,
    write_combined_polar,
)
from .flow import actual_conditions, solver_factors
from .adaptive import refinement_candidates
from .analysis.boundary_layer import boundary_layer_error
from .solver_settings import SolverSettings


def validate_coordinate_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Airfoil coordinate file not found: {path}")
    points = read_pairs(path)
    if len(points) < 5 or len(set(points)) < 4:
        raise ValueError(
            f"Airfoil needs at least five finite coordinate pairs: {path}"
        )
    xs, ys = zip(*points)
    if max(xs) <= min(xs) or max(ys) <= min(ys):
        raise ValueError(f"Airfoil has zero chord or thickness: {path}")
    # Do not silently discard nonfinite coordinate rows.
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            values = [float(v) for v in fields[:2]]
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Airfoil contains nonfinite coordinates: {path}")


def path_to_angle(start: float, target: float, step: float) -> list[float]:
    count = math.ceil(abs(target - start) / step)
    if count > 40000:
        raise ValueError(
            "Retry continuation requires too many intermediate angles"
        )
    direction = 1 if target >= start else -1
    return [start + direction * step * i for i in range(count)] + [target]


def solver_commands(
    config: dict,
    alphas: list[float],
    seed: float | None = None,
    retry_round: int = 0,
    graphics: bool = False,
    reset_boundary_layer: bool = False,
) -> list[str]:
    commands = [] if graphics else ["PLOP", "G", ""]
    flow_type = config.get("flow_type", 1)
    re_factor, mach_factor = solver_factors(config)
    settings = SolverSettings.from_config(config)
    commands += [
        (
            "LOAD airfoil_input.dat"
            if config["input"]["kind"] == "file"
            else f"NACA {config['input']['naca']}"
        )
    ]
    commands += (
        settings.panel_commands() if "solver_settings" in config else ["PANE"]
    )
    commands += [
        "PSAV geometry.dat",
        "OPER",
        "VISC",
        f"{config['re']:.10g}",
        "MACH",
        f"{config['mach']:.10g}",
        "TYPE",
        "1",
        "ITER",
        str(min(10000, config["iterations"] * 2**retry_round)),
    ]
    if "solver_settings" in config:
        commands += settings.transition_commands()
    warm = []
    if retry_round:
        seed = seed if seed is not None else 0.0
        warm = path_to_angle(0, seed, 1.0)
        warm += path_to_angle(
            seed, alphas[0], config["retry_step"] / 2 ** (retry_round - 1)
        )[1:-1]
    elif len(alphas) == 1 and abs(alphas[0]) > 1:
        warm = path_to_angle(0, alphas[0], 1.0)[:-1]
    if reset_boundary_layer and not warm:
        warm = [0.0]
    commands += [f"ALFA {alpha:.10g}" for alpha in warm]
    if flow_type != 1:
        # Initialize at ordinary Re/M before switching to CL-dependent factors.
        commands += [
            f"ALFA {alphas[0]:.10g}",
            "TYPE",
            str(flow_type),
            "RE",
            f"{re_factor:.10g}",
            "MACH",
            f"{mach_factor:.10g}",
        ]
    if reset_boundary_layer:
        # INIT toggles a flag, so first establish a viscous state above. Keep
        # the TYPE 1 initialization for CL-dependent Re/Mach before resetting
        # only the boundary layer at the requested operating condition.
        commands += ["INIT"]
    step = alphas[1] - alphas[0] if len(alphas) > 1 else 1
    commands += ["PACC", "polar.txt", ""]
    if config.get("boundary_layer") or config.get("explicit_alphas"):
        for index, alpha in enumerate(alphas, 1):
            commands += [
                f"ASEQ {alpha:.10g} {alpha:.10g} 1",
                f"CPWR cp_sequence_{index:04d}.txt",
            ]
            if config.get("boundary_layer"):
                commands += [f"DUMP bl_sequence_{index:04d}.txt"]
    else:
        commands += [
            "CSAV cp_sequence_",
            f"ASEQ {alphas[0]:.10g} {alphas[-1]:.10g} {step:.10g}",
        ]
    commands += ["PACC", "", "QUIT"]
    return commands


def execute(
    binary: Path,
    env: dict,
    work_dir: Path,
    commands: list[str],
    timeout: float,
    *,
    log_filename: str = "xfoil.log",
) -> tuple[str, int | None]:
    """Drain pipes with communicate; always reap XFOIL on timeout/interrupt."""
    text = "\n".join(commands) + "\n"
    atomic_text(work_dir / "commands.txt", text)
    with (work_dir / log_filename).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(binary)],
            cwd=work_dir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            process.communicate(text, timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return "timeout", process.returncode
        except KeyboardInterrupt:
            process.kill()
            process.communicate()
            return "interrupted", process.returncode
    return (
        "finished" if process.returncode == 0 else "solver_error"
    ), process.returncode


def solve(
    binary: Path,
    env: dict,
    root: Path,
    polar_path: Path,
    config: dict,
    previous_rows: list[dict],
    previous_attempts: list[dict],
    checkpoint: Callable[[list[dict], list[dict], str], None],
    graphics: bool = False,
) -> tuple[list[dict], list[dict], str]:
    requested = config["alphas"]
    rows = aligned_rows(previous_rows, requested)
    attempts = list(previous_attempts)
    deadline = time.monotonic() + config["timeout"]
    overall = "finished"
    attempt_root = root / "attempts"
    attempt_root.mkdir(exist_ok=True)

    def collect_outputs(directory, alphas, converged_rows, record):
        geometry = read_pairs(directory / "geometry.dat")
        if len(geometry) >= 3:
            shutil.copyfile(directory / "geometry.dat", root / "geometry.dat")
        errors = []
        for index, alpha in enumerate(alphas, 1):
            if row_for_alpha(converged_rows, alpha) is None:
                continue
            cp = directory / f"cp_sequence_{index:04d}.txt"
            if len(read_pairs(cp)) >= 3:
                shutil.copyfile(cp, root / cp_name(alpha))
            else:
                # A new converged polar row may come from a different BL
                # branch. Its pressure must never come from an older attempt.
                (root / cp_name(alpha)).unlink(missing_ok=True)
            if not config.get("boundary_layer"):
                continue
            bl = directory / f"bl_sequence_{index:04d}.txt"
            target = (
                root
                / "boundary_layer"
                / cp_name(alpha).replace("cp_", "bl_", 1)
            )
            error = boundary_layer_error(bl, geometry)
            if error:
                errors.append({"alpha": alpha, "error": error})
                # Never attach an earlier solution's boundary layer to this
                # newly accepted polar row if its own DUMP failed validation.
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(bl, target)
        if config.get("boundary_layer"):
            record["boundary_layer_errors"] = errors

    def run(
        alphas: list[float],
        retry_round: int = 0,
        seed: float | None = None,
        adaptive_round: int = 0,
        reset_boundary_layer: bool = False,
    ) -> str:
        nonlocal rows, overall
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            overall = "timeout"
            return overall
        number = (
            max(
                [
                    int(p.name)
                    for p in attempt_root.iterdir()
                    if p.is_dir() and p.name.isdigit()
                ]
                + [0]
            )
            + 1
        )
        directory = attempt_root / f"{number:04d}"
        directory.mkdir()
        if config["input"]["kind"] == "file":
            shutil.copyfile(
                root / "airfoil_input.dat", directory / "airfoil_input.dat"
            )
        record = {
            "directory": str(directory.relative_to(root)),
            "alphas": list(alphas),
            "retry_round": retry_round,
            "seed_alpha": seed,
            "status": "running",
            "adaptive_round": adaptive_round,
            "strategy": (
                "boundary_layer_reset"
                if reset_boundary_layer
                else "continuation" if retry_round else "initial_sweep"
            ),
            "initialization": (
                "INIT" if reset_boundary_layer else "continuation"
            ),
        }
        attempts.append(record)
        checkpoint(rows, attempts, "running")
        started = time.monotonic()
        try:
            status, code = execute(
                binary,
                env,
                directory,
                solver_commands(
                    config,
                    alphas,
                    seed,
                    retry_round,
                    graphics,
                    reset_boundary_layer,
                ),
                remaining,
            )
            record.update(status=status, returncode=code)
        except OSError as error:
            status = "solver_error"
            record.update(status=status, error=str(error))
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        raw = directory / "polar.txt"
        fresh = aligned_rows(
            (
                parse_polar_text(raw.read_text(errors="replace"))
                if raw.is_file()
                else []
            ),
            alphas,
        )
        fresh = [
            r for r in fresh if actual_conditions(config, r["CL"]) is not None
        ]
        record["converged_alphas"] = [r["alpha"] for r in fresh]
        merged = {r["alpha"]: r for r in rows}
        merged.update({r["alpha"]: r for r in fresh})
        rows = aligned_rows(list(merged.values()), requested)
        collect_outputs(directory, alphas, fresh, record)
        record["valid_output_alphas"] = [
            a for a in alphas if not output_missing(a)
        ]
        write_combined_polar(polar_path, rows, config)
        if status in {"timeout", "interrupted", "solver_error"}:
            overall = status
        checkpoint(rows, attempts, overall)
        return status

    def output_missing(a):
        return (
            row_for_alpha(rows, a) is None
            or len(read_pairs(root / cp_name(a))) < 3
            or (
                config.get("boundary_layer")
                and boundary_layer_error(
                    root
                    / "boundary_layer"
                    / cp_name(a).replace("cp_", "bl_", 1),
                    read_pairs(root / "geometry.dat"),
                )
                is not None
            )
        )

    if not attempts:
        status = run(requested)
        if status in {"interrupted", "timeout"}:
            return rows, attempts, status
    elif attempts:
        # Recover output from an abrupt stop before recomputing anything.
        for record in attempts:
            directory = root / record["directory"]
            raw = directory / "polar.txt"
            recovered = aligned_rows(
                (
                    parse_polar_text(raw.read_text(errors="replace"))
                    if raw.is_file()
                    else []
                ),
                record["alphas"],
            )
            recovered = [
                r
                for r in recovered
                if actual_conditions(config, r["CL"]) is not None
            ]
            merged = {r["alpha"]: r for r in rows}
            merged.update({r["alpha"]: r for r in recovered})
            rows = aligned_rows(list(merged.values()), requested)
            collect_outputs(directory, record["alphas"], recovered, record)
            if record["status"] == "running":
                record["status"] = "interrupted"
        write_combined_polar(polar_path, rows, config)
        checkpoint(rows, attempts, "running")

    def continuation_seed(alpha: float) -> float:
        candidates = rows
        previous_seed = next(
            (
                r["seed_alpha"]
                for r in reversed(attempts)
                if r.get("seed_alpha") is not None
                and any(abs(a - alpha) <= 0.00051 for a in r["alphas"])
            ),
            None,
        )
        if previous_seed is not None:
            # A failed approach can oscillate even at a larger iteration limit.
            # Prefer a converged point on the opposite side when available.
            opposite = [
                r
                for r in rows
                if (r["alpha"] - alpha) * (previous_seed - alpha) < 0
            ]
            candidates = opposite or rows
        return (
            min(candidates, key=lambda r: abs(r["alpha"] - alpha))["alpha"]
            if candidates
            else 0.0
        )

    def retry_missing(
        targets: list[float],
        rounds: int,
        start_round: int = 1,
        adaptive_round: int = 0,
    ) -> str | None:
        for retry_round in range(start_round, start_round + rounds):
            missing = [a for a in targets if output_missing(a)]
            for alpha in missing:
                status = run(
                    [alpha],
                    retry_round,
                    continuation_seed(alpha),
                    adaptive_round,
                )
                if status in {"interrupted", "timeout"}:
                    return status
        return None

    reset_attempted = set()

    def reset_missing(targets, retry_round, adaptive_round=0):
        if not config["retries"]:
            return None
        for alpha in targets:
            if not output_missing(alpha) or alpha in reset_attempted:
                continue
            # A new boundary-layer guess can escape a stalled Newton path.
            # One such fallback per angle and solve invocation is sufficient;
            # neither the physical inputs nor the convergence tolerance change.
            reset_attempted.add(alpha)
            status = run(
                [alpha],
                retry_round,
                continuation_seed(alpha),
                adaptive_round,
                reset_boundary_layer=True,
            )
            if status in {"interrupted", "timeout"}:
                return status
        return None

    # An incomplete run gets a fresh bounded retry budget on resume, even
    # with --retries 0.
    rounds = max(config["retries"], 1 if previous_attempts else 0)
    status = retry_missing(requested, rounds)
    if status:
        return rows, attempts, status
    status = reset_missing(requested, rounds)
    if status:
        return rows, attempts, status
    adaptive = config.get("adaptive", {})
    completed_round = max(
        (r.get("adaptive_round", 0) for r in attempts), default=0
    )
    for round_number in range(
        completed_round + 1, adaptive.get("rounds", 0) + 1
    ):
        additions = refinement_candidates(
            rows,
            requested,
            adaptive["max_points"],
            adaptive["min_step"],
            config.get("target_cl"),
        )
        if not additions:
            break
        missing_distances = {
            a: min((abs(r["alpha"] - a) for r in rows), default=math.inf)
            for a in requested
            if output_missing(a)
        }
        requested[:] = sorted(set(requested + additions))
        checkpoint(rows, attempts, "running")
        for alpha in additions:
            status = run([alpha], 1, continuation_seed(alpha), round_number)
            if status in {"interrupted", "timeout"}:
                return rows, attempts, status
        # Refinement happens after the initial sweep's retry loop. Give newly
        # requested points their configured retries within the same deadline.
        status = retry_missing(
            additions,
            config["retries"],
            start_round=2,
            adaptive_round=round_number,
        )
        if status:
            return rows, attempts, status
        status = reset_missing(additions, config["retries"] + 1, round_number)
        if status:
            return rows, attempts, status
        # Refinement may supply a closer converged seed for an older gap.
        # Revisit it only when that distance improves, at most once per round.
        for alpha, distance in missing_distances.items():
            closer = [
                a
                for a in additions
                if not output_missing(a) and abs(a - alpha) < distance
            ]
            if (
                not config["retries"]
                or not closer
                or not output_missing(alpha)
            ):
                continue
            status = run(
                [alpha],
                max(1, config["retries"]),
                min(closer, key=lambda a: abs(a - alpha)),
                round_number,
            )
            if status in {"interrupted", "timeout"}:
                return rows, attempts, status
    return rows, attempts, overall
