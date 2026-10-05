"""Single-case and batch checkpoints, recovery, and result exports."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .comparison import write_comparison
from .data import (
    APP_VERSION,
    aligned_rows,
    alpha_values,
    atomic_json,
    atomic_text,
    cp_name,
    fingerprint,
    parse_numeric_pairs,
    parse_polar_file,
    read_pairs,
    row_for_alpha,
    safe_label,
    sha256_file,
    timestamp_label,
    validate_solver_options,
)
from .execution import solve, validate_coordinate_file
from .adaptive import validate_adaptive
from .analysis.boundary_layer import boundary_layer_error
from .flow import (
    actual_conditions,
    condition_label,
    solver_factors,
    validate_flow,
)
from .solver_settings import SolverSettings
from .plotting import (
    RESULT_PLOT_FILENAMES,
    read_cp_series,
    write_cp_polar_outputs,
    write_geometry_plot,
)
from .preview import (
    build_geometry_commands,
    build_operating_state_commands,
    run_xfoil_interactive_result,
)
from .reporting import append_summary, summarize_polar, write_polar_csv
from .runtime import (
    AppPaths,
    FlowAssumptions,
    batch_preview_enabled,
    configure_matplotlib_cache,
    resolve_flow_inputs,
    start_xquartz,
    xfoil_env,
)


def calculation_source_hashes() -> dict[str, str]:
    """Fingerprint the calculation implementation independently of UI edits."""
    source_root = Path(__file__).resolve().parent
    return {
        f"xfoil_mac/{name}": sha256_file(source_root / name)
        for name in (
            "data.py",
            "execution.py",
            "runtime.py",
            "workflows.py",
            "flow.py",
            "adaptive.py",
            "solver_settings.py",
            "analysis/boundary_layer.py",
        )
    }


def run_label(
    label: str,
    re_value: float,
    mach: float,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
) -> str:
    return (
        f"{timestamp_label()}_{safe_label(label)}"
        f"_Re{safe_label(f'{re_value:g}')}"
        f"_M{safe_label(f'{mach:g}')}_a{safe_label(f'{alpha_start:g}')}"
        f"_{safe_label(f'{alpha_end:g}')}_{safe_label(f'{alpha_step:g}')}"
    )


def batch_session_dir(
    app: AppPaths,
    re_value: float,
    mach: float,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
) -> Path:
    return app.batch_root / (
        f"{timestamp_label()}_batch_Re{safe_label(f'{re_value:g}')}"
        f"_M{safe_label(f'{mach:g}')}_a{safe_label(f'{alpha_start:g}')}"
        f"_{safe_label(f'{alpha_end:g}')}_{safe_label(f'{alpha_step:g}')}"
    )


VERDICT_TEXT = {
    "ok": "可信",
    "suspect": "超出有效域",
    "invalid": "求解失败",
    "retry": "孤立失败，可重试",
    "unchecked": "无证据",
}
"""Verdict wording used inside the report table.

Kept beside the workflow rather than inside the trust layer so the
report can be reworded without touching the judgement itself.
"""


def run_polar(
    app: AppPaths,
    naca: str = "2412",
    airfoil_file: Path | None = None,
    reynolds: float | None = None,
    mach: float | None = None,
    iterations: int = 200,
    alpha_start: float = -4,
    alpha_end: float = 12,
    alpha_step: float = 1,
    out_file: Path | None = None,
    csv_file: Path | None = None,
    summary_file: Path | None = None,
    airfoil_label: str | None = None,
    force: bool = False,
    keep_result_open: bool = False,
    show_xfoil_geometry: bool = True,
    save_pressure_vectors: bool = True,
    live_cpx: bool = False,
    quiet: bool = False,
    resume: bool = False,
    timeout: float = 120,
    retries: int = 2,
    retry_step: float = 0.5,
    target_cl: float | None = None,
    flow_type: int = 1,
    reference_cl: float = 1.0,
    adaptive_rounds: int = 0,
    adaptive_max_points: int = 100,
    adaptive_min_step: float = 0.125,
    assumptions: FlowAssumptions | None = None,
    solver_settings: SolverSettings | None = None,
    boundary_layer: bool = False,
    alpha_targets: list[float] | None = None,
) -> Path:
    validate_solver_options(iterations, timeout, retries, retry_step)
    alphas = alpha_values(alpha_start, alpha_end, alpha_step)
    if alpha_targets is not None:
        alphas = list(dict.fromkeys(float(a) for a in alpha_targets))
        if (
            not alphas
            or len(alphas) > 1000
            or not all(math.isfinite(a) and abs(a) <= 90 for a in alphas)
        ):
            raise ValueError(
                "Require 1..1000 finite alpha targets in [-90, 90]"
            )
        alpha_start, alpha_end = min(alphas), max(alphas)
    validate_flow(flow_type, reference_cl)
    validate_adaptive(
        adaptive_rounds, adaptive_max_points, adaptive_min_step, len(alphas)
    )
    assumptions = assumptions or FlowAssumptions.from_env()
    re_value, mach_value, flow_note, flow_relation = resolve_flow_inputs(
        reynolds, mach, assumptions
    )
    if flow_note and not quiet:
        print(flow_note)
    if target_cl is not None and not math.isfinite(target_cl):
        raise ValueError("Target CL must be finite")
    if force and resume:
        raise ValueError("--force and --resume cannot be combined")
    if resume and out_file is None:
        raise ValueError("Resume requires an explicit --out path")
    if airfoil_file:
        airfoil_file = airfoil_file.expanduser().resolve()
        validate_coordinate_file(airfoil_file)
        input_info = {
            "kind": "file",
            "path": str(airfoil_file),
            "sha256": sha256_file(airfoil_file),
        }
        input_name = str(airfoil_file)
        label = safe_label(airfoil_label or airfoil_file.stem)
    else:
        if not re.fullmatch(r"(?:\d{4}|\d{5})", naca):
            raise ValueError("NACA number must have four or five digits")
        input_info = {"kind": "naca", "naca": naca}
        input_name = f"NACA {naca}"
        label = safe_label(airfoil_label or f"NACA{naca}")
    out_file = (
        (
            out_file
            or app.single_root
            / run_label(
                label, re_value, mach_value, alpha_start, alpha_end, alpha_step
            )
            / "polar.txt"
        )
        .expanduser()
        .resolve()
    )
    root = out_file.parent
    summary_file = (
        (summary_file or root / "summary.csv").expanduser().resolve()
    )
    manifest_path = root / "run.json"
    source_hashes = calculation_source_hashes()
    config = {
        "input": input_info,
        "re": re_value,
        "mach": mach_value,
        "iterations": iterations,
        "alphas": alphas,
        "alpha_start": float(alpha_start),
        "alpha_end": float(alpha_end),
        "alpha_step": float(alpha_step),
        "timeout": float(timeout),
        "retries": retries,
        "retry_step": float(retry_step),
        "target_cl": float(target_cl) if target_cl is not None else None,
        "flow_type": flow_type,
        "flow_relation": flow_relation,
        "reference_cl": reference_cl,
        "adaptive": {
            "rounds": adaptive_rounds,
            "max_points": adaptive_max_points,
            "min_step": adaptive_min_step,
        },
        "flow_assumptions": vars(assumptions),
        "version": APP_VERSION,
        "source_sha256": source_hashes,
        "binary_sha256": sha256_file(app.xfoil_bin),
        "polar_filename": out_file.name,
        "save_pressure_vectors": save_pressure_vectors,
    }
    if solver_settings is not None:
        config["solver_settings"] = solver_settings.to_dict()
    if boundary_layer:
        config["boundary_layer"] = True
    if alpha_targets is not None:
        config["explicit_alphas"] = True
    config_id = fingerprint(config)
    solver_factors(config)
    previous = None
    if manifest_path.exists():
        if resume:
            previous = json.loads(manifest_path.read_text())
            if previous.get("fingerprint") != config_id:
                raise ValueError(
                    "Cannot resume: input, settings, or program changed: "
                    f"{manifest_path}"
                )
        elif not force:
            raise ValueError(f"Run exists: {root}; use --resume or --force")
    elif out_file.exists() and (resume or not force):
        raise ValueError(
            f"Existing result has no matching run.json: {out_file}; "
            "use a new directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    if force:
        for directory in (
            root / "attempts",
            root / "pressure_vectors",
            root / "preview",
            root / "boundary_layer",
        ):
            if directory.exists():
                shutil.rmtree(directory)
        for path in [
            out_file,
            manifest_path,
            root / "geometry.dat",
            root / "geometry.png",
            root / "airfoil_input.dat",
            *root.glob("cp_alpha_*.txt"),
            *[root / name for name in RESULT_PLOT_FILENAMES],
        ]:
            if path.exists() and path != airfoil_file:
                path.unlink()
    if airfoil_file and airfoil_file != root / "airfoil_input.dat":
        shutil.copyfile(airfoil_file, root / "airfoil_input.dat")
    configure_matplotlib_cache(app)
    # Calculation is bounded; human-paced replay follows durable results.
    graphics = (show_xfoil_geometry or live_cpx) and not keep_result_open
    if graphics or keep_result_open:
        start_xquartz()
    manifest = previous or {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "airfoil": label,
        "config": config,
        "fingerprint": config_id,
        "python_version": sys.version,
        "input_note": flow_note,
    }
    rows = parse_polar_file(out_file) if previous else []
    attempts = previous.get("attempts", []) if previous else []
    alphas = (
        list(previous.get("requested_alphas", alphas))
        if previous
        else list(alphas)
    )
    solve_config = dict(config, alphas=alphas)

    def checkpoint(
        current_rows: list[dict],
        current_attempts: list[dict],
        execution_status: str,
    ) -> None:
        summary = summarize_polar(current_rows, alphas, target_cl)
        manifest.update(
            updated_at=datetime.now().astimezone().isoformat(),
            stage="calculating",
            execution_status=execution_status,
            summary=summary,
            attempts=current_attempts,
            requested_alphas=list(alphas),
        )
        atomic_json(manifest_path, manifest)

    complete = (
        previous
        and previous.get("stage") == "complete"
        and previous.get("status") == "ok"
    )
    raw_complete = len(read_pairs(root / "geometry.dat")) >= 3 and all(
        len(read_pairs(root / cp_name(a))) >= 3
        and row_for_alpha(rows, a) is not None
        for a in alphas
    )
    if boundary_layer:
        geometry = read_pairs(root / "geometry.dat")
        raw_complete = raw_complete and all(
            boundary_layer_error(
                root / "boundary_layer" / cp_name(a).replace("cp_", "bl_", 1),
                geometry,
            )
            is None
            for a in alphas
        )
    if not (complete and raw_complete):
        checkpoint(rows, attempts, "running")
        rows, attempts, execution_status = solve(
            app.xfoil_bin,
            xfoil_env(app),
            root,
            out_file,
            solve_config,
            rows,
            attempts,
            checkpoint,
            graphics=graphics,
        )
    else:
        execution_status = previous.get("execution_status", "finished")
        if not quiet:
            print(
                "Resume: calculation already complete; "
                f"refreshing outputs for {label}."
            )
    rows = aligned_rows(rows, alphas)
    summary = summarize_polar(rows, alphas, target_cl)
    missing_cp = [
        r["alpha"]
        for r in rows
        if len(read_pairs(root / cp_name(r["alpha"]))) < 3
    ]
    status = (
        execution_status
        if execution_status in {"timeout", "solver_error", "interrupted"}
        else summary["status"]
    )
    if status == "ok" and (
        missing_cp or len(read_pairs(root / "geometry.dat")) < 3
    ):
        status = "incomplete_outputs"
    boundary_errors = []
    if boundary_layer:
        geometry = read_pairs(root / "geometry.dat")
        for row in rows:
            alpha = row["alpha"]
            error = boundary_layer_error(
                root
                / "boundary_layer"
                / cp_name(alpha).replace("cp_", "bl_", 1),
                geometry,
            )
            if error:
                boundary_errors.append({"alpha": alpha, "error": error})
    missing_bl = [item["alpha"] for item in boundary_errors]
    if missing_bl and status == "ok":
        status = "incomplete_outputs"
    manifest["missing_boundary_layer_alphas"] = missing_bl
    manifest["boundary_layer_errors"] = boundary_errors
    manifest["missing_cp_alphas"] = missing_cp
    manifest.update(
        summary=summary,
        status=status,
        execution_status=execution_status,
        attempts=attempts,
        stage="saving",
    )
    atomic_json(manifest_path, manifest)
    # Save CSV and convergence records before potentially expensive plotting.
    write_polar_csv(rows, root / "polar.csv")
    point_buffer = io.StringIO()
    point_writer = csv.writer(point_buffer)
    point_writer.writerow(["alpha", "CL", "Re", "Mach", "flow_type"])
    for row in rows:
        condition = actual_conditions(config, row["CL"])
        if condition:
            point_writer.writerow(
                [row["alpha"], row["CL"], *condition, flow_type]
            )
    atomic_text(root / "operating_points.csv", point_buffer.getvalue())
    if csv_file and csv_file.expanduser().resolve() != root / "polar.csv":
        write_polar_csv(rows, csv_file.expanduser().resolve())
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["alpha", "status", "cp_available", "attempt_count"])
    for alpha in alphas:
        writer.writerow(
            [
                alpha,
                "converged" if row_for_alpha(rows, alpha) else "missing",
                len(read_pairs(root / cp_name(alpha))) >= 3,
                sum(
                    any(abs(a - alpha) <= 0.00051 for a in attempt["alphas"])
                    for attempt in attempts
                ),
            ]
        )
    atomic_text(root / "convergence.csv", buffer.getvalue())
    append_summary(
        summary_file,
        label,
        input_name,
        out_file,
        root / "polar.csv",
        re_value,
        mach_value,
        alpha_start,
        alpha_end,
        alpha_step,
        iterations,
        rows,
        target_cl,
        status,
        requested_alphas=alphas,
        flow_type=flow_type,
        reference_cl=reference_cl,
    )
    if status == "interrupted":
        manifest["stage"] = "interrupted"
        atomic_json(manifest_path, manifest)
        raise KeyboardInterrupt
    try:
        geometry = parse_numeric_pairs(root / "geometry.dat")
        write_geometry_plot(geometry, root, label)
        title = (
            f"{label} | {condition_label(config)} | "
            f"{summary['points']}/{len(alphas)} converged"
        )
        outputs = write_cp_polar_outputs(
            read_cp_series([(a, root / cp_name(a)) for a in alphas]),
            geometry,
            rows,
            root,
            title,
            save_pressure_vectors=save_pressure_vectors,
            requested_alphas=alphas,
        )
    except Exception as error:
        manifest.update(stage="plot_error", error=str(error))
        atomic_json(manifest_path, manifest)
        raise
    manifest.update(
        stage="complete",
        outputs=[
            str(p.relative_to(root))
            for p in outputs["result_plots"] + outputs["pressure_vectors"]
        ],
        updated_at=datetime.now().astimezone().isoformat(),
    )
    atomic_json(manifest_path, manifest)
    from .analysis.diagnostics import diagnose
    from .analysis.boundary_layer import write_boundary_reports
    from .analysis.report import report_page

    diagnose(root)
    bl_index = write_boundary_reports(root, rows) if boundary_layer else []
    verdicts = None
    if boundary_layer:
        try:
            from .analysis.trust import summarise_points, sweep_report

            # The requested angles are passed in full, failures
            # included, because a failed solve writes no polar row and
            # the numbered dumps can only be paired positionally when
            # the whole request is known.
            verdicts = summarise_points(
                sweep_report(
                    root,
                    alphas,
                    summary.get("failed_alphas") or [],
                )["points"]
            )
        except Exception as error:  # noqa: BLE001
            # A verdict is an extra reading of the same run. Losing it
            # must not lose the run.
            manifest.update(trust_error=str(error))
            atomic_json(manifest_path, manifest)
            verdicts = None
    links = [
        ("数值数据", "polar.csv"),
        ("收敛与重试诊断", "diagnostics.html"),
        ("计算设置与原始记录", "run.json"),
    ]
    links += [
        (f"边界层 α={p['alpha']:g}°", f"boundary_layer/{p['plot']}")
        for p in bl_index
    ]
    report_rows = rows
    if verdicts:
        by_alpha = verdicts.get("by_alpha") or {}
        report_rows = [
            dict(
                row,
                可信度=VERDICT_TEXT.get(
                    by_alpha.get(round(float(row["alpha"]), 6)), "—"
                ),
            )
            for row in rows
        ]
    report_page(
        root / "report.html",
        label,
        f"{condition_label(config)} · {status} · {len(rows)}/{len(alphas)} "
        "个点。原始尝试和日志保留在 attempts 中。",
        report_rows,
        links,
        [(p.name, p.name) for p in outputs["result_plots"]],
        trust=verdicts,
    )
    if not quiet:
        print(
            f"{label}: {status}; {summary['points']}/{len(alphas)} converged "
            f"({summary['convergence_rate']:.1%})"
        )
        if summary["failed_alphas"]:
            print(f"Missing angles: {summary['failed_alphas']}")
        print(f"Results and raw data: {root}")
    if keep_result_open and rows:
        preview = root / "preview"
        preview.mkdir(exist_ok=True)
        if airfoil_file:
            shutil.copyfile(
                root / "airfoil_input.dat", preview / "airfoil_input.dat"
            )
        load_command = (
            "LOAD airfoil_input.dat" if airfoil_file else f"NACA {naca}"
        )
        commands = build_geometry_commands(load_command)
        if solver_settings is not None:
            commands += solver_settings.panel_commands()
        commands += build_operating_state_commands(
            re_value, mach_value, iterations
        )
        if solver_settings is not None:
            commands += solver_settings.transition_commands()
        if flow_type != 1:
            re_factor, mach_factor = solver_factors(config)
            commands += [
                f"ALFA {rows[0]['alpha']:.10g}",
                "TYPE",
                str(flow_type),
                "RE",
                f"{re_factor:.10g}",
                "MACH",
                f"{mach_factor:.10g}",
            ]
        commands += [f"ALFA {row['alpha']:.10g}" for row in rows]
        print(
            "Saved results are complete. Native replay recomputes only angles "
            "with saved converged results."
        )
        run_xfoil_interactive_result(
            app, preview, commands, [r["alpha"] for r in rows]
        )
    return out_file


def run_batch(
    app: AppPaths,
    folder: Path,
    reynolds: float | None,
    mach: float | None,
    iterations: int,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    out_dir: Path | None,
    force: bool,
    resume: bool = False,
    timeout: float = 120,
    retries: int = 2,
    retry_step: float = 0.5,
    target_cl: float | None = None,
    headless: bool = False,
    flow_type: int = 1,
    reference_cl: float = 1.0,
    adaptive_rounds: int = 0,
    adaptive_max_points: int = 100,
    adaptive_min_step: float = 0.125,
    assumptions: FlowAssumptions | None = None,
    solver_settings: SolverSettings | None = None,
    boundary_layer: bool = False,
) -> Path:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Airfoil database folder not found: {folder}")
    validate_solver_options(iterations, timeout, retries, retry_step)
    alpha_values(alpha_start, alpha_end, alpha_step)
    assumptions = assumptions or FlowAssumptions.from_env()
    re_value, mach_value, flow_note, flow_relation = resolve_flow_inputs(
        reynolds, mach, assumptions
    )
    if flow_note:
        print(flow_note)
    if target_cl is not None and not math.isfinite(target_cl):
        raise ValueError("Target CL must be finite")
    if force and resume:
        raise ValueError("--force and --resume cannot be combined")
    if resume and out_dir is None:
        raise ValueError("Batch resume requires an explicit --outdir")
    out_dir = (
        (
            out_dir
            or batch_session_dir(
                app, re_value, mach_value, alpha_start, alpha_end, alpha_step
            )
        )
        .expanduser()
        .resolve()
    )
    files = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".dat", ".txt"}
    )
    if not files:
        raise ValueError(f"No airfoil coordinate files found: {folder}")
    if out_dir.exists() and any(out_dir.iterdir()) and not (force or resume):
        raise ValueError(
            f"Batch directory is not empty: {out_dir}; use --resume or --force"
        )
    labels = [safe_label(p.stem) for p in files]
    labels = [
        (
            label
            if labels.count(label) == 1
            else f"{label}_{fingerprint({'file': p.name})[:8]}"
        )
        for label, p in zip(labels, files)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        out_dir / "batch.json",
        {
            "cases": labels,
            "source_folder": str(folder),
            "re": re_value,
            "mach": mach_value,
            "flow_assumptions": vars(assumptions),
        },
    )
    summary_file = out_dir / "summary.csv"
    if force and summary_file.exists():
        summary_file.unlink()
    elif resume and summary_file.exists():
        current_paths = {
            str(out_dir / label / "polar.txt") for label in labels
        }
        with summary_file.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            records = [
                record
                for record in reader
                if record.get("polar_file") in current_paths
            ]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        atomic_text(summary_file, buffer.getvalue())
    from tqdm import tqdm

    failed = 0
    with tqdm(
        total=len(files), desc="Airfoils", unit="airfoil", dynamic_ncols=True
    ) as progress:
        for file_path, label in zip(files, labels):
            progress.set_postfix_str(label)
            polar = out_dir / label / "polar.txt"
            try:
                run_polar(
                    app,
                    airfoil_file=file_path,
                    reynolds=re_value,
                    mach=mach_value,
                    iterations=iterations,
                    alpha_start=alpha_start,
                    alpha_end=alpha_end,
                    alpha_step=alpha_step,
                    out_file=polar,
                    summary_file=summary_file,
                    airfoil_label=label,
                    force=force,
                    resume=resume,
                    timeout=timeout,
                    retries=retries,
                    retry_step=retry_step,
                    target_cl=target_cl,
                    flow_type=flow_type,
                    reference_cl=reference_cl,
                    adaptive_rounds=adaptive_rounds,
                    adaptive_max_points=adaptive_max_points,
                    adaptive_min_step=adaptive_min_step,
                    assumptions=assumptions,
                    solver_settings=solver_settings,
                    boundary_layer=boundary_layer,
                    show_xfoil_geometry=not headless
                    and batch_preview_enabled(),
                    quiet=True,
                )
                saved = json.loads((polar.parent / "run.json").read_text())
                failed += saved["status"] != "ok"
            except (ValueError, OSError, RuntimeError, SystemExit) as error:
                failed += 1
                tqdm.write(f"{label}: {error}")
                # Preserve valid artifacts on incompatible resume;
                # report the refusal separately.
                atomic_json(
                    out_dir / f"{label}.failure.json",
                    {
                        "airfoil": label,
                        "input_file": str(file_path),
                        "re": re_value,
                        "mach": mach_value,
                        "status": "failed",
                        "error": str(error),
                        "updated_at": datetime.now().astimezone().isoformat(),
                    },
                )
                append_summary(
                    summary_file,
                    label,
                    str(file_path),
                    polar,
                    None,
                    re_value,
                    mach_value,
                    alpha_start,
                    alpha_end,
                    alpha_step,
                    iterations,
                    [],
                    target_cl,
                    "failed",
                    str(error),
                    flow_type=flow_type,
                    reference_cl=reference_cl,
                )
            else:
                (out_dir / f"{label}.failure.json").unlink(missing_ok=True)
            progress.update(1)
    print(
        f"Batch finished: {len(files) - failed}/{len(files)} complete; "
        f"{failed} need attention. {summary_file}"
    )
    write_comparison([out_dir], out_dir / "comparison", target_cl)
    return summary_file
