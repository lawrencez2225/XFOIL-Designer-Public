"""Direct specified-lift solves, distinct from alpha-sweep interpolation."""

import json
import math
import re
import shutil
import time
from pathlib import Path

from ..data import (
    atomic_json,
    fingerprint,
    parse_polar_file,
    sha256_file,
    cp_name,
    read_pairs,
)
from ..execution import execute, solver_commands, validate_coordinate_file
from ..runtime import (
    FlowAssumptions,
    resolve_flow_inputs,
    xfoil_env,
    configure_matplotlib_cache,
)
from ..solver_settings import SolverSettings
from ..results import write_records
from .boundary_layer import boundary_layer_error
from .report import report_page


def _read_lift_outputs(folder, target_cl, boundary_layer):
    """Validate the CL solution and all requested raw outputs."""
    rows = parse_polar_file(folder / "polar.txt")
    matches = [row for row in rows if abs(row["CL"] - target_cl) <= 0.00015]
    errors = []
    if len(matches) != 1:
        errors.append(
            f"Expected one converged polar row matching CL={target_cl:g}"
        )
    if len(read_pairs(folder / "cp.txt")) < 3:
        errors.append("Pressure output has fewer than three finite records")
    geometry = read_pairs(folder / "geometry.dat")
    if len(geometry) < 3:
        errors.append("Panel geometry has fewer than three finite records")
    if boundary_layer:
        error = boundary_layer_error(folder / "boundary_layer.txt", geometry)
        if error:
            errors.append(error)
    return matches, errors


def run_lift(
    app,
    targets,
    destination: Path,
    *,
    naca="2412",
    airfoil=None,
    reynolds=None,
    mach=None,
    assumptions=None,
    solver_settings=None,
    iterations=200,
    retries=2,
    timeout=120,
    resume=False,
    boundary_layer=True,
):
    from ..data import validate_solver_options

    validate_solver_options(iterations, timeout, retries, 0.5)
    targets = list(dict.fromkeys(float(v) for v in targets))
    if (
        not targets
        or len(targets) > 200
        or not all(math.isfinite(v) and abs(v) <= 5 for v in targets)
    ):
        raise ValueError(
            "Supply 1..200 finite target CL values between -5 and 5"
        )
    if airfoil is None and not re.fullmatch(r"(?:\d{4}|\d{5})", naca):
        raise ValueError("NACA must have four or five digits")
    assumptions = assumptions or FlowAssumptions.from_env()
    settings = solver_settings or SolverSettings()
    reynolds, mach, note, relation = resolve_flow_inputs(
        reynolds, mach, assumptions
    )
    source = {"kind": "naca", "naca": naca}
    if airfoil is not None:
        airfoil = Path(airfoil).expanduser().resolve()
        validate_coordinate_file(airfoil)
        source = {
            "kind": "file",
            "path": str(airfoil),
            "sha256": sha256_file(airfoil),
        }
    from ..workflows import calculation_source_hashes

    config = {
        "input": source,
        "re": reynolds,
        "mach": mach,
        "iterations": iterations,
        "retry_step": 0.5,
        "flow_type": 1,
        "solver_settings": settings.to_dict(),
        "flow_assumptions": vars(assumptions),
        "flow_relation": relation,
        "targets": targets,
        "retries": retries,
        "timeout": timeout,
        "boundary_layer": boundary_layer,
        "binary_sha256": sha256_file(app.xfoil_bin),
        "source_hashes": calculation_source_hashes(),
        "lift_implementation": sha256_file(Path(__file__)),
    }
    destination = destination.expanduser().resolve()
    manifest = destination / "lift_run.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else None
    if previous and (
        not resume or previous["fingerprint"] != fingerprint(config)
    ):
        raise ValueError(
            "Lift run exists or its settings changed; resume matching "
            "settings or choose a new folder"
        )
    if not previous and destination.exists() and any(destination.iterdir()):
        raise ValueError("Use an empty destination for a new lift study")
    destination.mkdir(parents=True, exist_ok=True)
    configure_matplotlib_cache(app)
    saved = previous or {
        "config": config,
        "fingerprint": fingerprint(config),
        "points": [],
        "input_note": note,
    }
    deadline = time.monotonic() + timeout
    try:
        for index, target in enumerate(targets):
            point = (
                saved["points"][index]
                if index < len(saved["points"])
                else {"target_cl": target, "attempts": [], "status": "pending"}
            )
            if index == len(saved["points"]):
                saved["points"].append(point)
            if point.get("status") == "ok" and point.get("attempts"):
                folder = destination / point["attempts"][-1]["directory"]
                _, errors = _read_lift_outputs(folder, target, boundary_layer)
                hashes = point.get("raw_hashes", {})
                required = ["polar.txt", "cp.txt", "geometry.dat"]
                if boundary_layer:
                    required.append("boundary_layer.txt")
                if any(
                    str((folder / name).relative_to(destination)) not in hashes
                    for name in required
                ):
                    errors.append("Required raw output hashes are missing")
                for path, digest in hashes.items():
                    raw = destination / path
                    if not raw.is_file() or sha256_file(raw) != digest:
                        errors.append(
                            f"Raw output is missing or changed: {path}"
                        )
                if not errors:
                    continue
                point["resume_output_errors"] = errors
            point.pop("result", None)
            point.pop("raw_hashes", None)
            for retry in range(retries + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    point["status"] = "timeout"
                    break
                folder = (
                    destination
                    / "raw"
                    / f"point_{index+1:03d}"
                    / f"attempt_{len(point['attempts'])+1:03d}"
                )
                folder.mkdir(parents=True)
                if airfoil:
                    shutil.copyfile(airfoil, folder / "airfoil_input.dat")
                commands = solver_commands(config, [0.0], retry_round=retry)
                commands = commands[: commands.index("PACC")]
                commands += ["ALFA 0"]
                step = 0.2 / 2**retry
                count = max(1, math.ceil(abs(target) / step))
                commands += [f"CL {target*j/count:.10g}" for j in range(count)]
                commands += [
                    "PACC",
                    "polar.txt",
                    "",
                    f"CL {target:.10g}",
                    "CPWR cp.txt",
                ]
                if boundary_layer:
                    commands += ["DUMP boundary_layer.txt"]
                commands += ["PACC", "", "QUIT"]
                record = {
                    "directory": str(folder.relative_to(destination)),
                    "retry": retry,
                    "status": "running",
                }
                point["attempts"].append(record)
                saved["status"] = "running"
                atomic_json(manifest, saved)
                status, code = execute(
                    app.xfoil_bin, xfoil_env(app), folder, commands, remaining
                )
                matches, errors = _read_lift_outputs(
                    folder, target, boundary_layer
                )
                valid = status == "finished" and not errors
                record.update(
                    status=status,
                    returncode=code,
                    matched=valid,
                    output_errors=errors,
                )
                if valid:
                    point["status"] = "ok"
                elif status != "finished":
                    point["status"] = status
                else:
                    point["status"] = (
                        "incomplete_outputs"
                        if len(matches) == 1
                        else "not_converged"
                    )
                if valid:
                    point["result"] = matches[0]
                    point["raw_hashes"] = {
                        str(p.relative_to(destination)): sha256_file(p)
                        for p in folder.iterdir()
                        if p.suffix in (".txt", ".dat")
                    }
                    break
                if status in {"timeout", "interrupted"}:
                    break
            atomic_json(manifest, saved)
            if point.get("status") == "interrupted":
                raise KeyboardInterrupt
    finally:
        saved["status"] = (
            "ok"
            if len(saved["points"]) == len(targets)
            and all(p.get("status") == "ok" for p in saved["points"])
            else "needs_attention"
        )
        atomic_json(manifest, saved)
    records = [
        dict(
            target_cl=p["target_cl"], status=p["status"], **p.get("result", {})
        )
        for p in saved["points"]
    ]
    write_records(destination / "lift_results.csv", records)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    valid = [r for r in records if r["status"] == "ok"]
    for ax, key in zip(axes, ("alpha", "CD")):
        ax.plot([r["target_cl"] for r in valid], [r[key] for r in valid], "o")
        ax.set(xlabel="Specified CL", ylabel=key)
        ax.grid(alpha=0.2)
    fig.savefig(destination / "lift_results.png", dpi=150)
    plt.close(fig)
    detail_links = []
    from ..plotting import write_cp_distribution_plot
    from .boundary_layer import write_boundary_reports

    for point in saved["points"]:
        if point.get("status") != "ok":
            if point["attempts"]:
                detail_links.append(
                    (
                        f"CL={point['target_cl']:g} 失败日志",
                        point["attempts"][-1]["directory"] + "/xfoil.log",
                    )
                )
            continue
        folder = destination / point["attempts"][-1]["directory"]
        row = point["result"]
        plot = write_cp_distribution_plot(
            [(row["alpha"], read_pairs(folder / "cp.txt"))],
            folder,
            f"Specified CL={point['target_cl']:g}",
        )
        detail_links.append(
            (
                f"CL={point['target_cl']:g} 压力分布",
                str(plot.relative_to(destination)),
            )
        )
        if boundary_layer:
            (folder / "boundary_layer").mkdir(exist_ok=True)
            shutil.copyfile(
                folder / "boundary_layer.txt",
                folder
                / "boundary_layer"
                / cp_name(row["alpha"]).replace("cp_", "bl_", 1),
            )
            for item in write_boundary_reports(folder, [row]):
                detail_links.append(
                    (
                        f"CL={point['target_cl']:g} 边界层",
                        str(
                            (
                                folder / "boundary_layer" / item["plot"]
                            ).relative_to(destination)
                        ),
                    )
                )
    report_page(
        destination / "report.html",
        "按目标 CL 直接求解",
        "每个点由 XFOIL 的 CL 命令直接求解；仅接受实际 CL 与目标匹配且已收敛的记录。",
        records,
        [("数据", "lift_results.csv"), ("配置与原始记录", "lift_run.json")]
        + detail_links,
        [("目标 CL 求解结果", "lift_results.png")],
    )
    return manifest
