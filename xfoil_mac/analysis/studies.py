"""Numerical sensitivity and approach-direction checks.

Include sourced validation data.
"""

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

from ..data import atomic_json, parse_polar_file, row_for_alpha, sha256_file
from ..execution import execute, solver_commands
from ..runtime import xfoil_env
from ..solver_settings import SolverSettings
from ..results import write_records
from .report import report_page


def panel_study(
    app, destination, panels=(120, 240, 360), *, resume=False, **options
):
    from ..workflows import run_polar

    panels = sorted(set(panels))
    if not 2 <= len(panels) <= 8:
        raise ValueError("Choose 2..8 distinct panel counts")
    settings = options.pop("solver_settings", None) or SolverSettings()
    for n in panels:
        replace(settings, panels=n)
    rows, folders, statuses = {}, [], []
    for n in panels:
        folder = destination / "data" / f"panels_{n}"
        file = run_polar(
            app,
            out_file=folder / "polar.txt",
            solver_settings=replace(settings, panels=n),
            resume=resume and (folder / "run.json").exists(),
            **options,
        )
        rows[n] = parse_polar_file(file)
        statuses.append(
            json.loads((folder / "run.json").read_text())["status"]
        )
        folders.append(str(folder.relative_to(destination)))
    reference = rows[panels[-1]]
    records = []
    for n in panels:
        for row in rows[n]:
            ref = row_for_alpha(reference, row["alpha"])
            records.append(
                dict(
                    panels=n,
                    alpha=row["alpha"],
                    CL=row["CL"],
                    CD=row["CD"],
                    CM=row["CM"],
                    delta_CL=row["CL"] - ref["CL"] if ref else None,
                    delta_CD=row["CD"] - ref["CD"] if ref else None,
                    status="matched" if ref else "reference_missing",
                )
            )
    write_records(destination / "panel_sensitivity.csv", records)
    atomic_json(
        destination / "study.json",
        {
            "folders": folders,
            "kind": "panels",
            "status": (
                "ok" if all(s == "ok" for s in statuses) else "needs_attention"
            ),
        },
    )
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for n in panels:
        for ax, key in zip(axes, ("CL", "CD")):
            ax.scatter(
                [r["alpha"] for r in rows[n]],
                [r[key] for r in rows[n]],
                label=f"{n} panels",
                s=18,
            )
            ax.set(xlabel="alpha (deg)", ylabel=key)
            ax.grid(alpha=0.2)
    axes[0].legend()
    fig.savefig(destination / "panel_sensitivity.png", dpi=150)
    plt.close(fig)
    return report_page(
        destination / "report.html",
        "面板敏感性检查",
        f"差值相对于 {panels[-1]} 面板。只比较双方已收敛的点；接近不等于已经证明网格无关。",
        records,
        [("数据", "panel_sensitivity.csv")]
        + [(f"{n} 面板", f"data/panels_{n}/report.html") for n in panels],
        [("面板敏感性", "panel_sensitivity.png")],
    )


def check_directions(
    app,
    source: Path,
    destination: Path,
    angles=None,
    tolerance_cl=0.002,
    tolerance_cd=0.0002,
):
    from ..data import read_pairs

    source = source.expanduser().resolve()
    manifest = json.loads((source / "run.json").read_text())
    config = manifest["config"]
    rows = parse_polar_file(source / config["polar_filename"])
    angles = list(
        angles
        if angles is not None
        else manifest.get("summary", {}).get("failed_alphas", [])
    )
    if not angles and rows:
        angles = [
            max(
                rows,
                key=lambda r: r["CL"] / r["CD"] if r["CD"] > 0 else -math.inf,
            )["alpha"]
        ]
    if not 1 <= len(angles) <= 30 or not all(
        math.isfinite(a) and abs(a) < 89 for a in angles
    ):
        raise ValueError("Check 1..30 angles within (-89, 89)")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            "Use an empty directory for an independent direction check"
        )
    if config.get("binary_sha256") != sha256_file(app.xfoil_bin):
        raise ValueError(
            "Saved run used another binary; "
            "rerun with the current solver first"
        )
    records = []
    for i, alpha in enumerate(angles):
        pair = []
        for direction, offset in (("from_below", -1.0), ("from_above", 1.0)):
            folder = destination / "data" / f"point_{i+1:03d}" / direction
            folder.mkdir(parents=True)
            if config["input"]["kind"] == "file":
                shutil.copyfile(
                    source / "airfoil_input.dat", folder / "airfoil_input.dat"
                )
            commands = solver_commands(
                config, [alpha], seed=alpha + offset, retry_round=1
            )
            status, _ = execute(
                app.xfoil_bin,
                xfoil_env(app),
                folder,
                commands,
                config["timeout"],
            )
            row = row_for_alpha(parse_polar_file(folder / "polar.txt"), alpha)
            if (
                status != "finished"
                or len(read_pairs(folder / "cp_sequence_0001.txt")) < 3
            ):
                row = None
            pair.append(row)
            if status == "interrupted":
                raise KeyboardInterrupt
        complete = all(pair)
        dcl = abs(pair[0]["CL"] - pair[1]["CL"]) if complete else None
        dcd = abs(pair[0]["CD"] - pair[1]["CD"]) if complete else None
        records.append(
            dict(
                alpha=alpha,
                from_below=pair[0],
                from_above=pair[1],
                delta_CL=dcl,
                delta_CD=dcd,
                status=(
                    (
                        "consistent"
                        if dcl <= tolerance_cl and dcd <= tolerance_cd
                        else "direction_sensitive"
                    )
                    if complete
                    else "not_converged_both_directions"
                ),
            )
        )
        atomic_json(
            destination / "direction_check.json",
            {
                "source": str(source),
                "source_fingerprint": manifest["fingerprint"],
                "tolerance_CL": tolerance_cl,
                "tolerance_CD": tolerance_cd,
                "points": records,
            },
        )
    write_records(destination / "direction_check.csv", records)
    return report_page(
        destination / "report.html",
        "双向逼近检查",
        "分别从目标攻角的下方、上方逼近。差异保留在独立结果中，不覆盖原计算，不取平均填补失败点。",
        records,
        [("数据与原始路径", "direction_check.json")],
    )


def benchmark(
    app,
    destination,
    *,
    grit="80 grit",
    trip=0.05,
    max_alpha=10.2,
    resume=False,
    panels=240,
):
    """Compare measurements using an explicit roughness-trip approximation."""
    from ..workflows import run_polar
    from ..experiments import compare_experiment

    if (
        grit not in {"80 grit", "120 grit", "180 grit"}
        or not 0 <= max_alpha <= 15
    ):
        raise ValueError(
            "Choose one grit series and a maximum alpha "
            "between 0 and 15 degrees"
        )
    settings = SolverSettings(panels=panels, xtr_top=trip, xtr_bottom=trip)
    source = app.app_root / "examples/reference/ladson_naca0012.dat"
    measurements = []
    zone = None
    for line in source.read_text().splitlines():
        if line.startswith("zone"):
            zone = line.split('"')[1]
        elif zone == grit and line and not line.startswith("#"):
            alpha, cl, cd = map(float, line.split())
            if abs(alpha) <= max_alpha:
                measurements.append(
                    dict(
                        alpha=alpha,
                        CL=cl,
                        CD=cd,
                        Re=6e6,
                        Mach=0.15,
                        airfoil="NACA0012",
                    )
                )
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination / "reference_original.dat")
    write_records(destination / "reference.csv", measurements)
    provenance = {
        "source": (
            "https://tmbwg.github.io/turbmodels/"
            "NACA0012_validation/CLCD_Ladson_expdata.dat"
        ),
        "context": "https://tmbwg.github.io/turbmodels/naca0012_val.html",
        "report": "NASA TM 4074 (Ladson, 1988)",
        "sha256": sha256_file(source),
        "series": grit,
        "Re": 6e6,
        "Mach": 0.15,
        "max_abs_alpha": max_alpha,
        "model_assumptions": {
            "geometry": (
                "XFOIL original finite-TE NACA0012; "
                "not the modified sharp-TE TMR CFD geometry"
            ),
            "transition": (
                f"Forced Xtr={trip:g} on both sides is a user-visible "
                "modeling assumption; source table specifies grit, "
                "not a trip location. Grit roughness drag is not resolved."
            ),
        },
        "validation_status": "comparison_only_not_certification",
        "note": (
            "Experimental comparison is independent of software unit tests. "
            "High-angle separation and three-dimensional tunnel effects "
            "limit interpretation."
        ),
    }
    atomic_json(destination / "reference_provenance.json", provenance)
    folder = destination / "data"
    run_polar(
        app,
        naca="0012",
        reynolds=6e6,
        mach=0.15,
        solver_settings=settings,
        alpha_targets=[r["alpha"] for r in measurements],
        out_file=folder / "polar.txt",
        resume=resume and (folder / "run.json").exists(),
        show_xfoil_geometry=False,
        save_pressure_vectors=False,
        timeout=180,
        quiet=True,
    )
    comparison = compare_experiment(
        folder, destination / "reference.csv", destination
    )
    result = json.loads(comparison.read_text())
    return report_page(
        destination / "report.html",
        "NACA 0012 实测基准",
        "Ladson / NASA TM 4074，Re=6,000,000，M=0.15。仅使用一个粗糙带系列。"
        "Xtr 是近似假设；未模拟砂粒粗糙度阻力，报告误差不代表验证通过。"
        "来源、原始数据、几何与转捩假设见来源文件。",
        [dict(quantity=k, **v) for k, v in result["metrics"].items()],
        [
            ("来源和假设", "reference_provenance.json"),
            ("全部逐点误差", "experiment_matches.csv"),
            ("原始实验数据", "reference_original.dat"),
            ("计算与收敛", "data/report.html"),
        ],
        [(p, p) for p in result["plots"]],
    )
