"""Bounded, fully recorded shape search.

All aerodynamics come from real solves.
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np

from ..data import atomic_json, atomic_text, fingerprint, sha256_file
from ..geometry import inspect_points, load_coordinates
from ..wing_geometry import section_profile
from ..execution import execute
from ..runtime import xfoil_env, FlowAssumptions
from ..solver_settings import SolverSettings
from ..results import write_records
from .lift import run_lift
from .report import report_page


def profile_source(source):
    if "naca" in source:
        code = str(source["naca"])
        if len(code) != 4 or not code.isdigit():
            raise ValueError(
                "Shape design needs a four-digit NACA or a coordinate file"
            )
        return section_profile({"naca": code}, 101)
    report, points = inspect_points(
        load_coordinates(Path(source["airfoil"]).expanduser())
    )
    if not report["valid"]:
        raise ValueError(str(report["issues"]))
    return section_profile({"coordinates": points}, 101)


def make_shape(
    app,
    folder,
    source,
    *,
    thickness=1.0,
    camber=1.0,
    flap=0.0,
    hinge=0.75,
    blend_source=None,
    blend=0.0,
):
    """Modify a unit-chord profile within the documented search bounds.

    Thickness and camber are dimensionless multipliers of the blended
    profile. Flap is a deflection in degrees using XFOIL's GDES sign
    convention; hinge is its chordwise x/c location on the camber line.
    Blend is the fraction of the second profile, on a common x/c grid.
    These geometric changes alone imply no attached-flow validity.
    """
    if not all(
        math.isfinite(v) for v in (thickness, camber, flap, hinge, blend)
    ) or not (
        0.5 <= thickness <= 1.8
        and 0 <= camber <= 2
        and abs(flap) <= 15
        and 0.5 <= hinge <= 0.95
        and 0 <= blend <= 1
    ):
        raise ValueError(
            "Shape limits: thickness .5..1.8, camber 0..2, "
            "flap ±15°, hinge .5...95, blend 0..1"
        )
    if blend and blend_source is None:
        raise ValueError("Blending requires a second profile")
    points = profile_source(source)
    if blend_source is not None:
        points = (1 - blend) * points + blend * profile_source(blend_source)
    leading = int(np.argmin(points[:, 0]))
    x = points[: leading + 1, 0][::-1]
    upper = points[: leading + 1, 1][::-1]
    lower = np.r_[points[leading, 1], points[leading + 1 :, 1]]
    center = (upper + lower) * 0.5 * camber
    half = (upper - lower) * 0.5 * thickness
    modified = np.column_stack(
        (
            np.r_[x[::-1], x[1:]],
            np.r_[(center + half)[::-1], (center - half)[1:]],
        )
    )
    folder.mkdir(parents=True, exist_ok=True)
    source_path = folder / "base.dat"
    atomic_text(
        source_path,
        "Modified profile\n"
        + "".join(f"{a:.10g} {b:.10g}\n" for a, b in modified),
    )
    if flap:
        (folder / "flapped.dat").unlink(missing_ok=True)
        # XFOIL inserts/rebuilds the hinge region using its own GDES algorithm.
        status, _ = execute(
            app.xfoil_bin,
            xfoil_env(app),
            folder,
            [
                "PLOP",
                "G",
                "",
                "LOAD base.dat",
                "GDES",
                f"FLAP {hinge:.10g} "
                f"{float(np.interp(hinge, x, center)):.10g} {flap:.10g}",
                "EXEC",
                "",
                "SAVE flapped.dat",
                "QUIT",
            ],
            30,
        )
        if status == "interrupted":
            raise KeyboardInterrupt
        if status != "finished" or not (folder / "flapped.dat").is_file():
            raise ValueError(
                "Native flap generation failed; inspect xfoil.log"
            )
        modified = load_coordinates(folder / "flapped.dat")
    report, normalized = inspect_points(modified)
    atomic_json(folder / "geometry_check.json", report)
    if not report["valid"]:
        raise ValueError(
            f"Generated geometry failed inspection: {report['issues']}"
        )
    output = folder / "airfoil.dat"
    atomic_text(
        output,
        "Designed profile (normalized)\n"
        + "".join(f"{a:.10g} {b:.10g}\n" for a, b in normalized),
    )
    return output, {
        k: report[k]
        for k in (
            "thickness_ratio",
            "thickness_x",
            "max_camber",
            "trailing_edge_gap",
        )
    }


PRESCREEN_ALPHAS = tuple(float(v) for v in range(-4, 15))


def load_surrogate(root: Path | None = None):
    """The most recent trained surrogate, or None if there is none."""
    folder = (Path(root) if root else Path("xfoil_runs")) / "models"
    latest = folder / "surrogate_latest.joblib"
    if not latest.is_file():
        return None
    try:
        from . import surrogate

        return surrogate.load(latest)
    except Exception:  # noqa: BLE001
        return None


def prescreen_candidate(loaded, shape, conditions):
    """Predict every condition for one candidate without solving.

    Returns one entry per condition. A condition whose lift cannot be
    reached inside the swept range, or whose inputs fall outside the
    training data, is marked as such rather than given a number, because
    a tree returns a value everywhere and cannot decline.
    """
    from . import surrogate

    entries = []
    for condition in conditions:
        target = float(condition["cl"])
        try:
            predicted = surrogate.predict(
                loaded,
                coordinates=Path(shape),
                re=float(condition["re"]),
                mach=float(condition["mach"]),
                alphas=list(PRESCREEN_ALPHAS),
            )
        except (ValueError, OSError):
            entries.append(
                {
                    "target_cl": target,
                    "usable": False,
                    "reason": "shape could not be read",
                }
            )
            continue
        lift = [p["CL"] for p in predicted]
        if not lift or min(lift) > target or max(lift) < target:
            entries.append(
                {
                    "target_cl": target,
                    "usable": False,
                    "reason": "target lift outside the swept range",
                }
            )
            continue
        best = min(predicted, key=lambda p: abs(p["CL"] - target))
        if abs(best["CL"] - target) > 0.05:
            entries.append(
                {
                    "target_cl": target,
                    "usable": False,
                    "reason": "no angle close to the target lift",
                }
            )
            continue
        entries.append(
            {
                "target_cl": target,
                "usable": True,
                "extrapolates": bool(best["extrapolates"]),
                "note": "; ".join(best["outside"]),
                "alpha": best["alpha"],
                "CL": best["CL"],
                "CD": best["CD"],
                "CM": best["CM"],
                "lift_to_drag": (
                    best["CL"] / best["CD"] if best["CD"] > 0 else None
                ),
            }
        )
    return entries


def pareto_front(records):
    valid = [r for r in records if r.get("feasible")]
    return [
        r["candidate"]
        for r in valid
        if not any(
            s["mean_CD"] <= r["mean_CD"]
            and s["worst_LD"] >= r["worst_LD"]
            and (s["mean_CD"] < r["mean_CD"] or s["worst_LD"] > r["worst_LD"])
            for s in valid
        )
    ]


def design_search(app, config, destination, resume=False):
    source = config.get("source", {"naca": "2412"})
    blend_source = config.get("blend_source")
    variables = config.get(
        "variables",
        {"thickness": [1.0], "camber": [0.8, 1.0, 1.2], "flap": [0.0]},
    )
    if set(variables) - {"thickness", "camber", "flap", "blend"}:
        raise ValueError("Unknown design variable")
    arrays = [
        variables.get(k, [0.0] if k in {"flap", "blend"} else [1.0])
        for k in ("thickness", "camber", "flap", "blend")
    ]
    if any(not isinstance(a, list) or not a or len(a) > 10 for a in arrays):
        raise ValueError("Each variable needs 1..10 choices")
    candidates = list(itertools.product(*arrays))
    conditions = config.get(
        "conditions",
        [
            {"re": 1e6, "mach": 0.05, "cl": 0.5},
            {"re": 5e5, "mach": 0.03, "cl": 0.8},
        ],
    )
    if (
        len(candidates) > 60
        or not 1 <= len(conditions) <= 8
        or len(candidates) * len(conditions) > 160
    ):
        raise ValueError(
            "Search limit: 60 candidates, 8 conditions, 160 total solves"
        )
    constraints = config.get("constraints", {})
    min_t = float(constraints.get("min_thickness", 0.08))
    max_cm = float(constraints.get("max_abs_cm", 0.2))
    if not (0 < min_t < 0.4 and 0 < max_cm < 2):
        raise ValueError("Invalid thickness or pitching-moment constraint")
    for condition in conditions:
        if not 0 < float(condition["cl"]) <= 2:
            raise ValueError("Design objectives require target CL in (0, 2]")
    settings = SolverSettings(**config.get("solver_settings", {}))
    assumptions = FlowAssumptions(**config.get("flow", {}))
    from ..workflows import calculation_source_hashes

    identity = {
        "calculation_sources": calculation_source_hashes(),
        "lift_source": sha256_file(Path(__file__).with_name("lift.py")),
        "config": config,
        "implementation": sha256_file(Path(__file__)),
        "binary": sha256_file(app.xfoil_bin),
        "sources": [
            sha256_file(Path(s["airfoil"]).expanduser())
            for s in (source, blend_source)
            if s and "airfoil" in s
        ],
    }
    file = destination / "design_search.json"
    previous = json.loads(file.read_text()) if file.is_file() else None
    if previous and (
        not resume or previous["fingerprint"] != fingerprint(identity)
    ):
        raise ValueError("Design settings changed; use a new folder")
    if not previous and destination.exists() and any(destination.iterdir()):
        raise ValueError("Use an empty folder")
    destination.mkdir(parents=True, exist_ok=True)
    saved = {
        "fingerprint": fingerprint(identity),
        "identity": identity,
        "candidates": [],
        "status": "running",
    }
    prescreen_model = load_surrogate() if config.get("prescreen") else None
    if config.get("prescreen") and prescreen_model is None:
        raise ValueError(
            "Prescreening was requested but no trained model exists. "
            "Run: python surrogate_tool.py train"
        )
    for index, values in enumerate(candidates, 1):
        params = dict(zip(("thickness", "camber", "flap", "blend"), values))
        record = {"candidate": index, **params, "feasible": False}
        folder = destination / "data" / f"candidate_{index:03d}"
        try:
            shape, metrics = make_shape(
                app,
                folder / "geometry",
                source,
                **params,
                hinge=config.get("hinge", 0.75),
                blend_source=blend_source,
            )
            record.update(metrics)
            predicted = None
            if prescreen_model is not None and config.get("prescreen", False):
                predicted = prescreen_candidate(
                    prescreen_model, shape, conditions
                )
                record["surrogate"] = predicted
                scored = [
                    p
                    for p in predicted
                    if p.get("usable") and p.get("lift_to_drag") is not None
                ]
                if len(scored) != len(conditions):
                    reasons = sorted(
                        {
                            p.get("reason", "or lift not reached")
                            for p in predicted
                            if not p.get("usable")
                        }
                    )
                    raise ValueError(
                        "Prescreen rejected: " + "; ".join(reasons)
                    )
                # Extrapolated candidates are ranked and kept, not thrown
                # away, but they are marked so a shortlist can prefer the
                # ones the model has actually seen. Discarding them would
                # hide a usable answer behind a flag that a measured case
                # showed to be harmless.
                record["predicted_worst_LD"] = min(
                    p["lift_to_drag"] for p in scored
                )
                record["surrogate_extrapolates"] = any(
                    p.get("extrapolates") for p in scored
                )
            points = []
            for j, c in enumerate(conditions, 1):
                out = folder / f"condition_{j:02d}"
                manifest = run_lift(
                    app,
                    [c["cl"]],
                    out,
                    airfoil=shape,
                    reynolds=c["re"],
                    mach=c["mach"],
                    assumptions=assumptions,
                    solver_settings=settings,
                    resume=resume and (out / "lift_run.json").exists(),
                    timeout=config.get("timeout", 90),
                    boundary_layer=bool(config.get("boundary_layer", False)),
                    iterations=int(config.get("iterations", 200)),
                    retries=int(config.get("retries", 2)),
                )
                point = json.loads(manifest.read_text())["points"][0]
                if point["status"] != "ok":
                    raise ValueError(f"Condition {j}: {point['status']}")
                points.append(point["result"])
            record.update(
                mean_CD=sum(r["CD"] for r in points) / len(points),
                worst_LD=min(r["CL"] / r["CD"] for r in points if r["CD"] > 0),
                max_abs_CM=max(abs(r["CM"]) for r in points),
                status="evaluated",
                conditions=points,
            )
            record["feasible"] = (
                metrics["thickness_ratio"] >= min_t
                and record["max_abs_CM"] <= max_cm
            )
            if predicted:
                # The search measures its own surrogate: a candidate that
                # was kept still gets solved, so the prediction can be
                # compared against the truth it was standing in for.
                compared = []
                for entry, point in zip(predicted, points):
                    if not entry.get("usable"):
                        continue
                    compared.append(
                        {
                            "target_cl": entry["target_cl"],
                            "predicted_CD": entry["CD"],
                            "solved_CD": point["CD"],
                            "predicted_LD": entry["lift_to_drag"],
                            "solved_LD": (
                                point["CL"] / point["CD"]
                                if point["CD"] > 0
                                else None
                            ),
                        }
                    )
                record["surrogate_vs_solved"] = compared
                errors = [
                    abs(c["predicted_CD"] - c["solved_CD"]) for c in compared
                ]
                if errors:
                    record["surrogate_CD_error"] = sum(errors) / len(errors)
        except (ValueError, OSError, RuntimeError) as error:
            record.update(status="failed", error=str(error))
        saved["candidates"].append(record)
        atomic_json(file, saved)
    saved.update(
        status=(
            "ok"
            if any(r["feasible"] for r in saved["candidates"])
            else "no_feasible_candidate"
        ),
        pareto=pareto_front(saved["candidates"]),
    )
    for row in saved["candidates"]:
        row["pareto"] = row["candidate"] in saved["pareto"]
    atomic_json(file, saved)
    write_records(destination / "candidates.csv", saved["candidates"])
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for r in saved["candidates"]:
        if "mean_CD" not in r:
            continue
        ax.scatter(
            r["mean_CD"],
            r["worst_LD"],
            color=(
                "#d47927"
                if r["pareto"]
                else "#126782" if r["feasible"] else "#aaa"
            ),
        )
        ax.annotate(str(r["candidate"]), (r["mean_CD"], r["worst_LD"]))
    ax.set(
        xlabel="Mean CD across conditions (lower is better)",
        ylabel="Worst CL/CD across conditions (higher is better)",
        title="Orange: nondominated feasible candidates",
    )
    ax.grid(alpha=0.2)
    fig.savefig(destination / "pareto.png", dpi=150)
    plt.close(fig)
    report_page(
        destination / "report.html",
        "多工况翼型设计搜索",
        "在给定有限候选集合内比较平均阻力与最差升阻比，施加最小厚度和最大俯仰力矩约束。"
        "橙色为可行的 Pareto 候选。未收敛候选保留，不能据此声称全局最优。",
        saved["candidates"],
        [
            ("全部候选", "candidates.csv"),
            ("配置、失败原因和结果", "design_search.json"),
        ],
        [("Pareto 比较", "pareto.png")],
    )
    return file
