"""Screen airfoils against constraints over a full condition matrix."""

from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path
from .adaptive import validate_adaptive
from .data import (
    alpha_values,
    atomic_json,
    convergence_summary,
    safe_label,
    validate_solver_options,
)
from .geometry import inspect_points
from .data import read_pairs
from .results import (
    discover_failures,
    discover_runs,
    interpolate,
    write_records,
)


def load_criteria(path: Path) -> dict:
    c = json.loads(path.read_text())
    if not isinstance(c, dict):
        raise ValueError("Selection criteria must be a JSON object")
    allowed = {
        "reynolds",
        "mach",
        "target_cls",
        "min_thickness",
        "max_thickness",
        "max_abs_cm",
        "min_convergence",
        "alpha_range",
        "iterations",
        "timeout",
        "retries",
        "adaptive_rounds",
    }
    if set(c) - allowed:
        raise ValueError(
            f"Unknown selection settings: {sorted(set(c)-allowed)}"
        )
    for key in ("reynolds", "mach", "target_cls"):
        if (
            not isinstance(c.get(key), list)
            or not c[key]
            or not all(
                isinstance(x, (int, float)) and math.isfinite(x)
                for x in c[key]
            )
        ):
            raise ValueError(
                f"{key} must be a nonempty list of finite numbers"
            )
        c[key] = list(dict.fromkeys(c[key]))
    if any(r <= 0 for r in c["reynolds"]) or any(
        not 0 <= m < 1 for m in c["mach"]
    ):
        raise ValueError("Selection needs positive Re and subsonic Mach")
    for key, default in [
        ("min_thickness", 0.0),
        ("max_thickness", 1.0),
        ("max_abs_cm", 1.0),
        ("min_convergence", 1.0),
    ]:
        c.setdefault(key, default)
        if (
            not isinstance(c[key], (int, float))
            or not math.isfinite(c[key])
            or c[key] < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative")
    if c["max_thickness"] < c["min_thickness"] or c["min_convergence"] > 1:
        raise ValueError("Invalid thickness bounds or convergence fraction")
    c.setdefault("alpha_range", [-4, 12, 1])
    c.setdefault("iterations", 200)
    c.setdefault("timeout", 120.0)
    c.setdefault("retries", 2)
    c.setdefault("adaptive_rounds", 0)
    if (
        not isinstance(c["alpha_range"], list)
        or len(c["alpha_range"]) != 3
        or not all(
            isinstance(v, (int, float)) and math.isfinite(v)
            for v in c["alpha_range"]
        )
    ):
        raise ValueError(
            "alpha_range must contain finite start, end and step values"
        )
    requested = alpha_values(*c["alpha_range"])
    if isinstance(c["timeout"], bool) or not isinstance(
        c["timeout"], (int, float)
    ):
        raise ValueError("timeout must be a positive number")
    validate_solver_options(c["iterations"], c["timeout"], c["retries"], 0.5)
    validate_adaptive(c["adaptive_rounds"], 100, 0.125, len(requested))
    return c


def rank_runs(folders: list[Path], criteria: dict, destination: Path) -> Path:
    runs = discover_runs(folders, allow_empty=True)
    groups = defaultdict(list)
    for run in runs:
        groups[run.identity].append(run)
    ranking, details = [], []
    for identity, cases in sorted(groups.items()):
        label = cases[0].airfoil
        reasons = []
        scores = []
        coefficients = []
        solver_versions = {
            (
                r.config.get("binary_sha256"),
                r.config.get("version"),
                r.config.get("iterations"),
                json.dumps(
                    r.config.get("solver_settings", {}), sort_keys=True
                ),
            )
            for r in cases
        }
        if len(solver_versions) > 1:
            reasons.append(
                "Mixed solver versions, transition/panel controls or "
                "iteration settings across conditions; use a consistent "
                "study"
            )
        try:
            geometry, _ = inspect_points(
                read_pairs(cases[0].root / "geometry.dat")
            )
            thickness = (
                geometry.get("thickness_ratio") if geometry["valid"] else None
            )
            transform = geometry["transform"]
            if (
                abs(transform["chord"] - 1) > 0.001
                or abs(transform["rotation_deg"]) > 1
                or math.hypot(*transform["leading_edge"]) > 0.01
            ):
                reasons.append(
                    "Geometry needs unit-chord normalization before "
                    "comparing coefficients; inspect and recalculate"
                )
        except ValueError:
            thickness = None
        if thickness is None:
            reasons.append("No valid geometry for thickness check")
        elif (
            not criteria["min_thickness"]
            <= thickness
            <= criteria["max_thickness"]
        ):
            reasons.append("Thickness outside required range")
        for re_value in criteria["reynolds"]:
            for mach in criteria["mach"]:
                matched = [
                    r
                    for r in cases
                    if r.config.get("flow_type", 1) == 1
                    and math.isclose(r.config["re"], re_value, rel_tol=1e-9)
                    and math.isclose(r.config["mach"], mach, abs_tol=1e-9)
                ]
                for target in criteria["target_cls"]:
                    item = {
                        "airfoil": label,
                        "identity": identity,
                        "Re": re_value,
                        "Mach": mach,
                        "target_cl": target,
                    }
                    if len(matched) != 1:
                        reason = (
                            "missing_condition"
                            if not matched
                            else "duplicate_condition"
                        )
                        item["status"] = reason
                        reasons.append(
                            f"{reason}: Re={re_value:g}, M={mach:g}"
                        )
                    else:
                        run = matched[0]
                        point, method = interpolate(run, "CL", target)
                        coverage = convergence_summary(
                            run.rows, run.requested
                        )["convergence_rate"]
                        item.update(
                            status=method,
                            convergence_rate=coverage,
                            run_directory=str(run.root),
                        )
                        if coverage < criteria["min_convergence"]:
                            reasons.append(
                                f"Convergence below requirement: "
                                f"Re={re_value:g}, M={mach:g}"
                            )
                        if run.manifest.get("status") not in (
                            "ok",
                            "partial_convergence",
                        ):
                            reasons.append(
                                f"Incomplete or failed run: "
                                f"{run.manifest.get('status')}"
                            )
                        if point is None:
                            reasons.append(
                                f"{method}: CL={target:g}, "
                                f"Re={re_value:g}, M={mach:g}"
                            )
                        else:
                            item.update(
                                alpha=point["alpha"],
                                CD=point["CD"],
                                CM=point["CM"],
                            )
                            scores.append(point["CD"])
                            coefficients.append(abs(point["CM"]))
                            if abs(point["CM"]) > criteria["max_abs_cm"]:
                                reasons.append(
                                    f"Pitching moment exceeds limit at "
                                    f"CL={target:g}, Re={re_value:g}, "
                                    f"M={mach:g}"
                                )
                    details.append(item)
        ranking.append(
            {
                "airfoil": label,
                "identity": identity,
                "eligible": not reasons,
                "thickness_ratio": thickness,
                "worst_cd": max(scores) if scores else None,
                "mean_cd": sum(scores) / len(scores) if scores else None,
                "worst_abs_cm": max(coefficients) if coefficients else None,
                "reasons": list(dict.fromkeys(reasons)),
                "rank": None,
                "pareto": False,
            }
        )
    for failure_path, failure in discover_failures(folders):
        label = failure.get("airfoil") or (
            f"NACA{failure['naca']}"
            if failure.get("naca")
            else failure_path.stem
        )
        reason = "Failed calculation: " + failure.get(
            "error", "No usable result"
        )
        if not any(row["airfoil"] == label for row in ranking):
            ranking.append(
                {
                    "airfoil": label,
                    "identity": "failure:" + str(failure_path),
                    "eligible": False,
                    "thickness_ratio": None,
                    "worst_cd": None,
                    "mean_cd": None,
                    "worst_abs_cm": None,
                    "reasons": [reason],
                    "rank": None,
                    "pareto": False,
                }
            )
    valid = sorted(
        (r for r in ranking if r["eligible"]),
        key=lambda r: (r["worst_cd"], r["mean_cd"], r["airfoil"]),
    )
    for index, row in enumerate(valid, 1):
        row["rank"] = index
        row["pareto"] = not any(
            other["worst_cd"] <= row["worst_cd"]
            and other["worst_abs_cm"] <= row["worst_abs_cm"]
            and (
                other["worst_cd"] < row["worst_cd"]
                or other["worst_abs_cm"] < row["worst_abs_cm"]
            )
            for other in valid
        )
    ranking.sort(
        key=lambda r: (not r["eligible"], r["rank"] or 0, r["airfoil"])
    )
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(
        destination / "selection.json",
        {
            "criteria": criteria,
            "ranking_policy": (
                "eligible first; minimum worst-case CD, then mean CD; Pareto "
                "uses worst CD and absolute CM"
            ),
            "eligible": len(valid),
            "candidates": ranking,
        },
    )
    write_records(
        destination / "selection.csv",
        [dict(r, reasons="; ".join(r["reasons"])) for r in ranking],
    )
    write_records(destination / "selection_details.csv", details)
    return destination / "selection.json"


def select_source(
    app,
    source: Path,
    criteria: dict,
    destination: Path,
    resume=False,
    assumptions=None,
) -> Path:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Selection source not found: {source}")
    files = [
        p
        for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in (".dat", ".txt")
    ]
    if (
        any(
            (source / name).is_file()
            for name in ("run.json", "batch.json", "study.json")
        )
        or not files
        and any(source.rglob("run.json"))
    ):
        return rank_runs([source], criteria, destination)
    if not files:
        raise ValueError(
            "Selection source has neither saved runs nor coordinate files"
        )
    from .workflows import run_batch

    folders = []
    for re_value in criteria["reynolds"]:
        for mach in criteria["mach"]:
            out = (
                destination
                / "calculations"
                / f"Re{re_value:.10g}_M{mach:.10g}"
            )
            run_batch(
                app,
                source,
                re_value,
                mach,
                criteria["iterations"],
                *criteria["alpha_range"],
                out,
                False,
                resume=resume,
                timeout=criteria["timeout"],
                retries=criteria["retries"],
                headless=True,
                adaptive_rounds=criteria["adaptive_rounds"],
                assumptions=assumptions,
            )
            folders.append(out)
    result = rank_runs(folders, criteria, destination)
    # Invalid files never create a run manifest; keep them visible in the final
    # selection.
    report = json.loads(result.read_text())
    represented = {r["identity"] for r in report["candidates"]}
    represented_labels = {r["airfoil"] for r in report["candidates"]}
    from .data import sha256_file

    for path in files:
        identity = sha256_file(path)
        if (
            identity not in represented
            and safe_label(path.stem) not in represented_labels
        ):
            report["candidates"].append(
                {
                    "airfoil": path.stem,
                    "identity": identity,
                    "eligible": False,
                    "rank": None,
                    "pareto": False,
                    "reasons": [
                        "No usable calculation; inspect "
                        "calculations/*/*.failure.json"
                    ],
                }
            )
    atomic_json(result, report)
    write_records(
        destination / "selection.csv",
        [
            dict(r, reasons="; ".join(r["reasons"]))
            for r in report["candidates"]
        ],
    )
    return result
