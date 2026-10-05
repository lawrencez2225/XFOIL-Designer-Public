"""One-way viscous correction to AVL using local XFOIL polars.

Polars are used without extrapolation.
"""

import json
import math
from pathlib import Path

import numpy as np

from ..avl import parse_strips
from ..data import atomic_json, atomic_text, fingerprint, sha256_file
from ..results import discover_runs, interpolate, write_records
from ..runtime import FlowAssumptions
from ..solver_settings import SolverSettings
from ..wing_geometry import section_profile
from .report import report_page


def local_profile(surface, y):
    """Interpolate normalized section coordinates.

    The spanwise position y is in model units.
    """
    sections = surface["sections"]
    if surface["mirror"]:
        y = abs(y)
    for a, b in zip(sections, sections[1:]):
        if a["y"] <= y <= b["y"]:
            t = (y - a["y"]) / (b["y"] - a["y"])
            return (1 - t) * section_profile(a) + t * section_profile(b)
    raise ValueError(
        "Strip lies outside the section span; cannot extrapolate its profile"
    )


def coupled_drag(
    app,
    source,
    destination,
    *,
    speed,
    length_unit_m=1.0,
    flow=None,
    solver_settings=None,
    alpha_range=(-4.0, 14.0, 0.5),
    timeout=120,
    resume=False,
    iterations=200,
    retries=2,
    boundary_layer=False,
):
    """Integrate attached-flow strip profile drag at speed in m/s.

    ``length_unit_m`` converts each model length unit to metres for local
    Reynolds numbers. Area ratios are dimensionless, so the same length
    conversion cancels between strip area and reference area. This one-way
    estimate leaves AVL's inviscid loads unchanged and does not predict stall.
    """
    from ..workflows import run_polar

    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    saved = json.loads((source / "wing_run.json").read_text())
    wing = saved["config"]
    if not (
        math.isfinite(speed)
        and speed > 0
        and math.isfinite(length_unit_m)
        and length_unit_m > 0
    ):
        raise ValueError("Speed and model length unit must be positive")
    assumptions = flow or FlowAssumptions.from_env()
    settings = solver_settings or SolverSettings()
    mach = speed / assumptions.sound_speed
    if abs(mach - wing["mach"]) > 0.00051:
        raise ValueError(
            f"Speed implies Mach {mach:.6g}; "
            "rerun AVL at this Mach to match conditions"
        )
    surfaces = []
    for surface in wing["surfaces"]:
        for a, b in zip(surface["sections"], surface["sections"][1:]):
            dy = b["y"] - a["y"]
            if (
                dy <= 0
                or abs(b["z"] - a["z"]) / dy > math.tan(math.radians(15))
                or abs(b["x"] - a["x"]) / dy > math.tan(math.radians(20))
            ):
                raise ValueError(
                    "Viscous correction currently requires monotonic span, "
                    "dihedral ≤15° and LE sweep ≤20°"
                )
        surfaces.append(surface)
        if surface["mirror"]:
            surfaces.append(surface)
    if not saved["cases"]:
        raise ValueError("Saved AVL study contains no cases")
    case_hashes = {}
    strip_tables = {}
    strip_errors = {}
    for case in saved["cases"]:
        if abs(case["target_beta"]) > 1e-9:
            raise ValueError(
                "Viscous correction currently requires zero sideslip"
            )
        path = source / f"case_{case['case']:03d}" / "strips.txt"
        if case["status"] == "ok":
            try:
                actual_hash = sha256_file(path)
                case_hashes[str(case["case"])] = actual_hash
                if actual_hash != case.get("raw_sha256", {}).get("strips.txt"):
                    raise ValueError(
                        "AVL strip data changed after the saved solve"
                    )
                if any(
                    not isinstance(case.get(key), (float, int))
                    or not math.isfinite(case[key])
                    for key in ("Alpha", "CLtot", "CDind")
                ):
                    raise ValueError(
                        "Saved AVL case lacks finite alpha, lift "
                        "or induced drag"
                    )
                strip_tables[case["case"]] = parse_strips(
                    path.read_text(), wing
                )
            except (OSError, ValueError) as exc:
                strip_errors[case["case"]] = str(exc)
    from ..workflows import calculation_source_hashes

    identity = {
        "calculation_sources": calculation_source_hashes(),
        "wing_fingerprint": saved["fingerprint"],
        "wing_run_sha256": sha256_file(source / "wing_run.json"),
        "strip_hashes": case_hashes,
        "speed": speed,
        "length_unit_m": length_unit_m,
        "flow": vars(assumptions),
        "settings": settings.to_dict(),
        "alpha_range": list(alpha_range),
        "timeout": timeout,
        "iterations": iterations,
        "retries": retries,
        "boundary_layer": boundary_layer,
        "implementation": sha256_file(Path(__file__)),
    }
    manifest = destination / "coupling.json"
    old = json.loads(manifest.read_text()) if manifest.exists() else None
    if old and (not resume or old["fingerprint"] != fingerprint(identity)):
        raise ValueError("Coupling settings changed; use a new folder")
    if not old and destination.exists() and any(destination.iterdir()):
        raise ValueError("Use an empty folder")
    destination.mkdir(parents=True, exist_ok=True)
    cache = {}
    details = []
    records = []
    payload = {
        "fingerprint": fingerprint(identity),
        "config": identity,
        "cases": records,
        "strips": details,
        "status": "running",
        "excluded_AVL_profile_drag": wing["profile_drag"],
    }
    atomic_json(manifest, payload)
    for case in saved["cases"]:
        if case["status"] != "ok" or case["case"] in strip_errors:
            records.append(
                {
                    "case": case["case"],
                    "status": (
                        "AVL_not_converged"
                        if case["status"] != "ok"
                        else "invalid_AVL_outputs"
                    ),
                    "avl_status": case["status"],
                    "reason": strip_errors.get(
                        case["case"], case.get("strip_error", case["status"])
                    ),
                    "CD_profile": None,
                    "CD_total": None,
                    "coverage": None,
                }
            )
            atomic_json(manifest, payload)
            continue
        strips = strip_tables[case["case"]]
        successful_area = 0.0
        total_area = 0.0
        profile_cd = 0.0
        successful_strips = 0
        for strip in strips:
            total_area += strip["Area"]
            surface_index = int(strip["surface"].split()[0]) - 1
            profile = local_profile(surfaces[surface_index], strip["Yle"])
            chord = strip["Chord"] * length_unit_m
            reynolds = (
                assumptions.air_density
                * speed
                * chord
                / assumptions.air_viscosity
            )
            key = fingerprint(
                {
                    "profile": np.round(profile, 10).tolist(),
                    "Re": reynolds,
                    "Mach": mach,
                }
            )[:16]
            if key not in cache:
                if len(cache) >= 100:
                    raise ValueError(
                        "Correction exceeds 100 local polars; "
                        "use a coarser span mesh"
                    )
                geometry = destination / "geometry" / f"section_{key}.dat"
                atomic_text(
                    geometry,
                    "Local interpolated profile\n"
                    + "".join(f"{x:.10g} {z:.10g}\n" for x, z in profile),
                )
                folder = destination / "data" / key
                run_polar(
                    app,
                    airfoil_file=geometry,
                    reynolds=reynolds,
                    mach=mach,
                    solver_settings=settings,
                    assumptions=FlowAssumptions(
                        chord_m=chord,
                        air_density=assumptions.air_density,
                        air_viscosity=assumptions.air_viscosity,
                        sound_speed=assumptions.sound_speed,
                    ),
                    alpha_start=alpha_range[0],
                    alpha_end=alpha_range[1],
                    alpha_step=alpha_range[2],
                    out_file=folder / "polar.txt",
                    resume=resume and (folder / "run.json").exists(),
                    timeout=timeout,
                    iterations=iterations,
                    retries=retries,
                    boundary_layer=boundary_layer,
                    show_xfoil_geometry=False,
                    save_pressure_vectors=False,
                    quiet=True,
                )
                cache[key] = discover_runs([folder])[0]
            row, method = interpolate(cache[key], "CL", strip["cl"])
            accepted = (
                row is not None and math.isfinite(row["CD"]) and row["CD"] > 0
            )
            if accepted:
                successful_strips += 1
                successful_area += strip["Area"]
                profile_cd += (
                    row["CD"] * strip["Area"] / wing["reference"]["area"]
                )
            details.append(
                dict(
                    case=case["case"],
                    surface=strip["surface"],
                    y=strip["Yle"],
                    chord_m=chord,
                    Re=reynolds,
                    local_CL=strip["cl"],
                    profile_CD=row["CD"] if accepted else None,
                    area=strip["Area"],
                    status=(
                        method if accepted else "no_unique_valid_polar_bracket"
                    ),
                    polar=f"data/{key}/report.html",
                )
            )
        # Every strip must have a valid polar bracket. An area tolerance must
        # not hide an unmatched but very narrow tip strip.
        complete = successful_strips == len(strips)
        records.append(
            dict(
                case=case["case"],
                alpha=case["Alpha"],
                CL=case["CLtot"],
                CD_induced=case["CDind"],
                CD_profile=profile_cd if complete else None,
                CD_total=case["CDind"] + profile_cd if complete else None,
                coverage=successful_area / total_area,
                status="ok" if complete else "incomplete_strip_coverage",
            )
        )
        atomic_json(manifest, payload)
    payload["status"] = (
        "ok"
        if all(r["status"] == "ok" for r in records)
        else "needs_attention"
    )
    atomic_json(manifest, payload)
    write_records(destination / "strip_profile_drag.csv", details)
    write_records(destination / "wing_drag.csv", records)
    report_page(
        destination / "report.html",
        "机翼黏性阻力估算",
        "逐条带按当地弦长求 Re，用当地翼型极曲线在 cl 上插值。"
        "总阻力 = AVL 诱导阻力 + XFOIL 型阻；原 AVL 的 CDvis/固定型阻不重复加入。"
        "缺失、跨断点或多解时不外推，总阻力留空。"
        "这是单向附着流修正，不会反过来修正载荷，也不是三维失速预测。",
        records,
        [
            ("条带匹配数据", "strip_profile_drag.csv"),
            ("设置、范围和原始索引", "coupling.json"),
        ],
    )
    return manifest
