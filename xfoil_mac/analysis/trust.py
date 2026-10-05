"""Solver-domain evidence from boundary-layer dumps; no stall inference.

XFOIL reports ``status=ok`` and ``convergence_rate=1.0`` for sweeps whose
boundary layer has separated over a large part of the chord, which is
outside the "limited trailing edge separation" the method documents. It
also reports converged iterations at points whose surface iteration count
has reached the panel count, which is a structural cap rather than a
convergence signal. Convergence therefore cannot stand in for physical
validity, and this module never treats it as such.

Verdicts describe where the *solver* may be trusted. They are not stall
predictions. A passing verdict is not a validation, and a failing verdict
is not evidence that the real flow has stalled.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

# Column order of XFOIL's DUMP output, confirmed against the reader in
# boundary_layer.py: s, x, y, Ue/Vinf, Dstar, Theta, Cf, H. The header
# written into the file names further variables that the data rows do not
# contain, so the header text must never be used as a column map.
DUMP_COLUMNS = ("s", "x", "y", "Ue_Vinf", "Dstar", "Theta", "Cf", "H")

# Empirical screening thresholds. These are screening triggers, not
# physical boundaries and not error bounds. What the measured states look
# like, from a 16 condition matrix spanning 4 airfoils (0012, 2412, 4412,
# 0021), Re 3.7e5..1.7e6 and M 0.059..0.25, all at 0.5 deg spacing:
#
#   attached          one separated run at most, or none at all
#   leading-edge      one run near the leading edge, from about 8 deg
#   both ends         a leading-edge run and a trailing-edge run present
#                     together on the same surface
#   collapsed         one run covering most of the chord, far past any
#                     documented regime
#
# Only "both ends" and "collapsed" are used as verdicts. Separation
# confined to one end of the chord is reported as evidence but not as a
# failure, because a thin bubble and a limited trailing-edge separation
# are exactly the regimes the method documents support for.
#
# A peak shape-factor gate was tried and removed. It is not comparable
# across geometries: on NACA 0012 at Re=3.7e5 it flagged 4.5..5.0 deg
# while leaving 5.5..8.0 deg unflagged, and the flagged stations sat at
# x=0.99 where no domain edge exists. A domain edge cannot be
# non-monotonic in angle, so the gate was reporting trailing-edge
# thickness rather than the solver leaving its domain.
LONGEST_RUN_FLAG = 0.25
"""Longest continuous separated run, as a chord fraction, above which a
surface is reported as separated over a large part of the chord."""

EXTREME_SHAPE_FACTOR = 30.0
"""Peak kinematic shape factor recorded as a non-physical value.

This is a recording threshold, not a verdict. It is deliberately far
above the trailing-edge values that thin airfoils reach in ordinary
attached flow, because a shape-factor gate was measured to be
unreliable: on NACA 0012 at Re=3.7e5 it flagged alpha=4.5..5.0 deg while
leaving 5.5..8.0 deg unflagged, which no real domain edge can do. Above
roughly 30 the integral closure has broken down rather than reported a
thick boundary layer, and values near 90 were seen past alpha=17 deg."""

MIN_RUNS_FLAG = 2
"""Separated runs on one surface at which a leading-edge bubble and a
trailing-edge separation are present together. Either alone is a
documented regime; both at once on the same surface is the measured
marker of the sweep leaving the supported domain."""

ISOLATED_NEIGHBOUR_SPAN = 1.0
"""Angle span, in degrees, within which converged neighbours on both
sides make a failed point an isolated numerical failure rather than a
domain edge."""


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def alpha_from_dump_name(path: Path) -> float | None:
    """Recover the angle encoded in a ``bl_alpha_<label>`` file name."""
    match = re.search(r"bl_alpha_([-\dpm]+)\.txt$", path.name)
    if not match:
        return None
    return _finite(match.group(1).replace("p", ".").replace("m", "-"))


def read_dump(path: Path) -> list[dict]:
    """Read one DUMP file without geometry validation."""
    records = []
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if not fields:
            continue
        try:
            values = [
                float(v.replace("D", "E").replace("d", "e")) for v in fields
            ]
        except ValueError:
            continue
        if len(values) < len(DUMP_COLUMNS):
            continue
        if not all(math.isfinite(v) for v in values[: len(DUMP_COLUMNS)]):
            continue
        records.append(dict(zip(DUMP_COLUMNS, values)))
    return records


def pair_numbered_dumps(paths: list[Path], alphas: list[float]) -> dict:
    """Match ``bl_sequence_NNNN`` dumps to the angles they solved.

    A direct solver_commands run writes one dump per solved angle, in
    requested order, but an angle that fails to converge writes none. So
    pairing by count alone is wrong whenever a sweep is incomplete. The
    solved angles are taken from the requested ones minus the unsolved,
    which preserves order while tolerating gaps at the end.
    """
    solved = [round(float(a), 6) for a in alphas if _finite(a) is not None]
    if len(paths) == len(solved):
        return dict(zip(solved, paths))
    return {}


def split_surfaces(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split a DUMP into upper and lower surfaces.

    Stations run from the upper trailing edge to the leading edge and then
    to the lower trailing edge. The wake continues past x=1 and is
    excluded: it behaves differently from the airfoil surface and would
    otherwise be counted as separated length.
    """
    if len(records) < 3:
        return [], []
    leading = min(range(len(records)), key=lambda i: records[i]["x"])
    if leading in (0, len(records) - 1):
        return [], []
    upper = records[: leading + 1]
    lower = [
        record
        for record in records[leading + 1 :]
        if record["x"] <= 1.0 + 1e-9
    ]
    return upper, lower


def separated_runs(surface: list[dict]) -> list[dict]:
    """Contiguous runs of negative skin friction, in file order.

    Each run records its chordwise extent and station count. Runs are the
    meaningful unit: a leading-edge bubble and a trailing-edge separation
    are two runs that happen to sit on the same surface, and treating the
    span between them as separated length would overstate separation by a
    wide margin.
    """
    runs = []
    current = []
    for record in surface:
        if record["Cf"] < 0:
            current.append(record)
            continue
        if current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [
        {
            "stations": len(run),
            "x_start": min(r["x"] for r in run),
            "x_end": max(r["x"] for r in run),
        }
        for run in runs
    ]


def surface_evidence(surface: list[dict]) -> dict:
    """Separated runs, peak shape factor, and lowest skin friction."""
    if not surface:
        return {
            "stations": 0,
            "separated_fraction": None,
            "longest_run_fraction": None,
            "run_count": 0,
            "runs": [],
            "max_h": None,
            "max_h_x": None,
            "min_cf": None,
        }
    Cf = [r["Cf"] for r in surface]
    H = [r["H"] for r in surface]
    runs = separated_runs(surface)
    separated = sum(run["stations"] for run in runs)
    longest = max((run["stations"] for run in runs), default=0)
    peak = max(range(len(H)), key=lambda i: H[i])
    return {
        "stations": len(surface),
        "separated_fraction": separated / len(surface),
        "longest_run_fraction": longest / len(surface),
        "run_count": len(runs),
        "runs": runs,
        "max_h": H[peak],
        "max_h_x": surface[peak]["x"],
        "min_cf": min(Cf),
    }


def point_verdict(
    *,
    evidence: dict,
    converged: bool = True,
) -> tuple[str, list[str]]:
    """Return a verdict and the evidence codes that produced it."""
    if not converged:
        return "invalid", ["not_converged"]
    if not evidence or evidence.get("stations", 0) == 0:
        return "unchecked", ["no_boundary_layer_evidence"]
    reasons = []
    if (evidence.get("run_count") or 0) >= MIN_RUNS_FLAG:
        reasons.append("simultaneous_separated_regions")
    longest = evidence.get("longest_run_fraction")
    if longest is not None and longest > LONGEST_RUN_FLAG:
        reasons.append("large_separated_run")
    return ("suspect", reasons) if reasons else ("ok", [])


def worst_surface(*evidences: dict) -> dict:
    """Combine surfaces so the flagged one determines the verdict."""
    present = [e for e in evidences if e and e.get("stations")]
    if not present:
        return {}
    if len(present) == 1:
        return present[0]
    return max(
        present,
        key=lambda e: (
            e.get("run_count") or 0,
            e.get("longest_run_fraction") or 0.0,
            e.get("max_h") or 0.0,
        ),
    )


def sweep_report(
    case_dir: Path,
    alphas: list[float],
    failed_alphas: list[float] | None = None,
    iterations: dict[float, tuple] | None = None,
) -> dict:
    """Build the evidence table for one solved case directory.

    ``iterations`` optionally maps an angle to the printed per-surface
    iteration counts. They are carried through only so a caller can price
    a sweep from measured work rather than from an assumption; no verdict
    depends on them, because a count that reaches the panel cap is a
    structural limit rather than a convergence signal.
    """
    failed = {
        round(float(a), 6)
        for a in (failed_alphas or [])
        if _finite(a) is not None
    }
    dump_dir = case_dir / "boundary_layer"
    dumps = {}
    if dump_dir.is_dir():
        for path in sorted(dump_dir.glob("bl_alpha_*.txt")):
            alpha = alpha_from_dump_name(path)
            if alpha is not None:
                dumps[round(alpha, 6)] = path
    if not dumps and dump_dir.is_dir():
        numbered = sorted(dump_dir.glob("bl_sequence_*.txt"))
        if numbered:
            dumps = pair_numbered_dumps(numbered, alphas)

    # A failed solve leaves no row in the polar, so a requested angle that
    # never converged is absent from the solved list. Union the two so a
    # failure cannot be silently dropped from the evidence table.
    requested = {round(float(a), 6) for a in alphas if _finite(a) is not None}
    requested |= failed

    points = []
    for alpha in sorted(requested):
        key = round(float(alpha), 6)
        converged = key not in failed
        path = dumps.get(key)
        upper = lower = {}
        if path is not None:
            up_records, lo_records = split_surfaces(read_dump(path))
            upper = surface_evidence(up_records)
            lower = surface_evidence(lo_records)
        governing = worst_surface(upper, lower)
        verdict, reasons = point_verdict(
            evidence=governing, converged=converged
        )
        points.append(
            {
                "alpha": float(alpha),
                "verdict": verdict,
                "reasons": reasons,
                "converged": converged,
                "governing_surface": (
                    "upper"
                    if governing is upper and upper
                    else "lower" if governing is lower and lower else None
                ),
                "upper": upper,
                "lower": lower,
                "iterations": (iterations or {}).get(key),
            }
        )

    mark_isolated_failures(points)
    return {
        "case": str(case_dir),
        "points": points,
        "counts": _counts(points),
    }


def mark_isolated_failures(points: list[dict]) -> None:
    """Reclassify failures bracketed by converged neighbours.

    Isolated failures were measured inside healthy, fully attached
    regions, so a lone failure is not by itself evidence of a domain
    edge. Only a failure without converged neighbours on both sides is
    left reported as a boundary.
    """
    usable = sorted(
        point["alpha"]
        for point in points
        if point["converged"] and point["verdict"] != "invalid"
    )
    for point in points:
        if point["converged"]:
            continue
        left = [a for a in usable if a < point["alpha"]]
        right = [a for a in usable if a > point["alpha"]]
        if not left or not right:
            continue
        if (
            point["alpha"] - max(left) <= ISOLATED_NEIGHBOUR_SPAN
            and min(right) - point["alpha"] <= ISOLATED_NEIGHBOUR_SPAN
        ):
            point["verdict"] = "retry"
            point["reasons"] = ["isolated_failure_with_converged_neighbours"]


def verdicts_from_records(records_by_alpha: dict, failed_alphas=None) -> list:
    """Judge points whose boundary-layer records are already in memory.

    A caller that has read the dumps does not need to read them again
    from disk, and the shape of the directory is none of this module's
    business. ``records_by_alpha`` maps an angle to the list of station
    records read from its dump.
    """
    failed = {
        round(float(a), 6)
        for a in (failed_alphas or [])
        if _finite(a) is not None
    }
    angles = {round(float(a), 6) for a in records_by_alpha}
    angles |= failed
    points = []
    for alpha in sorted(angles):
        key = round(float(alpha), 6)
        records = records_by_alpha.get(key) or []
        converged = key not in failed
        upper = lower = {}
        if records:
            up_records, lo_records = split_surfaces(records)
            upper = surface_evidence(up_records)
            lower = surface_evidence(lo_records)
        governing = worst_surface(upper, lower)
        verdict, reasons = point_verdict(
            evidence=governing, converged=converged
        )
        points.append(
            {
                "alpha": float(alpha),
                "verdict": verdict,
                "reasons": reasons,
                "converged": converged,
                "governing_surface": (
                    "upper"
                    if governing is upper and upper
                    else "lower" if governing is lower and lower else None
                ),
                "upper": upper,
                "lower": lower,
            }
        )
    mark_isolated_failures(points)
    return points


def summarise_points(points: list) -> dict:
    """Counts per verdict, plus the last angle with trusted evidence."""
    counts = {}
    for point in points:
        name = point["verdict"]
        counts[name] = counts.get(name, 0) + 1
    trusted = [
        float(p["alpha"])
        for p in points
        if p["verdict"] in {"ok", "suspect"} and p.get("converged", True)
    ]
    suspect = [float(p["alpha"]) for p in points if p["verdict"] == "suspect"]
    if suspect:
        trusted = [a for a in trusted if a < min(suspect)]
    return {
        "counts": counts,
        "points": len(points),
        "last_trusted_alpha": max(trusted) if trusted else None,
        "first_suspect_alpha": min(suspect) if suspect else None,
        "by_alpha": {
            round(float(p["alpha"]), 6): p["verdict"] for p in points
        },
    }


def _counts(points: list[dict]) -> dict:
    counts = {}
    for point in points:
        counts[point["verdict"]] = counts.get(point["verdict"], 0) + 1
    return counts
