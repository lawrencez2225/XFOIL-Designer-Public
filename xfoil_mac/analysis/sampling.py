"""Price a requested angle sweep before it is run.

Solver calls are the scarce resource, and every call costs a different
amount of work: measured iteration counts on one NACA 2412 polar rise
from about 26 near zero lift to about 98 near the top of the sweep. A
uniform grid therefore spends most of its budget re-confirming the
nearly straight part of the curve.

This module proposes a two-region grid instead: coarse spacing through
the near-linear range, fine spacing from the onset of the nonlinear
range to the last angle the solver is still trusted at. Planning needs no
solver calls, and the measured evidence of an existing sweep can be
supplied so the estimate is priced from real work rather than assumed.
"""

from __future__ import annotations

import math

# Measured on NACA 2412 at Re=3.67e5, M=0.059, panels=240. These are the
# printed per-surface iteration counts near the ends of a sweep that ran
# from -4 to 12 degrees, used only to price the work when no measurement
# is supplied. The count is not a convergence signal.
ITERATIONS_AT_LOW_ALPHA = 26.0
ITERATIONS_AT_HIGH_ALPHA = 98.0

DEFAULT_ALPHA_MIN = -4.0
DEFAULT_ALPHA_MAX = 18.0
DEFAULT_SPLIT = 8.0
"""Angle at which the near-linear range ends and spacing tightens.

Measured drag rise on NACA 2412 starts between 7 and 8 degrees: the
slope of CD against alpha jumps by a factor of 3.4 across that interval,
and the transition point begins moving forward faster than at any lower
angle.
"""

DEFAULT_COARSE_STEP = 2.0
DEFAULT_FINE_STEP = 0.5

DEFAULT_DOMAIN_LIMIT = 11.0
"""Angle at or above which the solver stops being trusted by default.

The median of 16 measured conditions: 4 airfoils (0012, 2412, 4412,
0021) at Re 3.7e5..1.7e6 and M 0.059..0.25, swept at 0.5 deg spacing.
Per-condition limits ran from 9.0 to 13.0 degrees, so this is a central
screening value and not a boundary any single case is guaranteed to
share. It is a default only: whenever a sweep supplies its own measured
evidence, that evidence replaces this number entirely.
"""

DEFAULT_FINE_POINTS_MIN = 12
"""Below this many fine-region points the plan still reports a stopping
angle, so a caller can tell a genuinely small domain from a grid that
would be misleadingly dense."""

_EPSILON = 1e-9


def _round_alpha(value: float) -> float:
    return round(value + 0.0, 6)


def grid_angles(start: float, stop: float, step: float) -> list[float]:
    """Inclusive angle list, with the stop angle always present."""
    if step <= 0:
        raise ValueError("Sampling step must be positive")
    if stop < start - _EPSILON:
        return []
    count = int(math.floor((stop - start) / step + _EPSILON))
    values = [start + index * step for index in range(count + 1)]
    if values[-1] < stop - _EPSILON:
        values.append(stop)
    return [_round_alpha(value) for value in values]


def _iteration_curve(points: list[dict]) -> list[tuple[float, float]]:
    """Measured per-angle work, from the larger of the two surfaces."""
    measured = []
    for point in points:
        counts = point.get("iterations")
        if not counts:
            continue
        values = [v for v in counts if isinstance(v, (int, float))]
        values = [v for v in values if math.isfinite(v) and v > 0]
        if values:
            measured.append((float(point["alpha"]), max(values)))
    return sorted(measured)


def _interpolate_cost(
    alpha: float, measured: list[tuple[float, float]]
) -> float | None:
    """Interpolated measured work, or None outside the measured range."""
    if not measured:
        return None
    if alpha < measured[0][0] - _EPSILON:
        return None
    if alpha > measured[-1][0] + _EPSILON:
        return None
    for (a0, c0), (a1, c1) in zip(measured, measured[1:]):
        if a0 - _EPSILON <= alpha <= a1 + _EPSILON:
            if a1 - a0 <= _EPSILON:
                return max(c0, c1)
            fraction = (alpha - a0) / (a1 - a0)
            return c0 + fraction * (c1 - c0)
    return measured[-1][1]


def estimate_iterations(
    alpha: float,
    *,
    measured: list[tuple[float, float]] | None = None,
    low: float,
    high: float,
) -> float:
    """Per-point work, from measurement where it exists.

    Outside the measured range the work is extrapolated along the line
    through the measured end points when they exist, so a dense region
    close to the top of a sweep is not priced as if it were cheap.
    """
    measured = measured or []
    inside = _interpolate_cost(alpha, measured)
    if inside is not None:
        return inside
    if len(measured) >= 2:
        (a0, c0), (a1, c1) = measured[0], measured[-1]
        if a1 - a0 > _EPSILON:
            slope = (c1 - c0) / (a1 - a0)
            return max(c0, c0 + slope * (alpha - a0))
    span = high - low
    if span <= _EPSILON:
        return ITERATIONS_AT_LOW_ALPHA
    fraction = (alpha - low) / span
    return ITERATIONS_AT_LOW_ALPHA + fraction * (
        ITERATIONS_AT_HIGH_ALPHA - ITERATIONS_AT_LOW_ALPHA
    )


def domain_limit(points: list[dict]) -> dict:
    """Highest angle that measured evidence still supports.

    Returns the last angle whose verdict is not suspect, together with
    the first angle that was flagged and the first angle that failed to
    converge, so a caller can see which evidence stopped the sweep.
    """
    suspect = [
        float(p["alpha"]) for p in points if p.get("verdict") == "suspect"
    ]
    failed = [
        float(p["alpha"])
        for p in points
        if not p.get("converged", True)
        or p.get("verdict") in {"invalid", "retry"}
    ]
    trusted = [
        float(p["alpha"])
        for p in points
        if p.get("verdict") in {"ok", "suspect"} and p.get("converged", True)
    ]
    beyond = [a for a in trusted if suspect and a >= min(suspect)]
    clean = [a for a in trusted if a not in beyond]
    return {
        "last_trusted": max(clean) if clean else None,
        "first_suspect": min(suspect) if suspect else None,
        "first_failed": min(failed) if failed else None,
        "evidence": bool(points),
    }


DOMAIN_NEIGHBOUR_LOG_RE = 0.5
"""Half-width of the Reynolds neighbourhood, in natural log, when
estimating a stopping angle from measured conditions. Reynolds numbers
span orders of magnitude, so the neighbourhood is multiplicative."""

DOMAIN_NEIGHBOUR_MACH = 0.08
"""Half-width of the Mach neighbourhood when estimating a stopping
angle from measured conditions."""


def _finite(value) -> float | None:
    """Numeric value, or None when the input is missing or unusable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def domain_limits_from_measurements(rows: list[dict]) -> list[dict]:
    """Stopping angle per measured configuration.

    Reads the per-run boundary from an evidence table. When several
    airfoils share a condition the limits are kept separate, because the
    spread across geometries at one condition is the quantity that makes
    a single default unsafe.
    """
    grouped: dict = {}
    for row in rows:
        if "last_trusted_alpha" in row:
            # One row per run, as produced by the domain table.
            value = _finite(row.get("last_trusted_alpha"))
            if value is None:
                continue
            identity = row.get("airfoil") or row.get("run_id")
        else:
            # One row per angle, as produced by the trust layer. Only a
            # point that passed defines a boundary: a flagged point is
            # already past it. This matches the domain table, where the
            # boundary is the last trusted angle.
            value = _finite(row.get("alpha"))
            if value is None or row.get("verdict") != "ok":
                continue
            identity = row.get("airfoil") or row.get("run_id")
        key = (row.get("re"), row.get("mach"))
        entry = grouped.setdefault(
            key,
            {"re": key[0], "mach": key[1], "limits": [], "airfoils": set()},
        )
        entry["limits"].append(value)
        if identity:
            entry["airfoils"].add(str(identity))
    measured = []
    for entry in grouped.values():
        limits = sorted(entry["limits"])
        measured.append(
            {
                "re": entry["re"],
                "mach": entry["mach"],
                "airfoils": len(entry["airfoils"]) or len(limits),
                "min": limits[0],
                "median": limits[len(limits) // 2],
                "max": limits[-1],
            }
        )
    measured.sort(key=lambda e: ((e["re"] or 0), (e["mach"] or 0)))
    return measured


def conservative_domain_limit(
    measured: list[dict],
    *,
    re: float | None = None,
    mach: float | None = None,
    fallback: float = DEFAULT_DOMAIN_LIMIT,
) -> dict:
    """Smallest trusted angle among measured conditions near a target.

    The smallest is used rather than the median because the stopping
    angle varies with geometry at a fixed condition, and a plan for an
    unsolved airfoil has no way to know where in that spread it will
    land. Planning to the median would put half of new sweeps past a
    boundary that was already measured.
    """
    if not measured:
        return {
            "limit": float(fallback),
            "matched": 0,
            "basis": "no measured conditions; using the default",
            "spread": None,
        }
    candidates = []
    for entry in measured:
        if re is not None and entry["re"]:
            if abs(math.log(max(entry["re"], 1e-9) / max(re, 1e-9))) > (
                DOMAIN_NEIGHBOUR_LOG_RE
            ):
                continue
        if mach is not None and entry["mach"] is not None:
            if abs(entry["mach"] - mach) > DOMAIN_NEIGHBOUR_MACH:
                continue
        candidates.append(entry)
    if not candidates:
        return {
            "limit": float(fallback),
            "matched": 0,
            "basis": (
                "no measured condition is near the target; "
                "using the default"
            ),
            "spread": None,
        }
    lows = [entry["min"] for entry in candidates]
    highs = [entry["max"] for entry in candidates]
    limit = min(lows)
    spread = (min(lows), max(highs))
    if spread[0] != spread[1]:
        basis = (
            f"smallest of {len(candidates)} measured condition(s), "
            f"spanning {spread[0]:g}..{spread[1]:g} deg"
        )
    else:
        basis = f"measured at {len(candidates)} condition(s)"
    return {
        "limit": float(limit),
        "matched": len(candidates),
        "basis": basis,
        "spread": spread,
    }


def _resolve_limit(
    points: list[dict] | None, fallback: float
) -> tuple[float, dict]:
    """Decide the stopping angle from evidence, or from the default.

    Supplied evidence always wins, including when it supports nothing: a
    sweep whose every solved angle is flagged must not fall back to the
    default and plan straight into the region the evidence rejected. The
    default is used only when no measured evidence exists at all, which
    is the case before a first sweep has ever been run.
    """
    report = domain_limit(points or [])
    if not points:
        return float(fallback), report
    if report["last_trusted"] is not None:
        return float(report["last_trusted"]), report
    # Nothing measured is trusted. An explicit unbounded sentinel keeps
    # this distinct from "the range happens to be narrow", which a
    # near-zero number could not express.
    return float("-inf"), report


def plan_sweep(
    *,
    alpha_min: float = DEFAULT_ALPHA_MIN,
    alpha_max: float = DEFAULT_ALPHA_MAX,
    split: float = DEFAULT_SPLIT,
    coarse_step: float = DEFAULT_COARSE_STEP,
    fine_step: float = DEFAULT_FINE_STEP,
    domain_limit_angle: float = DEFAULT_DOMAIN_LIMIT,
    points: list[dict] | None = None,
    domain: list[dict] | None = None,
    re: float | None = None,
    mach: float | None = None,
) -> dict:
    """Propose an angle grid, its measured work, and its total price.

    ``points`` accepts the evidence table of an existing sweep, so the
    stopping angle and the per-point work both come from measurement
    rather than from the defaults. That evidence describes one airfoil.

    ``domain`` accepts per-condition boundaries from
    :func:`domain_limits_from_measurements`, which describe a solved set
    of airfoils. With a target ``re`` and ``mach``, the stopping angle
    becomes the smallest boundary measured near that condition, so a plan
    for an unsolved airfoil does not reach past an observed limit.
    Evidence for the specific sweep still outranks the population.
    """
    limit, evidence = _resolve_limit(points, domain_limit_angle)
    population = {"used": False}
    if not points and domain:
        estimate = conservative_domain_limit(
            domain, re=re, mach=mach, fallback=domain_limit_angle
        )
        population = {"used": True, **estimate}
        limit = estimate["limit"]
    measured = _iteration_curve(points or [])
    reachable = min(alpha_max, limit)
    if reachable < alpha_min - _EPSILON:
        return {
            "angles": [],
            "regions": [],
            "costs": [],
            "total_iterations": 0.0,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "stop_angle": None,
            "fine_points": 0,
            "measured": measured,
            "evidence": evidence,
            "population": population,
            "notes": [
                "no measured angle is trusted, so nothing is planned; "
                "the flagged angles are kept as labels"
            ],
        }
    boundary = min(split, reachable)
    angles = grid_angles(alpha_min, boundary, coarse_step)
    if boundary < reachable - _EPSILON:
        fine = grid_angles(boundary, reachable, fine_step)
        angles = angles + [a for a in fine if a not in set(angles)]
    angles = sorted(set(angles))
    costs = [
        estimate_iterations(
            alpha, measured=measured, low=alpha_min, high=alpha_max
        )
        for alpha in angles
    ]
    fine_start = min(split, reachable)
    fine_count = sum(1 for a in angles if a >= fine_start - _EPSILON)
    regions = _regions(alpha_min, boundary, reachable, coarse_step, fine_step)
    notes = []
    if fine_count < DEFAULT_FINE_POINTS_MIN:
        notes.append("few fine-region points; the trusted range is short")
    if evidence["first_suspect"] is not None:
        notes.append(
            "stopping at the last angle with trusted evidence; "
            f"first flagged angle was {evidence['first_suspect']:g}"
        )
    if evidence["first_failed"] is not None:
        notes.append(
            "a solve failed at "
            f"{evidence['first_failed']:g}; failures are kept as labels"
        )
    if not evidence["evidence"]:
        notes.append(
            "no measured evidence supplied; the stop angle is the default"
        )
    return {
        "angles": angles,
        "regions": regions,
        "costs": costs,
        "total_iterations": sum(costs),
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "stop_angle": reachable,
        "fine_points": fine_count,
        "measured": measured,
        "evidence": evidence,
        "population": population,
        "notes": notes,
    }


def _regions(
    alpha_min: float,
    boundary: float,
    reachable: float,
    coarse_step: float,
    fine_step: float,
) -> list[dict]:
    regions = []
    if boundary > alpha_min + _EPSILON:
        regions.append(
            {
                "name": "coarse",
                "alpha_min": alpha_min,
                "alpha_max": boundary,
                "step": coarse_step,
                "reason": "near-linear range",
            }
        )
    if reachable > boundary + _EPSILON:
        regions.append(
            {
                "name": "fine",
                "alpha_min": boundary,
                "alpha_max": reachable,
                "step": fine_step,
                "reason": "nonlinear range up to the trusted limit",
            }
        )
    return regions


def compare_to_uniform(
    plan: dict, step: float, fine_start: float | None = None
) -> dict:
    """Price the same trusted range on a uniform grid.

    Reporting the comparison keeps the claim honest: the saving is only
    meaningful against a stated alternative. ``fine_start`` only affects
    the reported count of points inside the nonlinear range.
    """
    measured = plan.get("measured") or []
    angles = grid_angles(plan["alpha_min"], plan["stop_angle"], step)
    costs = [
        estimate_iterations(
            alpha,
            measured=measured,
            low=plan["alpha_min"],
            high=plan["alpha_max"],
        )
        for alpha in angles
    ]
    if fine_start is None:
        fine_start = min(
            (
                region["alpha_min"]
                for region in plan["regions"]
                if region["name"] == "fine"
            ),
            default=plan["stop_angle"],
        )
    return {
        "step": step,
        "angles": angles,
        "total_iterations": sum(costs),
        "fine_points": sum(1 for a in angles if a >= fine_start - _EPSILON),
    }
