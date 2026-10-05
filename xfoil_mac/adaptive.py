"""Bounded midpoint refinement; missing results never imply measured stall."""

from __future__ import annotations
import math
from .data import row_for_alpha


def validate_adaptive(
    rounds: int, max_points: int, min_step: float, initial_count: int
) -> None:
    if not isinstance(rounds, int) or not 0 <= rounds <= 8:
        raise ValueError("Adaptive rounds must be between 0 and 8")
    if not isinstance(max_points, int) or not 1 <= max_points <= 9999:
        raise ValueError("Adaptive max points must be between 1 and 9999")
    if rounds and initial_count > max_points:
        raise ValueError("Initial sweep already exceeds adaptive max points")
    if not math.isfinite(min_step) or min_step < 0.001:
        raise ValueError(
            "Adaptive minimum spacing must be at least 0.001 degrees"
        )


def refinement_candidates(
    rows, requested, max_points, min_step, target_cl=None
):
    """Choose extra angles in degrees without estimating missing results.

    Slope-change thresholds below are empirical sampling triggers, not
    error bounds or stall criteria. Absolute floors suppress refinement
    driven by printed-coefficient noise; the relative term requires a
    substantial change in the local curve slope.
    """
    ordered = sorted(set(requested))
    priorities = {}

    def add(i, priority):
        if 0 <= i < len(ordered) - 1:
            left, right = ordered[i : i + 2]
            midpoint = round((left + right) / 2, 3)
            if (
                min(midpoint - left, right - midpoint) >= min_step - 1e-9
                and midpoint not in ordered
            ):
                priorities[midpoint] = max(
                    priorities.get(midpoint, 0), priority
                )

    samples = [row_for_alpha(rows, a) for a in ordered]
    for key, priority in [("LD", 3), ("CL", 2)]:
        valid = [
            (
                i,
                (
                    r["CL"] / r["CD"]
                    if key == "LD" and r["CD"] > 0
                    else r["CL"] if key == "CL" else -math.inf
                ),
            )
            for i, r in enumerate(samples)
            if r
        ]
        if valid:
            i, value = max(valid, key=lambda pair: pair[1])
            if math.isfinite(value):
                add(i - 1, priority)
                add(i, priority)
    for i, (left, right) in enumerate(zip(samples, samples[1:])):
        if bool(left) != bool(right):
            add(i, 4)  # Diagnose boundary, not a claim of physical stall.
        if (
            left
            and right
            and target_cl is not None
            and (left["CL"] - target_cl) * (right["CL"] - target_cl) <= 0
        ):
            add(i, 5)
    for i in range(1, len(samples) - 1):
        a, b, c = samples[i - 1 : i + 2]
        if not all((a, b, c)):
            continue
        for key in ("CL", "CD"):
            s1 = (b[key] - a[key]) / (ordered[i] - ordered[i - 1])
            s2 = (c[key] - b[key]) / (ordered[i + 1] - ordered[i])
            # CL/degree or CD/degree; 0.3 is a dimensionless slope change.
            threshold = 0.03 if key == "CL" else 0.0005
            if abs(s2 - s1) > max(threshold, 0.3 * max(abs(s1), abs(s2))):
                add(i - 1, 2)
                add(i, 2)
    available = max(0, max_points - len(ordered))
    return sorted(
        sorted(priorities, key=lambda a: (-priorities[a], a))[:available]
    )
