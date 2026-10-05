"""Validate inputs and store reproducible convergence results."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime
from decimal import Decimal
import tempfile
from pathlib import Path
from typing import Sequence

APP_VERSION = "3.0.2"
ALPHA_TOLERANCE = 0.00051  # XFOIL polar files print alpha to three decimals.
POLAR_FIELDS = (
    "alpha",
    "CL",
    "CD",
    "CDp",
    "CM",
    "Top_Xtr",
    "Bot_Xtr",
    "Top_Itr",
    "Bot_Itr",
)


def positive_finite(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def validated_alphas(start: float, end: float, step: float) -> list[float]:
    if not all(math.isfinite(v) for v in (start, end, step)):
        raise ValueError("Alpha start, end, and step must be finite")
    if max(abs(start), abs(end)) > 180:
        raise ValueError("Alpha must be between -180 and 180 degrees")
    if step == 0 or (end - start) * step < 0:
        raise ValueError(
            "Alpha step must be nonzero and point toward the end angle"
        )
    if start != end and abs(step) < 0.001:
        raise ValueError(
            "Alpha step must be at least 0.001 degrees (polar file precision)"
        )
    count = math.floor((end - start) / step + 1e-8) + 1
    if not 1 <= count <= 9999:
        raise ValueError(
            "An alpha sweep must contain between 1 and 9999 points"
        )
    values = [round(float(start) + i * float(step), 10) for i in range(count)]
    if any(
        not math.isclose(v, round(v, 3), abs_tol=1e-8, rel_tol=0)
        for v in values
    ):
        raise ValueError(
            "Requested angles must align to 0.001 degrees "
            "(polar file precision)"
        )
    if len({f"{v:.3f}" for v in values}) != len(values):
        raise ValueError(
            "Requested angles overlap at XFOIL's 0.001 degree precision"
        )
    return values


def validate_solver_options(
    iterations: int, timeout: float, retries: int, retry_step: float
) -> None:
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or not 1 <= iterations <= 10000
    ):
        raise ValueError("Iterations must be an integer between 1 and 10000")
    positive_finite(timeout, "Timeout")
    if (
        isinstance(retries, bool)
        or not isinstance(retries, int)
        or not 0 <= retries <= 5
    ):
        raise ValueError("Retries must be an integer between 0 and 5")
    if not math.isfinite(retry_step) or not 0.01 <= retry_step <= 1:
        raise ValueError("Retry step must be between 0.01 and 1 degree")


def cl_values(start: float, end: float, step: float) -> list[float]:
    """Generate up to 200 dimensionless CL targets, including aligned ends.

    CL targets do not inherit the polar file's 0.001-degree angle grid.
    Decimal arithmetic keeps a user-entered endpoint on its intended grid;
    the direct solver still checks the achieved CL against its tolerance.
    """
    if not all(math.isfinite(value) for value in (start, end, step)):
        raise ValueError("CL start, end, and step must be finite")
    if max(abs(start), abs(end)) > 5:
        raise ValueError("Target CL must be between -5 and 5")
    if step == 0 or (end > start and step < 0) or (end < start and step > 0):
        raise ValueError("CL step must be nonzero and point toward the end")
    first, last, increment = (Decimal(str(v)) for v in (start, end, step))
    intervals = (last - first) / increment
    if intervals >= 200:
        raise ValueError("A CL sequence must contain between 1 and 200 points")
    count = int(intervals) + 1
    values = [float(first + i * increment) for i in range(count)]
    if len(set(values)) != len(values):
        raise ValueError("CL step is too small to distinguish target values")
    return values


def parse_polar_text(text: str) -> list[dict[str, float]]:
    rows = []
    in_data = False
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[:2] == ["alpha", "CL"]:
            in_data = True
            continue
        if not in_data or len(parts) < 7:
            continue
        try:
            values = [float(part.replace("D", "E")) for part in parts]
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in values[:7]) or values[2] < 0:
            continue
        values += [math.nan] * max(0, 9 - len(values))
        # Some bundled versions interleave each surface's transition and index.
        if values[6] > 1 and 0 <= values[7] <= 1:
            values[6], values[7] = values[7], values[6]
        rows.append(dict(zip(POLAR_FIELDS, values)))
    return rows


def row_for_alpha(
    rows: Sequence[dict[str, float]], alpha: float
) -> dict[str, float] | None:
    if not rows:
        return None
    row = min(rows, key=lambda r: abs(r["alpha"] - alpha))
    return row if abs(row["alpha"] - alpha) <= ALPHA_TOLERANCE else None


def aligned_rows(
    rows: Sequence[dict[str, float]], requested: Sequence[float]
) -> list[dict[str, float]]:
    return [
        dict(row, alpha=alpha)
        for alpha in requested
        if (row := row_for_alpha(rows, alpha)) is not None
    ]


def convergence_summary(
    rows: Sequence[dict[str, float]], requested: Sequence[float]
) -> dict:
    missing = [a for a in requested if row_for_alpha(rows, a) is None]
    count = len(requested) - len(missing)
    return {
        "points": count,
        "requested_points": len(requested),
        "convergence_rate": count / len(requested) if requested else 0,
        "failed_alphas": missing,
        "status": (
            "ok"
            if requested and not missing
            else "partial_convergence" if count else "no_converged_points"
        ),
    }


def gapped_rows(
    rows: Sequence[dict[str, float]], requested: Sequence[float]
) -> list[dict[str, float]]:
    """NaNs break plotted curves across every missing requested point."""
    result = []
    for alpha in sorted(requested):
        row = row_for_alpha(rows, alpha)
        result.append(
            dict(row, alpha=alpha)
            if row
            else {
                key: alpha if key == "alpha" else math.nan
                for key in POLAR_FIELDS
            }
        )
    return result


def performance_metrics(
    rows: Sequence[dict[str, float]],
    requested: Sequence[float],
    target_cl: float | None = None,
) -> dict:
    ratios = [
        (r["CL"] / r["CD"], r)
        for r in rows
        if r.get("CD", 0) > 0 and math.isfinite(r["CL"] / r["CD"])
    ]
    best = max(ratios, key=lambda item: item[0]) if ratios else None
    result = {
        "max_lift_to_drag": best[0] if best else None,
        "alpha_at_max_lift_to_drag": best[1]["alpha"] if best else None,
        "target_cl": target_cl,
        "cd_at_target_cl": None,
        "alpha_at_target_cl": None,
        "target_cl_status": (
            "not_requested" if target_cl is None else "not_bracketed"
        ),
    }
    if target_cl is None:
        return result
    if not math.isfinite(target_cl):
        raise ValueError("Target CL must be finite")
    candidates = []
    ordered = gapped_rows(rows, requested)
    for row in ordered:
        if math.isfinite(row["CL"]) and math.isclose(
            row["CL"], target_cl, abs_tol=1e-9
        ):
            candidates.append((row["alpha"], row["CD"], "exact"))
    for left, right in zip(ordered, ordered[1:]):
        if not all(
            math.isfinite(r[k]) for r in (left, right) for k in ("CL", "CD")
        ):
            continue
        if (left["CL"] - target_cl) * (right["CL"] - target_cl) < 0:
            fraction = (target_cl - left["CL"]) / (right["CL"] - left["CL"])
            candidates.append(
                (
                    left["alpha"]
                    + fraction * (right["alpha"] - left["alpha"]),
                    left["CD"] + fraction * (right["CD"] - left["CD"]),
                    "interpolated",
                )
            )
    # Multiple crossings can occur near stall. Do not silently choose a branch.
    unique = {round(a, 8): (a, cd, method) for a, cd, method in candidates}
    if len(unique) == 1:
        alpha, cd, method = next(iter(unique.values()))
        result.update(
            cd_at_target_cl=cd,
            alpha_at_target_cl=alpha,
            target_cl_status=method,
        )
    elif unique:
        result["target_cl_status"] = "ambiguous_multiple_crossings"
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
    )


def write_combined_polar(
    path: Path, rows: Sequence[dict[str, float]], metadata: dict
) -> None:
    lines = [
        "XFOIL-MAC validated, merged polar; "
        "raw solver outputs are in attempts/",
        f"Re = {metadata['re']:.10g}   Mach = {metadata['mach']:.10g}   "
        f"TYPE = {metadata.get('flow_type', 1)}   "
        f"reference_CL = {metadata.get('reference_cl', 1):.10g}",
        " ".join(POLAR_FIELDS),
        " ".join("----------" for _ in POLAR_FIELDS),
    ]
    lines.extend(
        " ".join(f"{row.get(key, math.nan):.10g}" for key in POLAR_FIELDS)
        for row in rows
    )
    atomic_text(path, "\n".join(lines) + "\n")


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "airfoil"


def timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def alpha_values(
    alpha_start: float, alpha_end: float, alpha_step: float
) -> list[float]:
    return validated_alphas(alpha_start, alpha_end, alpha_step)


def alpha_file_label(alpha: float) -> str:
    return safe_label(f"{alpha:g}".replace("-", "m").replace(".", "p"))


def cp_name(alpha: float) -> str:
    label = f"{alpha:.10g}".replace("-", "m").replace(".", "p")
    return f"cp_alpha_{label}.txt"


def read_pairs(path: Path) -> list[tuple[float, float]]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        try:
            pair = float(fields[0]), float(fields[1])
        except (ValueError, IndexError):
            continue
        if all(math.isfinite(value) for value in pair):
            result.append(pair)
    return result


def cp_filename_for_alpha(alpha: float) -> str:
    return cp_name(alpha)


def parse_numeric_pairs(path: Path) -> list[tuple[float, float]]:
    return read_pairs(path)


def parse_polar_file(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    return parse_polar_text(path.read_text(errors="ignore"))


def parse_cp_file(path: Path) -> list[tuple[float, float]]:
    return parse_numeric_pairs(path)
