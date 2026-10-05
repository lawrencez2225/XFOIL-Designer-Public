"""Convergence/performance summaries and atomic CSV exports."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Sequence

from .data import (
    POLAR_FIELDS,
    aligned_rows,
    alpha_values,
    atomic_text,
    convergence_summary,
    performance_metrics,
)


def summarize_polar(
    rows: Sequence[dict[str, float]],
    requested_alphas: Sequence[float] | None = None,
    target_cl: float | None = None,
) -> dict:
    requested = (
        list(requested_alphas)
        if requested_alphas is not None
        else [r["alpha"] for r in rows]
    )
    rows = aligned_rows(rows, requested)
    summary = convergence_summary(rows, requested)
    summary.update(performance_metrics(rows, requested, target_cl))
    if not rows:
        return summary
    cl_max_row = max(rows, key=lambda row: row["CL"])
    cd_min_row = min(rows, key=lambda row: row["CD"])
    alpha0_row = min(rows, key=lambda row: abs(row["alpha"]))
    final_row = rows[-1]
    summary.update(
        {
            "cl_max": cl_max_row["CL"],
            "alpha_at_cl_max": cl_max_row["alpha"],
            "cd_min": cd_min_row["CD"],
            "alpha_at_cd_min": cd_min_row["alpha"],
            "cl_at_cd_min": cd_min_row["CL"],
            "cm_at_cd_min": cd_min_row["CM"],
            "cl_near_alpha0": alpha0_row["CL"],
            "cd_near_alpha0": alpha0_row["CD"],
            "cm_near_alpha0": alpha0_row["CM"],
            "final_alpha": final_row["alpha"],
        }
    )
    for key in POLAR_FIELDS[1:]:
        value = final_row.get(key, math.nan)
        summary["final_" + key.lower()] = (
            value if math.isfinite(value) else None
        )
    return summary


def write_polar_csv(rows: Sequence[dict[str, float]], csv_path: Path) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=POLAR_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(csv_path, buffer.getvalue())


def append_summary(
    summary_file: Path,
    airfoil_label: str,
    input_file: str,
    polar_file: Path,
    csv_file: Path | None,
    re_value: float,
    mach: float,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    iterations: int,
    rows: Sequence[dict[str, float]],
    target_cl: float | None = None,
    status: str | None = None,
    error: str = "",
    requested_alphas: Sequence[float] | None = None,
    flow_type: int = 1,
    reference_cl: float = 1.0,
) -> None:
    summary = summarize_polar(
        rows,
        (
            requested_alphas
            if requested_alphas is not None
            else alpha_values(alpha_start, alpha_end, alpha_step)
        ),
        target_cl,
    )
    record = {
        "airfoil": airfoil_label,
        "input_file": input_file,
        "re": re_value,
        "mach": mach,
        "alpha_start": alpha_start,
        "alpha_end": alpha_end,
        "alpha_step": alpha_step,
        "iter": iterations,
        **summary,
        "polar_file": str(polar_file),
        "csv_file": str(csv_file) if csv_file else "",
        "error": error,
        "flow_type": flow_type,
        "reference_cl": reference_cl,
    }
    record["failed_alphas"] = json.dumps(record["failed_alphas"])
    if status:
        record["status"] = status
    records = []
    if summary_file.is_file():
        with summary_file.open(newline="", encoding="utf-8") as handle:
            records = [
                r
                for r in csv.DictReader(handle)
                if r.get("polar_file") != str(polar_file)
            ]
    records.append(record)
    fields = list(dict.fromkeys(key for row in records for key in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    atomic_text(summary_file, buffer.getvalue())
