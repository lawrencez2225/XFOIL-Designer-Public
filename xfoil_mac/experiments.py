"""Compare matched reference data without filling missing solver points."""

from __future__ import annotations
import csv
import math
from pathlib import Path
from .data import atomic_json, sha256_file
from .flow import actual_conditions
from .results import discover_runs, interpolate, write_records


def load_experiment(source: Path):
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields):
            raise ValueError("Experimental CSV has duplicate column names")
        required = {"alpha", "CL", "CD", "Re", "Mach"}
        if not required.issubset(fields):
            raise ValueError(
                "Experimental CSV requires alpha,CL,CD,Re,Mach; "
                "CM and airfoil are optional"
            )
        rows = []
        for number, raw in enumerate(reader, 2):
            row = {
                "line": number,
                "airfoil": (raw.get("airfoil") or "").strip(),
            }
            for key in (*sorted(required), "CM"):
                if key == "CM" and not (raw.get(key) or "").strip():
                    continue
                try:
                    value = float(raw[key])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Invalid {key} at CSV line {number}"
                    ) from None
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite {key} at CSV line {number}")
                row[key] = value
            if row["Re"] <= 0 or not 0 <= row["Mach"] < 1:
                raise ValueError(
                    f"Invalid flow condition at CSV line {number}"
                )
            rows.append(row)
    if not rows:
        raise ValueError("Experimental CSV has no measurements")
    return rows


def compare_experiment(
    folder: Path,
    source: Path,
    destination: Path,
    re_tolerance=0.01,
    mach_tolerance=0.002,
) -> Path:
    import matplotlib.pyplot as plt

    if not all(
        math.isfinite(v) and v >= 0 for v in (re_tolerance, mach_tolerance)
    ):
        raise ValueError("Condition tolerances must be finite and nonnegative")
    runs = discover_runs([folder])
    measurements = load_experiment(source)
    records = []
    for measured in measurements:
        candidates = []
        reasons = set()
        for run in runs:
            if (
                measured["airfoil"]
                and measured["airfoil"].casefold() != run.airfoil.casefold()
            ):
                continue
            predicted, method = interpolate(run, "alpha", measured["alpha"])
            if predicted is None:
                reasons.add(method)
                continue
            condition = actual_conditions(run.config, predicted["CL"])
            if (
                condition is None
                or abs(condition[0] - measured["Re"])
                > measured["Re"] * re_tolerance + 1e-9
                or abs(condition[1] - measured["Mach"])
                > mach_tolerance + 1e-12
            ):
                reasons.add("condition_mismatch")
                continue
            candidates.append((run, predicted, method, condition))
        item = {f"measured_{k}": v for k, v in measured.items()}
        item["status"] = (
            "ambiguous_matching_runs"
            if len(candidates) > 1
            else "|".join(sorted(reasons)) or "no_matching_airfoil"
        )
        if len(candidates) == 1:
            run, predicted, method, condition = candidates[0]
            item.update(
                status="matched",
                method=method,
                airfoil=run.airfoil,
                run_directory=str(run.root),
                predicted_Re=condition[0],
                predicted_Mach=condition[1],
            )
            for key in ("CL", "CD", "CM"):
                if key in measured:
                    item[f"predicted_{key}"] = predicted[key]
                    item[f"error_{key}"] = predicted[key] - measured[key]
        records.append(item)
    metrics = {}
    for key in ("CL", "CD", "CM"):
        matched = [
            r
            for r in records
            if r["status"] == "matched" and f"error_{key}" in r
        ]
        if not matched:
            continue
        errors = [r[f"error_{key}"] for r in matched]
        metrics[key] = {
            "count": len(errors),
            "bias": sum(errors) / len(errors),
            "mae": sum(abs(e) for e in errors) / len(errors),
            "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "max_abs_error": max(abs(e) for e in errors),
        }
    destination = destination.expanduser().resolve()
    if source.expanduser().resolve() in [
        destination / "experiment_matches.csv",
        destination / "experiment_report.json",
    ]:
        raise ValueError("Output must not overwrite the experimental input")
    destination.mkdir(parents=True, exist_ok=True)
    from .plotting import (
        DARK_BG,
        DARK_GRID,
        DARK_MUTED,
        DARK_TEXT,
        apply_dark_engineering_style,
        style_dark_legend,
    )

    plots = []
    for key in metrics:
        matched = [
            r
            for r in records
            if r["status"] == "matched" and f"error_{key}" in r
        ]
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(9, 6),
            sharex=True,
            constrained_layout=True,
            facecolor=DARK_BG,
        )
        apply_dark_engineering_style(fig, axes)
        x = [r["measured_alpha"] for r in matched]
        axes[0].scatter(
            x,
            [r[f"measured_{key}"] for r in matched],
            marker="o",
            facecolors="none",
            edgecolors="#f59e0b",
            linewidths=1.6,
            label="Reference measurements",
        )
        axes[0].scatter(
            x,
            [r[f"predicted_{key}"] for r in matched],
            marker="x",
            color="#38bdf8",
            linewidths=1.8,
            label="XFOIL at matched conditions",
        )
        axes[0].set_ylabel(key, color=DARK_MUTED)
        axes[0].set_title(
            f'{key} comparison | RMSE={metrics[key]["rmse"]:.5g}',
            color="#38bdf8",
            fontweight="bold",
        )
        leg = axes[0].legend()
        style_dark_legend(leg)
        axes[1].scatter(
            x, [r[f"error_{key}"] for r in matched], color="#10b981", s=28
        )
        axes[1].axhline(0, color=DARK_MUTED, linestyle="--", linewidth=1.0)
        axes[1].set_xlabel("alpha (deg)", color=DARK_MUTED)
        axes[1].set_ylabel("Prediction - reference", color=DARK_MUTED)
        for ax in axes:
            ax.grid(True, color=DARK_GRID, alpha=0.85)
        name = f"experiment_{key}.png"
        fig.savefig(destination / name, dpi=160, facecolor=DARK_BG)
        plt.close(fig)
        plots.append(name)
    write_records(destination / "experiment_matches.csv", records)
    atomic_json(
        destination / "experiment_report.json",
        {
            "reference_file": str(source.resolve()),
            "reference_sha256": sha256_file(source),
            "measurements": len(records),
            "matched": sum(r["status"] == "matched" for r in records),
            "metrics": metrics,
            "plots": plots,
            "re_relative_tolerance": re_tolerance,
            "mach_absolute_tolerance": mach_tolerance,
            "note": (
                "Errors apply only to matched measurements; no extrapolation "
                "or interpolation across missing requested angles."
            ),
        },
    )
    return destination / "experiment_report.json"
