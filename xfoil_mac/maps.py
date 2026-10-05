"""Observed Re/alpha maps mark missing and unrequested cells explicitly."""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from .data import atomic_json, row_for_alpha, safe_label
from .flow import actual_conditions
from .results import discover_runs, variant_key, write_records


def write_maps(folders: list[Path], destination: Path) -> Path:
    import numpy as np
    import matplotlib.pyplot as plt

    runs = discover_runs(folders)
    groups = defaultdict(list)
    for run in runs:
        groups[(run.identity, run.config["mach"], variant_key(run))].append(
            run
        )
    destination.mkdir(parents=True, exist_ok=True)
    plots, records = [], []
    for group_number, ((_, mach, variant), cases) in enumerate(
        sorted(groups.items()), 1
    ):
        reynolds = sorted({r.config["re"] for r in cases})
        alphas = sorted({a for r in cases for a in r.requested})
        cells = {}
        for run in cases:
            for alpha in run.requested:
                key = (run.config["re"], alpha)
                if key in cells:
                    raise ValueError(
                        f"Duplicate observations for Re={key[0]}, "
                        f"alpha={alpha}; select one run per condition"
                    )
                row = row_for_alpha(run.rows, alpha)
                condition = (
                    actual_conditions(run.config, row["CL"]) if row else None
                )
                cells[key] = row
                records.append(
                    {
                        "group": group_number,
                        "airfoil": run.airfoil,
                        "re_reference": key[0],
                        "mach_reference": mach,
                        "flow_type": run.config.get("flow_type", 1),
                        "reference_cl": run.config.get("reference_cl", 1),
                        "alpha": alpha,
                        "status": "converged" if row else "missing",
                        "Re": condition[0] if condition else None,
                        "Mach": condition[1] if condition else None,
                        **{
                            k: row[k] if row else None
                            for k in ("CL", "CD", "CM")
                        },
                        "LD": (
                            row["CL"] / row["CD"]
                            if row and row["CD"] > 0
                            else None
                        ),
                        "run_directory": str(run.root),
                    }
                )
        for metric in ("CL", "CD", "CM", "LD"):
            grid = np.full((len(reynolds), len(alphas)), np.nan)
            for iy, reynolds_value in enumerate(reynolds):
                for ix, alpha in enumerate(alphas):
                    row = cells.get((reynolds_value, alpha))
                    if row:
                        grid[iy, ix] = (
                            row["CL"] / row["CD"]
                            if metric == "LD" and row["CD"] > 0
                            else row[metric] if metric != "LD" else np.nan
                        )
            fig, ax = plt.subplots(
                figsize=(max(8, min(15, len(alphas) * 0.3)), 5),
                constrained_layout=True,
            )
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("#e5e7eb")
            picture = ax.imshow(
                np.ma.masked_invalid(grid),
                origin="lower",
                aspect="auto",
                cmap=cmap,
                interpolation="nearest",
            )
            selected = list(range(0, len(alphas), max(1, len(alphas) // 16)))
            ax.set_xticks(selected, [f"{alphas[i]:g}" for i in selected])
            ax.set_yticks(range(len(reynolds)), [f"{r:g}" for r in reynolds])
            for iy, re_value in enumerate(reynolds):
                for ix, alpha in enumerate(alphas):
                    if (re_value, alpha) in cells and cells[
                        (re_value, alpha)
                    ] is None:
                        ax.plot(ix, iy, "x", color="#c2410c", markersize=5)
            kind = cases[0].config.get("flow_type", 1)
            ax.set(
                xlabel="alpha (deg; sampled positions)",
                ylabel=(
                    "Re at reference CL"
                    if kind != 1
                    else "Re (sampled conditions)"
                ),
                title=(
                    f"{cases[0].airfoil} | {metric} | M reference={mach:g} | "
                    f"TYPE {kind}\nGray: no value; x: requested but missing; "
                    f"no interpolation"
                ),
            )
            fig.colorbar(picture, ax=ax, label=metric)
            name = (
                f"map_{group_number:03d}_"
                f"{safe_label(cases[0].airfoil)}_{metric}.png"
            )
            fig.savefig(destination / name, dpi=160)
            plt.close(fig)
            plots.append(name)
    write_records(destination / "performance_map.csv", records)
    atomic_json(
        destination / "maps.json",
        {
            "plots": plots,
            "groups": len(groups),
            "cells": len(records),
            "interpolated": False,
        },
    )
    return destination / "maps.json"
