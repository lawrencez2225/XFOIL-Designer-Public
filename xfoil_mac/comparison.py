"""Compare saved runs without invoking XFOIL or joining across missing data."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path

from .solver_settings import SolverSettings
from .data import (
    atomic_json,
    atomic_text,
    convergence_summary,
    gapped_rows,
    parse_polar_text,
    performance_metrics,
)


def write_comparison(
    folders: list[Path], out_dir: Path, target_cl: float | None = None
) -> Path:
    import matplotlib.pyplot as plt

    if target_cl is not None and not math.isfinite(target_cl):
        raise ValueError("Target CL must be finite")
    manifests = set()
    failures = set()
    pending = list(folders)
    visited = set()
    while pending:
        folder = pending.pop()
        folder = folder.expanduser().resolve()
        if folder in visited:
            continue
        visited.add(folder)
        if not folder.is_dir():
            raise ValueError(f"Comparison directory not found: {folder}")
        if (folder / "failure.json").is_file():
            failures.add(folder / "failure.json")
        elif (folder / "run.json").is_file():
            manifests.add(folder / "run.json")
        elif (folder / "study.json").is_file():
            pending.extend(
                folder / p
                for p in json.loads((folder / "study.json").read_text())[
                    "folders"
                ]
            )
        elif (folder / "batch.json").is_file():
            batch = json.loads((folder / "batch.json").read_text())
            manifests.update(
                folder / case / "run.json"
                for case in batch["cases"]
                if (folder / case / "run.json").is_file()
            )
            failures.update(
                folder / f"{case}.failure.json"
                for case in batch["cases"]
                if (folder / f"{case}.failure.json").is_file()
            )
        else:
            manifests.update(
                p
                for p in folder.rglob("run.json")
                if "_archive" not in p.relative_to(folder).parts
                and "_cache" not in p.relative_to(folder).parts
            )
            failures.update(
                p
                for p in folder.rglob("failure.json")
                if "_archive" not in p.relative_to(folder).parts
            )
            failures.update(folder.glob("*.failure.json"))
    if folders and not manifests and not failures:
        raise ValueError(
            "No saved run.json found. Legacy results must be rerun in a "
            "new directory with this version."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    groups = defaultdict(list)
    for path in sorted(manifests):
        if (path.parent / "failure.json").exists() or (
            path.parent.parent / f"{path.parent.name}.failure.json"
        ).exists():
            continue
        saved = json.loads(path.read_text())
        config = saved["config"]
        polar = path.parent / config["polar_filename"]
        rows = parse_polar_text(polar.read_text()) if polar.is_file() else []
        requested = saved.get("requested_alphas", config["alphas"])
        summary = convergence_summary(rows, requested)
        metrics = performance_metrics(rows, requested, target_cl)
        record = {
            "airfoil": saved["airfoil"],
            "re": config["re"],
            "mach": config["mach"],
            "alpha_start": config["alpha_start"],
            "alpha_end": config["alpha_end"],
            "alpha_step": config["alpha_step"],
            **summary,
            **metrics,
            "execution_status": saved.get("execution_status", "unknown"),
            "run_status": saved.get("status", "unknown"),
            "flow_type": config.get("flow_type", 1),
            "reference_cl": config.get("reference_cl", 1.0),
            "run_directory": str(path.parent),
        }
        record["failed_alphas"] = json.dumps(record["failed_alphas"])
        records.append(record)
        settings_key = json.dumps(
            SolverSettings.from_config(config).to_dict(), sort_keys=True
        )
        record["solver_settings"] = settings_key
        groups[
            (
                config["re"],
                config["mach"],
                config.get("flow_type", 1),
                config.get("reference_cl", 1.0),
                settings_key,
            )
        ].append((record, gapped_rows(rows, requested)))
    for path in sorted(failures):
        failure = json.loads(path.read_text())
        records.append(
            {
                "airfoil": failure.get(
                    "airfoil", f"NACA{failure.get('naca', '?')}"
                ),
                "re": failure.get("re"),
                "mach": failure.get("mach"),
                "status": "failed",
                "run_status": "failed",
                "error": failure.get("error", ""),
                "failure_record": str(path),
            }
        )
    fields = list(
        dict.fromkeys(key for record in records for key in record)
    ) or [
        "airfoil",
        "re",
        "mach",
        "status",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    atomic_text(out_dir / "comparison.csv", buffer.getvalue())
    generated = []
    # Paginate large databases; every case appears, and legends stay readable.
    for group_number, (
        (reynolds, mach, flow_type, reference_cl, settings_key),
        cases,
    ) in enumerate(sorted(groups.items()), 1):
        for page_start in range(0, len(cases), 12):
            page = page_start // 12 + 1
            selected = cases[page_start : page_start + 12]
            for name, xkey, ykey, xlabel, ylabel in (
                ("cl_vs_alpha", "alpha", "CL", "alpha (deg)", "CL"),
                ("cd_vs_alpha", "alpha", "CD", "alpha (deg)", "CD"),
                ("cl_vs_cd", "CD", "CL", "CD", "CL"),
                (
                    "lift_to_drag_vs_alpha",
                    "alpha",
                    "LD",
                    "alpha (deg)",
                    "CL/CD",
                ),
            ):
                fig, ax = plt.subplots(
                    figsize=(11.5, 7), constrained_layout=True
                )
                for record, rows in selected:
                    xs = [r[xkey] for r in rows]
                    ys = (
                        [
                            r["CL"] / r["CD"] if r["CD"] > 0 else math.nan
                            for r in rows
                        ]
                        if ykey == "LD"
                        else [r[ykey] for r in rows]
                    )
                    label = (
                        f"{record['airfoil']} "
                        f"({record['points']}/{record['requested_points']})"
                    )
                    ax.plot(xs, ys, marker=".", linewidth=1.3, label=label)
                condition = f"Re={reynolds:g}, Mach={mach:g}" + (
                    f" at CL={reference_cl:g}, TYPE {flow_type}"
                    if flow_type != 1
                    else ""
                )
                condition += (
                    f" | Ncrit={json.loads(settings_key)['ncrit']:g}, "
                    f"panels={json.loads(settings_key)['panels']}"
                )
                ax.set(
                    xlabel=xlabel,
                    ylabel=ylabel,
                    title=(
                        f"{condition} | page {page} | legend: "
                        f"converged/requested"
                    ),
                )
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=8, loc="best")
                filename = (
                    f"condition_{group_number:03d}_page_{page:03d}_{name}.png"
                )
                fig.savefig(out_dir / filename, dpi=160)
                generated.append(filename)
                plt.close(fig)
    previous_manifest = out_dir / "comparison_manifest.json"
    if previous_manifest.is_file():
        for old in json.loads(previous_manifest.read_text()).get("plots", []):
            if (
                old not in generated
                and Path(old).name == old
                and old.startswith("condition_")
                and old.endswith(".png")
            ):
                (out_dir / old).unlink(missing_ok=True)
    atomic_json(previous_manifest, {"plots": generated, "runs": len(records)})
    atomic_text(
        out_dir / "README.txt",
        "Comparison groups use identical reference Re/Mach, flow "
        "type, reference CL, transition and panel settings. Each "
        "page includes up to 12 runs.\n"
        "Maximum CL/CD is the maximum over sampled converged points, "
        "not an optimized value.\n"
        "Missing requested angles break curves. Compare convergence "
        "rate and alpha range before choosing an airfoil.\n"
        "Target CL: exact or interpolation between adjacent "
        "converged requested points; no extrapolation.\n"
        "Multiple crossings are ambiguous and have no single "
        "reported CD.\n"
        "Legacy runs without run.json cannot be compared "
        "reproducibly; rerun them with this version.\n",
    )
    return out_dir / "comparison.csv"
