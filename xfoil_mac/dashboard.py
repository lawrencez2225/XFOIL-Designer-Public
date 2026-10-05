"""Portable offline analysis explorer, with data embedded and no remote
resources.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from .data import atomic_text, cp_name, read_pairs, row_for_alpha
from .flow import actual_conditions, condition_label
from .plotting import pressure_vector_components
from .results import discover_runs


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def write_dashboard(folders: list[Path], output: Path) -> Path:
    cases = []
    for run in discover_runs(folders):
        geometry = read_pairs(run.root / "geometry.dat")
        from .analysis.diagnostics import diagnose

        diagnostics = diagnose(run.root)["points"]
        samples = []
        for alpha in sorted(run.requested):
            row = row_for_alpha(run.rows, alpha)
            cp = read_pairs(run.root / cp_name(alpha)) if row else []
            condition = (
                actual_conditions(run.config, row["CL"]) if row else None
            )
            rotated, xs, ys, us, vs = (
                pressure_vector_components(cp, geometry, alpha)
                if cp and geometry
                else (geometry, [], [], [], [])
            )
            samples.append(
                {
                    "diagnostic": next(
                        (d for d in diagnostics if d["alpha"] == alpha), {}
                    ),
                    "alpha": alpha,
                    "row": row,
                    "cp": cp,
                    "geometry": rotated,
                    "vectors": list(zip(xs, ys, us, vs)),
                    "re": condition[0] if condition else None,
                    "mach": condition[1] if condition else None,
                }
            )
        cases.append(
            {
                "name": run.airfoil,
                "condition": condition_label(run.config),
                "status": run.manifest.get("status", "unknown"),
                "directory": str(run.root),
                "samples": samples,
            }
        )
    data = (
        json.dumps(_finite(cases), ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    template = (
        Path(__file__)
        .with_name("assets")
        .joinpath("explorer.html")
        .read_text()
    )
    atomic_text(output, template.replace("__XFOIL_DATA__", data))
    return output
