"""One flat table of solver results with trust labels attached.

Every row is one solved operating point: what was asked of the solver,
what it returned, and whether the returned numbers sit inside the domain
the method documents support for. Rows are never silently dropped. A
point that failed to converge keeps its place with a verdict, and a run
whose boundary layer was not saved is labelled ``unchecked`` rather than
being assumed sound.

The table is a derived artefact. It can always be rebuilt from the run
directories, which remain the only source of truth.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from ..analysis.trust import sweep_report
from ..data import atomic_json, sha256_file, timestamp_label
from ..geometry import inspect_points, load_coordinates
from ..flow import actual_conditions

SCHEMA_VERSION = 2

COLUMNS = (
    "run_id",
    "case_dir",
    "airfoil",
    "geometry_kind",
    "naca",
    "coordinate_file",
    "thickness_ratio",
    "thickness_x",
    "max_camber",
    "trailing_edge_gap",
    "geometry_valid",
    "re",
    "mach",
    "flow_relation",
    "re_reference",
    "mach_reference",
    "flow_type",
    "reference_cl",
    "condition_status",
    "alpha",
    "CL",
    "CD",
    "CDp",
    "CM",
    "top_xtr",
    "bot_xtr",
    "converged",
    "verdict",
    "reasons",
    "governing_surface",
    "separated_fraction",
    "longest_run_fraction",
    "run_count",
    "max_h",
    "max_h_x",
    "has_boundary_layer",
    "ncrit",
    "xtr_top",
    "xtr_bottom",
    "panels",
    "panel_bunching",
    "te_le_ratio",
)

REASON_SEPARATOR = ";"


def discover_runs(root: Path) -> list[Path]:
    """Every run directory under ``root``, in a stable order.

    A run directory is the one holding ``run.json``. Attempt directories
    also hold geometry files, so discovery is anchored on that name.
    """
    return sorted(
        path.parent
        for path in Path(root).expanduser().resolve().glob("**/run.json")
    )


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _is_readable_run(run_dir: Path) -> bool:
    """A run record must parse; an empty document is not a run.

    Malformed configuration is reported as unreadable rather than being
    presented as a run that requested no angles, which would look like a
    legitimate empty result.
    """
    path = run_dir / "run.json"
    if not path.is_file():
        return False
    return bool(_read_json(path))


def _geometry_descriptors(run_dir: Path) -> dict:
    """Compact shape parameters for the airfoil actually solved.

    These come from the normalized coordinates the solver used, not from
    the requested design, so a repaired or rescaled contour is described
    as it was run. They exist to give a surrogate model a smooth,
    self-controlled parameterization without depending on any external
    airfoil library.
    """
    blank = {
        "thickness_ratio": None,
        "thickness_x": None,
        "max_camber": None,
        "trailing_edge_gap": None,
        "geometry_valid": None,
    }
    path = run_dir / "geometry.dat"
    if not path.is_file():
        return blank
    try:
        report, _ = inspect_points(load_coordinates(path))
    except (OSError, ValueError):
        return blank
    blank.update(
        {
            "thickness_ratio": report.get("thickness_ratio"),
            "thickness_x": report.get("thickness_x"),
            "max_camber": report.get("max_camber"),
            "trailing_edge_gap": report.get("trailing_edge_gap"),
            "geometry_valid": bool(report.get("valid")),
        }
    )
    return blank


def _polar_rows(path: Path) -> dict:
    """Solved rows keyed by angle, empty when no polar was written."""
    if not path.is_file():
        return {}
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            alpha = _number(row.get("alpha"))
            if alpha is None:
                continue
            rows[round(alpha, 6)] = row
    return rows


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value) -> str | None:
    return None if value is None else str(value)


def load_run(run_dir: Path, root: Path) -> dict:
    """Configuration, geometry, solver settings, and evidence for one run."""
    document = _read_json(run_dir / "run.json")
    config = document.get("config") or {}
    source = config.get("input") or {}
    settings = config.get("solver_settings") or {}
    try:
        run_id = str(run_dir.relative_to(root))
    except ValueError:
        run_id = str(run_dir)
    requested = [_number(a) for a in (document.get("requested_alphas") or [])]
    requested = sorted({round(a, 6) for a in requested if a is not None})
    failed = [
        _number(a)
        for a in (document.get("summary") or {}).get("failed_alphas", []) or []
    ]
    failed = sorted({round(a, 6) for a in failed if a is not None})
    report = sweep_report(run_dir, requested, failed)
    by_alpha = {round(point["alpha"], 6): point for point in report["points"]}
    geometry = _geometry_descriptors(run_dir)
    return {
        "run_id": run_id,
        "case_dir": str(run_dir),
        "airfoil": document.get("airfoil"),
        "geometry_kind": source.get("kind"),
        "naca": source.get("naca"),
        "coordinate_file": source.get("path", source.get("airfoil")),
        "re": _number(config.get("re")),
        "mach": _number(config.get("mach")),
        "flow_relation": config.get("flow_relation"),
        "flow_config": config,
        "ncrit": _number(settings.get("ncrit")),
        "xtr_top": _number(settings.get("xtr_top")),
        "xtr_bottom": _number(settings.get("xtr_bottom")),
        "panels": settings.get("panels"),
        "panel_bunching": _number(settings.get("panel_bunching")),
        "te_le_ratio": _number(settings.get("te_le_ratio")),
        "has_boundary_layer": bool(config.get("boundary_layer")),
        "geometry": geometry,
        "points": by_alpha,
        "polar": _polar_rows(run_dir / "polar.csv"),
        "requested": requested,
    }


def _row_for(run: dict, alpha: float) -> dict:
    key = round(alpha, 6)
    point = run["points"].get(key) or {}
    polar = run["polar"].get(key) or {}
    governing = point.get("governing_surface")
    evidence = point.get(governing) if governing else None
    evidence = evidence or point.get("upper") or point.get("lower") or {}
    cl = _number(polar.get("CL"))
    condition = None
    condition_status = "missing_lift"
    if cl is not None:
        try:
            condition = actual_conditions(run["flow_config"], cl)
            if condition is not None and (
                not all(math.isfinite(v) for v in condition)
                or condition[0] <= 0
                or not 0 <= condition[1] < 1
            ):
                condition = None
            condition_status = "ok" if condition else "invalid_condition"
        except (KeyError, TypeError, ValueError):
            condition_status = "invalid_condition"
    row = {
        "run_id": run["run_id"],
        "case_dir": run["case_dir"],
        "airfoil": _text(run["airfoil"]),
        "geometry_kind": _text(run["geometry_kind"]),
        "naca": _text(run["naca"]),
        "coordinate_file": _text(run["coordinate_file"]),
        "re": condition[0] if condition else None,
        "mach": condition[1] if condition else None,
        "flow_relation": run.get("flow_relation"),
        "re_reference": run["re"],
        "mach_reference": run["mach"],
        "flow_type": run["flow_config"].get("flow_type", 1),
        "reference_cl": run["flow_config"].get("reference_cl", 1.0),
        "condition_status": condition_status,
        "alpha": key,
        "CL": _number(polar.get("CL")),
        "CD": _number(polar.get("CD")),
        "CDp": _number(polar.get("CDp")),
        "CM": _number(polar.get("CM")),
        "top_xtr": _number(polar.get("Top_Xtr")),
        "bot_xtr": _number(polar.get("Bot_Xtr")),
        "converged": point.get("converged", False),
        "verdict": point.get("verdict", "unchecked"),
        "reasons": REASON_SEPARATOR.join(point.get("reasons") or []),
        "governing_surface": _text(governing),
        "separated_fraction": evidence.get("separated_fraction"),
        "longest_run_fraction": evidence.get("longest_run_fraction"),
        "run_count": evidence.get("run_count"),
        "max_h": evidence.get("max_h"),
        "max_h_x": evidence.get("max_h_x"),
        "has_boundary_layer": run["has_boundary_layer"],
        "ncrit": run["ncrit"],
        "xtr_top": run["xtr_top"],
        "xtr_bottom": run["xtr_bottom"],
        "panels": run["panels"],
        "panel_bunching": run["panel_bunching"],
        "te_le_ratio": run["te_le_ratio"],
    }
    row.update(run["geometry"])
    return row


def collect_rows(root: Path) -> tuple[list[dict], list[dict]]:
    """Build every row, plus a per-run provenance record."""
    root = Path(root).expanduser().resolve()
    rows = []
    provenance = []
    for run_dir in discover_runs(root):
        if not _is_readable_run(run_dir):
            provenance.append(
                {
                    "run_id": str(run_dir),
                    "status": "unreadable",
                    "error": "run.json is missing or does not parse",
                }
            )
            continue
        try:
            run = load_run(run_dir, root)
        except (OSError, ValueError) as error:
            provenance.append(
                {
                    "run_id": str(run_dir),
                    "status": "unreadable",
                    "error": str(error),
                }
            )
            continue
        run_rows = [_row_for(run, alpha) for alpha in run["requested"]]
        rows.extend(run_rows)
        provenance.append(
            {
                "run_id": run["run_id"],
                "status": "read",
                "airfoil": run["airfoil"],
                "re": run["re"],
                "mach": run["mach"],
                "flow_relation": run.get("flow_relation"),
                "requested_points": len(run["requested"]),
                "rows": len(run_rows),
                "boundary_layer": run["has_boundary_layer"],
                "run_json_sha256": _sha256(run_dir / "run.json"),
                "polar_sha256": _sha256(run_dir / "polar.csv"),
                "geometry_sha256": _sha256(run_dir / "geometry.dat"),
            }
        )
    rows.sort(key=lambda row: (row["run_id"], row["alpha"]))
    return rows, provenance


def _sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def summarise(rows: list[dict]) -> dict:
    """Counts by verdict, plus how many rows carry usable evidence."""
    verdicts = {}
    for row in rows:
        verdicts[row["verdict"]] = verdicts.get(row["verdict"], 0) + 1
    checked = [r for r in rows if r["verdict"] != "unchecked"]
    return {
        "rows": len(rows),
        "runs": len({row["run_id"] for row in rows}),
        "verdicts": verdicts,
        "rows_with_evidence": len(checked),
        "rows_without_evidence": len(rows) - len(checked),
        "runs_without_boundary_layer": len(
            {r["run_id"] for r in rows if not r["has_boundary_layer"]}
        ),
    }


def write_dataset(root: Path, destination: Path) -> dict:
    """Write the table and its provenance, returning a summary."""
    destination = Path(destination).expanduser().resolve()
    rows, provenance = collect_rows(root)
    summary = summarise(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in COLUMNS})
    document = {
        "schema_version": SCHEMA_VERSION,
        "created": timestamp_label(),
        "root": str(Path(root).expanduser().resolve()),
        "destination": str(destination),
        "columns": list(COLUMNS),
        "summary": summary,
        "runs": provenance,
        "notes": [
            "verdict describes solver-domain evidence, not physical stall",
            "unchecked means the run saved no boundary layer, so no "
            "evidence exists; it is not a pass",
            "a failed solve leaves no polar row, so its values are empty "
            "and its verdict carries the failure",
        ],
    }
    atomic_json(destination.with_suffix(".provenance.json"), document)
    return summary


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return value
