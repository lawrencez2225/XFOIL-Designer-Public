"""Where the solver stops being trustworthy, summarised across runs.

The trust layer answers the question for one angle of one run. This
turns those answers into a table keyed by configuration, because the
useful question is not "is this point suspect" but "how far can this
airfoil be pushed, and does that distance depend on the flow condition".

Reads only what is already on disk. Runs no solver.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from .trust import sweep_report

AIRFOIL_PATTERN = re.compile(
    r"NACA(?P<naca>[\w-]+)_M(?P<mach>[\d.]+)_Re(?P<re>\d+)"
)

COLUMNS = (
    "run_id",
    "airfoil",
    "mach",
    "re",
    "points",
    "last_trusted_alpha",
    "first_suspect_alpha",
    "first_failed_alpha",
    "cl_max",
    "alpha_at_cl_max",
    "verdicts",
)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _polar_rows(path: Path) -> list:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def label_for(run_dir: Path, document: dict) -> dict:
    """Recover the configuration from the directory name or the record."""
    match = AIRFOIL_PATTERN.search(run_dir.name)
    if match:
        return {
            "airfoil": f"NACA{match.group('naca')}",
            "mach": _number(match.group("mach")),
            "re": _number(match.group("re")),
        }
    config = document.get("config") or {}
    return {
        "airfoil": document.get("airfoil") or run_dir.name,
        "mach": _number(config.get("mach")),
        "re": _number(config.get("re")),
    }


def describe_run(run_dir: Path) -> dict | None:
    """One row of the table, or None when the run holds no evidence."""
    manifest = run_dir / "run.json"
    if not manifest.is_file():
        return None
    try:
        document = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return None
    requested = [_number(a) for a in (document.get("requested_alphas") or [])]
    requested = [a for a in requested if a is not None]
    rows = _polar_rows(run_dir / "polar.csv")
    alphas = requested or [
        a for a in (_number(r.get("alpha")) for r in rows) if a is not None
    ]
    if not alphas:
        return None
    failed = (document.get("summary") or {}).get("failed_alphas") or []
    report = sweep_report(run_dir, alphas, failed)
    verdicts = {}
    for point in report["points"]:
        name = point["verdict"]
        verdicts[name] = verdicts.get(name, 0) + 1
    evidence = report["evidence"] if "evidence" in report else {}
    limit = _limit_from(report["points"])
    lift = [(_number(r.get("CL")), _number(r.get("alpha"))) for r in rows]
    lift = [(c, a) for c, a in lift if c is not None and a is not None]
    return {
        "run_id": run_dir.name,
        **label_for(run_dir, document),
        "points": len(alphas),
        "last_trusted_alpha": limit["last_trusted"],
        "first_suspect_alpha": limit["first_suspect"],
        "first_failed_alpha": limit["first_failed"],
        "cl_max": max((c for c, _ in lift), default=None),
        "alpha_at_cl_max": (max(lift)[1] if lift else None),
        "verdicts": verdicts,
        "evidence": evidence,
    }


def _limit_from(points: list) -> dict:
    """Boundary of the trusted range, using the verdicts as given."""
    trusted = [
        float(p["alpha"])
        for p in points
        if p.get("verdict") in {"ok", "suspect"} and p.get("converged", True)
    ]
    suspect = [
        float(p["alpha"]) for p in points if p.get("verdict") == "suspect"
    ]
    failed = [
        float(p["alpha"])
        for p in points
        if not p.get("converged", True)
        or p.get("verdict") in {"invalid", "retry"}
    ]
    if suspect:
        first_suspect = min(suspect)
        trusted = [a for a in trusted if a < first_suspect]
    return {
        "last_trusted": max(trusted) if trusted else None,
        "first_suspect": min(suspect) if suspect else None,
        "first_failed": min(failed) if failed else None,
    }


def summarise_root(root: Path) -> tuple:
    """One row per run found under ``root``."""
    rows = []
    for manifest in sorted(
        Path(root).expanduser().resolve().glob("**/run.json")
    ):
        described = describe_run(manifest.parent)
        if described:
            rows.append(described)
    rows.sort(key=lambda r: (str(r["airfoil"]), r["mach"] or 0))
    return rows, _distribution(rows)


def _distribution(rows: list) -> dict:
    values = [
        r["last_trusted_alpha"]
        for r in rows
        if r["last_trusted_alpha"] is not None
    ]
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "max": ordered[-1],
        "counts": {str(a): ordered.count(a) for a in sorted(set(ordered))},
    }


def write_summary(root: Path, destination: Path) -> dict:
    """Write the table and return the distribution it shows."""
    rows, distribution = summarise_root(root)
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(COLUMNS), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            written = dict(row)
            written["verdicts"] = json.dumps(row["verdicts"])
            writer.writerow(
                {k: ("" if v is None else v) for k, v in written.items()}
            )
    return {
        "rows": len(rows),
        "destination": str(destination),
        "distribution": distribution,
    }
