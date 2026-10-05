"""One discovery and interpolation policy for all post-processing tools."""

from __future__ import annotations
import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from .solver_settings import SolverSettings
from .data import atomic_text, fingerprint, gapped_rows, parse_polar_file


@dataclass
class SavedRun:
    root: Path
    manifest: dict
    rows: list[dict]

    @property
    def config(self):
        return self.manifest["config"]

    @property
    def requested(self):
        return self.manifest.get("requested_alphas", self.config["alphas"])

    @property
    def airfoil(self):
        return self.manifest["airfoil"]

    @property
    def identity(self):
        source = self.config["input"]
        return (
            "NACA" + source["naca"]
            if source["kind"] == "naca"
            else source["sha256"]
        )

    @property
    def condition(self):
        return (
            self.config["re"],
            self.config["mach"],
            self.config.get("flow_type", 1),
            self.config.get("reference_cl", 1.0),
        )


def discover_runs(
    folders: list[Path], allow_empty: bool = False
) -> list[SavedRun]:
    pending = [p.expanduser().resolve() for p in folders]
    visited, manifests = set(), set()
    while pending:
        folder = pending.pop()
        if folder in visited:
            continue
        visited.add(folder)
        if not folder.is_dir():
            raise ValueError(f"Result folder not found: {folder}")
        if (folder / "failure.json").is_file():
            continue
        if (folder / "run.json").is_file():
            manifests.add(folder / "run.json")
        elif (folder / "study.json").is_file():
            pending.extend(
                folder / p
                for p in json.loads((folder / "study.json").read_text())[
                    "folders"
                ]
            )
        elif (folder / "batch.json").is_file():
            pending.extend(
                folder / case
                for case in json.loads((folder / "batch.json").read_text())[
                    "cases"
                ]
                if (folder / case).is_dir()
                and not (folder / f"{case}.failure.json").exists()
            )
        else:
            # Walk through manifests rather than bypassing study/batch failure
            # records.
            children = [
                p
                for p in folder.iterdir()
                if p.is_dir()
                and not p.is_symlink()
                and p.name
                not in (
                    "attempts",
                    "preview",
                    "pressure_vectors",
                    "matplotlib_cache",
                    "_archive",
                    "_cache",
                    "settings",
                    "raw",
                )
            ]
            pending.extend(children)
    runs = []
    for path in sorted(manifests):
        if (path.parent.parent / f"{path.parent.name}.failure.json").exists():
            continue
        manifest = json.loads(path.read_text())
        polar_name = manifest["config"]["polar_filename"]
        if Path(polar_name).name != polar_name:
            raise ValueError(f"Invalid saved polar filename in {path}")
        runs.append(
            SavedRun(
                path.parent,
                manifest,
                parse_polar_file(path.parent / polar_name),
            )
        )
    if not runs and not allow_empty:
        raise ValueError("No usable saved run.json found")
    return runs


def interpolate(
    run: SavedRun, key: str, value: float
) -> tuple[dict | None, str]:
    """Unique exact match or adjacent converged bracket, with no
    extrapolation.
    """
    if not math.isfinite(value):
        raise ValueError("Interpolation target must be finite")
    ordered = gapped_rows(run.rows, run.requested)
    candidates = []
    for row in ordered:
        if math.isfinite(row[key]) and math.isclose(
            row[key], value, rel_tol=0, abs_tol=1e-9
        ):
            candidates.append((dict(row), "exact"))
    for left, right in zip(ordered, ordered[1:]):
        if not all(
            math.isfinite(r[k])
            for r in (left, right)
            for k in ("alpha", "CL", "CD", "CM")
        ):
            continue
        if (left[key] - value) * (right[key] - value) < 0:
            ratio = (value - left[key]) / (right[key] - left[key])
            row = {
                k: left[k] + ratio * (right[k] - left[k])
                for k in ("alpha", "CL", "CD", "CM")
            }
            candidates.append((row, "interpolated"))
    if len(candidates) == 1:
        return candidates[0]
    return None, (
        "ambiguous_multiple_crossings" if candidates else "not_bracketed"
    )


def write_records(
    path: Path, records: list[dict], fields: list[str] | None = None
) -> None:
    fields = (
        fields
        or list(dict.fromkeys(k for r in records for k in r))
        or ["status"]
    )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    atomic_text(path, output.getvalue())


def variant_key(run: SavedRun) -> str:
    c = run.config
    return fingerprint(
        {
            "input": run.identity,
            "flow_type": c.get("flow_type", 1),
            "reference_cl": c.get("reference_cl", 1),
            "iterations": c["iterations"],
            "binary": c.get("binary_sha256"),
            "version": c.get("version"),
            "solver_settings": SolverSettings.from_config(c).to_dict(),
        }
    )[:10]


def discover_failures(folders: list[Path]) -> list[tuple[Path, dict]]:
    """Honor active batch/study inventories, including cases without run
    manifests.
    """
    pending = [p.expanduser().resolve() for p in folders]
    visited, failures = set(), set()
    while pending:
        folder = pending.pop()
        if folder in visited or not folder.is_dir():
            continue
        visited.add(folder)
        own = folder / "failure.json"
        sibling = folder.parent / f"{folder.name}.failure.json"
        if own.is_file() or sibling.is_file():
            failures.add(own if own.is_file() else sibling)
        elif (folder / "batch.json").is_file():
            for case in json.loads((folder / "batch.json").read_text())[
                "cases"
            ]:
                failure = folder / f"{case}.failure.json"
                if failure.is_file():
                    failures.add(failure)
                else:
                    pending.append(folder / case)
        elif (folder / "study.json").is_file():
            pending.extend(
                folder / p
                for p in json.loads((folder / "study.json").read_text())[
                    "folders"
                ]
            )
        elif not (folder / "run.json").is_file():
            failures.update(folder.glob("*.failure.json"))
            pending.extend(
                p
                for p in folder.iterdir()
                if p.is_dir()
                and not p.is_symlink()
                and p.name
                not in (
                    "attempts",
                    "preview",
                    "pressure_vectors",
                    "matplotlib_cache",
                    "_archive",
                    "_cache",
                    "settings",
                    "raw",
                )
            )
    return [(path, json.loads(path.read_text())) for path in sorted(failures)]
