"""History and named presets preserve traceable scientific files."""

import json
import os
import zipfile
from pathlib import Path

from ..data import atomic_json
from ..analysis.report import report_page


def within(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path must stay inside the results directory")
    return path


def history(app, include_archive=False):
    root = app.run_root
    if not root.exists():
        return []
    results = []
    job_roots = set()
    for path in root.glob("projects/*/job.json"):
        try:
            item = json.loads(path.read_text())
            job_roots.add(path.parent)
            results.append(
                {
                    "id": str(path.parent.relative_to(root)),
                    "kind": item["spec"]["kind"],
                    "title": item.get("title", path.parent.name),
                    "status": item.get("status", "unknown"),
                    "created": item.get("created", path.parent.name),
                    "report": (
                        str((path.parent / "report.html").relative_to(root))
                        if (path.parent / "report.html").exists()
                        else None
                    ),
                }
            )
        except (OSError, ValueError, KeyError):
            continue
    for name, kind in (
        ("run.json", "polar"),
        ("wing_run.json", "wing"),
        ("lift_run.json", "lift"),
    ):
        for path in root.rglob(name):
            if any(p in job_roots for p in path.parents):
                continue
            if not include_archive and "_archive" in path.parts:
                continue
            try:
                item = json.loads(path.read_text())
                config = item["config"]
                results.append(
                    {
                        "id": str(path.parent.relative_to(root)),
                        "kind": kind,
                        "title": item.get(
                            "airfoil", config.get("name", path.parent.name)
                        ),
                        "status": item.get("status", "unknown"),
                        "created": item.get(
                            "created_at", str(path.stat().st_mtime)
                        ),
                        "report": (
                            str(
                                (path.parent / "report.html").relative_to(root)
                            )
                            if (path.parent / "report.html").exists()
                            else None
                        ),
                    }
                )
            except (OSError, ValueError, KeyError):
                continue
    return sorted(
        results,
        key=lambda r: max(
            (
                p.stat().st_mtime
                for name in (
                    "job.json",
                    "run.json",
                    "wing_run.json",
                    "lift_run.json",
                )
                if (p := root / r["id"] / name).is_file()
            ),
            default=0,
        ),
        reverse=True,
    )


def presets(app):
    path = app.run_root / "settings" / "presets.json"
    return (
        json.loads(path.read_text())
        if path.exists()
        else {"presets": {}, "last": {}}
    )


def save_preset(app, name, spec):
    if not name.strip() or len(name) > 80:
        raise ValueError("Preset name must contain 1..80 characters")
    state = presets(app)
    state["presets"][name.strip()] = spec
    atomic_json(app.run_root / "settings" / "presets.json", state)


def save_last(app, spec):
    state = presets(app)
    state["last"] = spec
    atomic_json(app.run_root / "settings" / "presets.json", state)


def write_index(app):
    rows = history(app)
    report_page(
        app.run_root / "index.html",
        "我的气动计算",
        "日常使用本页或本地工作台。projects 是新版计算；原有 study、single_calculations 和 wing "
        "目录保持原路径，settings 保存命名设置。不要手动移动单次计算内的 "
        "data、attempts 与清单文件。",
        rows,
        [(r["title"], r["report"]) for r in rows if r["report"]],
    )
    return app.run_root / "index.html"


def export_report(folder: Path, output: Path):
    folder = folder.resolve()
    output = output.resolve()
    if not (folder / "report.html").exists():
        raise ValueError("This result has no report yet")
    files = [
        p
        for p in folder.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and p.resolve() != output
        and p.suffix != ".zip"
    ]
    if sum(p.stat().st_size for p in files) > 500 * 1024**2:
        raise ValueError("Report exceeds the 500 MB export limit")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, p.relative_to(folder))
    os.replace(temp, output)
    return output


def spec_from_history(folder):
    job = folder / "job.json"
    if job.exists():
        return json.loads(job.read_text())["spec"]
    if (folder / "wing_run.json").exists():
        from ..ui.wing import clean_model

        return {
            "kind": "wing",
            "model": clean_model(
                json.loads((folder / "wing_run.json").read_text())["config"]
            ),
        }
    path = folder / (
        "run.json" if (folder / "run.json").exists() else "lift_run.json"
    )
    c = json.loads(path.read_text())["config"]
    source = c["input"]
    spec = {
        "kind": "polar" if path.name == "run.json" else "lift",
        "source": (
            {"naca": source["naca"]}
            if source["kind"] == "naca"
            else {
                "airfoil": (
                    str(folder / "airfoil_input.dat")
                    if (folder / "airfoil_input.dat").exists()
                    else source["path"]
                )
            }
        ),
        "re": c["re"],
        "mach": c["mach"],
        "flow": c.get("flow_assumptions", {}),
        "solver_settings": c.get("solver_settings"),
        "iterations": c["iterations"],
        "timeout": c["timeout"],
        "retries": c["retries"],
        "boundary_layer": c.get("boundary_layer", False),
    }
    if path.name == "run.json":
        spec.update(
            aseq=[c["alpha_start"], c["alpha_end"], c["alpha_step"]],
            flow_type=c.get("flow_type", 1),
            reference_cl=c.get("reference_cl", 1.0),
            retry_step=c["retry_step"],
            target_cl=c.get("target_cl"),
            adaptive_rounds=c.get("adaptive", {}).get("rounds", 0),
            save_pressure_vectors=c.get("save_pressure_vectors", True),
        )
        if c.get("explicit_alphas"):
            spec["alpha_targets"] = c["alphas"]
    else:
        spec["targets"] = c["targets"]
    return spec
