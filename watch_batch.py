"""Progress of a long solver batch, from the files it has already written.

Reports the state of the most recent batch directory without touching
the solver, so it can be run while the batch is still working. Every
number comes from artefacts on disk: the count of run records, their
statuses, and the spacing of their modification times.

Usage:
    python watch_batch.py
    python watch_batch.py xfoil_runs/uiuc_20261002_232834
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

DEFAULT_ROOT = Path("xfoil_runs")
BATCH_GLOB = "uiuc_*"


def latest_batch(root: Path = DEFAULT_ROOT) -> Path:
    """Most recently modified batch directory under ``root``."""
    candidates = [
        path for path in Path(root).glob(BATCH_GLOB) if path.is_dir()
    ]
    if not candidates:
        raise SystemExit(f"No {BATCH_GLOB} directory under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_records(batch: Path) -> list:
    """Every ``run.json`` below the batch, oldest first."""
    records = list(Path(batch).glob("*/*/run.json"))
    records.sort(key=lambda p: p.stat().st_mtime)
    return records


def read_status(path: Path) -> str:
    try:
        return str(json.loads(path.read_text()).get("status", "?"))
    except (OSError, ValueError):
        return "unreadable"


def report(batch: Path, expected: int | None = None) -> dict:
    records = run_records(batch)
    if not records:
        return {"batch": str(batch), "done": 0, "expected": expected}
    times = [p.stat().st_mtime for p in records]
    done = len(records)
    elapsed = max(times[-1] - times[0], 1e-9)
    rate = done / elapsed
    statuses: dict = {}
    for path in records:
        name = read_status(path)
        statuses[name] = statuses.get(name, 0) + 1
    usable = statuses.get("ok", 0) + statuses.get("partial", 0)
    remaining = None if expected is None else max(expected - done, 0)
    return {
        "batch": str(batch),
        "done": done,
        "expected": expected,
        "elapsed_minutes": elapsed / 60.0,
        "rate_per_minute": rate * 60.0,
        "remaining": remaining,
        "eta_minutes": None if remaining is None else remaining / rate / 60.0,
        "statuses": statuses,
        "usable_share": usable / done,
        "last_write": times[-1],
    }


def format_report(summary: dict) -> str:
    lines = [f"目录    {summary['batch']}"]
    done = summary["done"]
    expected = summary.get("expected")
    if expected:
        lines.append(
            f"进度    {done}/{expected} ({done / expected * 100:.1f}%)"
        )
    else:
        lines.append(f"进度    {done} 个 run")
    if not done:
        return "\n".join(lines)
    lines.append(
        f"已运行  {summary['elapsed_minutes']:.1f} 分钟   "
        f"速率 {summary['rate_per_minute']:.1f}/分钟"
    )
    if summary["eta_minutes"] is not None:
        finish = time.time() + summary["eta_minutes"] * 60
        lines.append(
            f"预计    {summary['eta_minutes']:.0f} 分钟后完成 "
            f"({time.strftime('%H:%M', time.localtime(finish))})"
        )
    statuses = summary["statuses"]
    lines.append(
        "状态    " + "  ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
    )
    lines.append(f"可用率  {summary['usable_share'] * 100:.0f}%")
    idle = time.time() - summary["last_write"]
    lines.append(f"最近写入 {idle:.0f} 秒前")
    if idle > 120:
        lines.append("        (超过 2 分钟没有新记录，批次可能已停止)")
    return "\n".join(lines)


def main(argv: list) -> int:
    flags = [a for a in argv[1:] if a.startswith("--")]
    positional = [a for a in argv[1:] if not a.startswith("--")]
    expected = None
    for flag in flags:
        if flag.startswith("--expected="):
            expected = int(flag.split("=", 1)[1])
    try:
        batch = Path(positional[0]) if positional else latest_batch()
        summary = report(batch, expected)
    except SystemExit as error:
        print(error)
        return 1
    print(format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
