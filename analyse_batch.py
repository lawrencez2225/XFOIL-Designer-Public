"""One pass over a finished batch: domain, dataset, surrogate, summary.

Runs no solver. Every number comes from artefacts already on disk, so it
is safe to run while nothing else is using the machine and safe to re-run
after any additional batch.

Steps, in order, because each depends on the one before:

1. Domain table   - per-run trusted angle, from the trust layer.
2. Dataset        - the flat table the surrogate learns from.
3. Sufficiency    - whether the dataset can support a fit at all.
4. Holdout scores - selected on the training pool, reported once on runs
                    reserved before any fitting.

Usage:
    python analyse_batch.py
    python analyse_batch.py xfoil_runs/uiuc_20261002_232834 --folds 5
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_ROOT = Path("xfoil_runs")
BATCH_GLOB = "uiuc_*"
OUTPUT_DIR = "datasets"
RECORD_DIR = "models"


def latest_batch(root: Path = DEFAULT_ROOT) -> Path:
    candidates = [p for p in Path(root).glob(BATCH_GLOB) if p.is_dir()]
    if not candidates:
        raise SystemExit(f"No {BATCH_GLOB} directory under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_batch(path: Path) -> Path:
    """Accept a batch directory, or a tree that holds several batches.

    A tree holding several batches is named after the newest one rather
    than after the tree, so a report always says which batch it read. A
    tree that holds results directly but no batch directory is used as
    given, which is what an older layout looks like.
    """
    path = Path(path)
    if (path / "run.json").is_file():
        return path
    try:
        return latest_batch(path)
    except SystemExit:
        return path


def step_domain(root: Path, destination: Path) -> dict:
    from xfoil_mac.analysis.domain import summarise_root, write_summary
    from xfoil_mac.analysis.sampling import domain_limits_from_measurements

    rows, distribution = summarise_root(root)
    report = write_summary(root, destination)
    measured = domain_limits_from_measurements(rows)
    return {
        "runs": len(rows),
        "distribution": distribution,
        "table": report["destination"],
        "conditions": measured,
    }


def step_dataset(root: Path, destination: Path) -> dict:
    from xfoil_mac.storage.dataset import write_dataset

    return write_dataset(root, destination)


def step_sufficiency(rows: list) -> dict:
    from xfoil_mac.analysis.surrogate import refusal_reason, sufficiency

    report = sufficiency(rows)
    report["refusal"] = refusal_reason(report)
    return report


def step_holdout(rows: list, *, folds: int, log_path: Path) -> dict:
    from xfoil_mac.analysis import surrogate

    outcome = surrogate.holdout_evaluate(
        rows,
        surrogate.boosting_factory,
        log_path=log_path,
        folds=folds,
    )
    if not outcome.get("reported"):
        return {"reported": False, "reason": outcome.get("reason")}
    document = outcome["document"]
    return {
        "reported": True,
        "train_runs": document["train_runs"],
        "test_runs": document["test_runs"],
        "selection": document["selection_scores"],
        "test": document["test_scores"],
        "per_tier": {
            target: outcome["test"][target]["tiers"]
            for target in outcome["test"]
        },
        "log": document.get("log"),
    }


def _format_interval(entry: dict) -> str:
    if not entry or entry.get("mae") is None:
        return "n/a"
    low, high = entry.get("low"), entry.get("high")
    if low is None or high is None:
        return f"{entry['mae']:.5f}"
    return f"{entry['mae']:.5f} [{low:.5f}, {high:.5f}]"


def format_report(summary: dict) -> str:
    lines = []
    lines.append("=" * 68)
    lines.append(f"批次     {summary['batch']}")
    lines.append(f"耗时     {summary['seconds']:.0f} 秒")
    lines.append("=" * 68)

    domain = summary.get("domain") or {}
    lines.append("")
    lines.append("步骤 1  有效域表格")
    if domain:
        dist = domain.get("distribution") or {}
        lines.append(f"  run 数     {domain.get('runs')}")
        if dist.get("n"):
            lines.append(
                f"  可信上界   n={dist['n']}  最小 {dist['min']:g}"
                f"  中位 {dist['median']:g}  最大 {dist['max']:g}"
            )
        lines.append(f"  实测工况   {len(domain.get('conditions') or [])}")
        lines.append(f"  表格       {domain.get('table')}")

    dataset = summary.get("dataset") or {}
    lines.append("")
    lines.append("步骤 2  数据集")
    if dataset:
        lines.append(f"  行数       {dataset.get('rows')}")
        lines.append(f"  run 数     {dataset.get('runs')}")
        verdicts = dataset.get("verdicts") or {}
        lines.append(
            "  判定       "
            + "  ".join(f"{k}={v}" for k, v in sorted(verdicts.items()))
        )

    suff = summary.get("sufficiency") or {}
    lines.append("")
    lines.append("步骤 3  充分性")
    if suff:
        lines.append(f"  可训练行   {suff.get('usable_rows')}")
        lines.append(f"  run 数     {suff.get('runs')}")
        lines.append(f"  特征取值   {suff.get('distinct_conditions')}")
        if suff.get("fit_allowed"):
            lines.append("  可以拟合   yes")
        else:
            lines.append("  可以拟合   no")
            lines.append(f"  原因       {suff.get('refusal')}")

    holdout = summary.get("holdout") or {}
    lines.append("")
    lines.append("步骤 4  留出评估")
    if not holdout.get("reported"):
        lines.append(f"  未出报告   {holdout.get('reason')}")
    else:
        lines.append(
            f"  训练池 {holdout['train_runs']} run / "
            f"测试集 {holdout['test_runs']} run"
        )
        lines.append("")
        lines.append(
            f"  {'目标':<5} {'验证集(选模型)':>28} {'测试集(只评一次)':>28}"
        )
        for target in sorted(holdout["test"]):
            selection = holdout["selection"].get(target, {})
            test = holdout["test"].get(target, {})
            lines.append(
                f"  {target:<5} {_format_interval(selection):>28} "
                f"{_format_interval(test):>28}"
            )
        lines.append("")
        lines.append("  测试集按 trust 分层:")
        for target in sorted(holdout.get("per_tier", {})):
            tiers = holdout["per_tier"][target]
            rendered = "  ".join(
                f"{name}={entry['mae']:.5f}(n={entry['n']})"
                for name, entry in sorted(tiers.items())
                if entry.get("mae") is not None
            )
            lines.append(f"    {target:<5} {rendered}")
        if holdout.get("log"):
            lines.append(f"  审计日志   {holdout['log']}")

    warnings = summary.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("注意")
        for item in warnings:
            lines.append(f"  - {item}")
    return "\n".join(lines)


def analyse(
    batch: Path,
    *,
    folds: int = 4,
    root: Path = DEFAULT_ROOT,
    scope: Path | None = None,
) -> dict:
    """Analyse one batch, or every batch under ``scope``.

    ``batch`` names the report. ``scope`` names the tree the dataset and
    the scores read from; when it is None the batch itself is read, so a
    single-batch run cannot silently pull in other conditions.
    """
    batch = resolve_batch(batch)
    read_from = Path(scope) if scope is not None else batch
    started = time.perf_counter()
    output_dir = Path(root) / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings = []

    domain = step_domain(read_from, output_dir / f"domain_{batch.name}.csv")
    dataset = step_dataset(read_from, output_dir / f"dataset_{batch.name}.csv")

    from xfoil_mac.storage.dataset import collect_rows

    rows, _ = collect_rows(read_from)
    if read_from != batch:
        conditions = {
            (round(r["re"]), round(r["mach"], 4)) for r in rows if r.get("re")
        }
        warnings.append(
            f"scored across {len(conditions)} flow condition(s) from "
            f"{read_from}; a single batch of one condition cannot be "
            f"fitted on its own"
        )
    suff = step_sufficiency(rows)
    if suff.get("usable_rows"):
        verdicts = dataset.get("verdicts") or {}
        unchecked = verdicts.get("unchecked", 0)
        if unchecked:
            warnings.append(
                f"{unchecked} rows are unchecked and are excluded from "
                f"learning; enabling boundary_layer on those runs would "
                f"add them"
            )
    holdout = step_holdout(
        rows,
        folds=folds,
        log_path=Path(root) / RECORD_DIR / "experiments.jsonl",
    )
    return {
        "batch": str(batch),
        "seconds": time.perf_counter() - started,
        "domain": domain,
        "dataset": dataset,
        "sufficiency": suff,
        "holdout": holdout,
        "warnings": warnings,
    }


def main(argv: list) -> int:
    flags = [a for a in argv[1:] if a.startswith("--")]
    positional = [a for a in argv[1:] if not a.startswith("--")]
    folds = 4
    for flag in flags:
        if flag.startswith("--folds="):
            folds = int(flag.split("=", 1)[1])
    all_batches = any(f == "--all" for f in flags)
    batch = resolve_batch(Path(positional[0]) if positional else DEFAULT_ROOT)
    scope = Path(DEFAULT_ROOT) if all_batches else None
    summary = analyse(batch, folds=folds, scope=scope)
    print(format_report(summary))
    label = "all" if all_batches else batch.name
    record = Path(DEFAULT_ROOT) / RECORD_DIR / f"analysis_{label}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n记录     {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
