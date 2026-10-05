"""Train the surrogate and ask it for numbers.

Two commands:

    python surrogate_tool.py train
    python surrogate_tool.py predict --airfoil naca2412 --re 5e5 --mach 0.1

``predict`` prints one row per angle and marks any row whose inputs fall
outside the values the model was trained on. A tree returns a number
everywhere, so without that mark an extrapolation is indistinguishable
from an interpolation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_ROOT = Path("xfoil_runs")
MODEL_DIR = "models"
LATEST = "surrogate_latest.joblib"


def _latest_model() -> Path:
    candidates = sorted((DEFAULT_ROOT / MODEL_DIR).glob("surrogate_*.joblib"))
    if not candidates:
        raise SystemExit(
            "No trained surrogate. Run: python surrogate_tool.py train"
        )
    return candidates[-1]


def command_train(argv: list) -> int:
    from xfoil_mac.analysis import surrogate
    from xfoil_mac.data import timestamp_label
    from xfoil_mac.storage.dataset import collect_rows

    root = DEFAULT_ROOT
    for flag in argv:
        if flag.startswith("--root="):
            root = Path(flag.split("=", 1)[1])
    print(f"读取 {root} ...")
    rows, _ = collect_rows(root)
    print(f"  {len(rows)} 行")
    report = surrogate.sufficiency(rows)
    print(f"  可训练 {report['usable_rows']} 行 / {report['runs']} run")
    if not report["fit_allowed"]:
        print("\n拒绝拟合:")
        print("  " + surrogate.refusal_reason(report))
        return 1
    outcome = surrogate.fit(rows)
    if not outcome["fitted"]:
        print(f"\n拟合失败: {outcome['reason']}")
        return 1
    destination = (
        DEFAULT_ROOT / MODEL_DIR / f"surrogate_{timestamp_label()}.joblib"
    )
    written = surrogate.save(outcome, destination)
    latest = DEFAULT_ROOT / MODEL_DIR / LATEST
    latest.write_bytes(Path(written["model"]).read_bytes())
    latest.with_suffix(".metadata.json").write_text(
        Path(written["metadata"]).read_text()
    )
    print(f"\n已保存: {written['model']}")
    print(f"         {written['metadata']}")
    print(f"         {latest}  (指向最新)")
    return 0


def command_predict(argv: list) -> int:
    from xfoil_mac.analysis import surrogate

    options = {
        "airfoil": None,
        "coordinates": None,
        "re": None,
        "mach": None,
        "alphas": None,
        "model": None,
        "csv": None,
    }
    for flag in argv:
        if not flag.startswith("--") or "=" not in flag:
            continue
        name, _, value = flag[2:].partition("=")
        if name in options:
            options[name] = value
    if not options["airfoil"] and not options["coordinates"]:
        raise SystemExit("Give --airfoil=NAME or --coordinates=PATH")
    if options["re"] is None or options["mach"] is None:
        raise SystemExit("Give --re=... and --mach=...")
    if options["alphas"]:
        alphas = [
            float(part) for part in options["alphas"].replace(",", " ").split()
        ]
    else:
        alphas = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]

    model_path = (
        Path(options["model"]) if options["model"] else _latest_model()
    )
    loaded = surrogate.load(model_path)
    metadata = loaded.get("metadata") or {}
    print(f"模型     {model_path.name}")
    print(
        f"特征     {metadata.get('feature_count', '?')} 个"
        f"  (含形状 {metadata.get('shape_count', '?')} 个, "
        f"{metadata.get('training_rows', '?')} 行训练)"
    )
    if not loaded["version_matches"]:
        print("警告     scikit-learn 版本与训练时不同，结果可能不可靠")
    target = options["airfoil"] or options["coordinates"]
    print(f"对象     {target}")
    print(f"工况     Re={float(options['re']):,.0f}  Mach={options['mach']}")
    print()
    rows = surrogate.predict(
        loaded,
        airfoil=options["airfoil"],
        coordinates=(
            Path(options["coordinates"]) if options["coordinates"] else None
        ),
        re=float(options["re"]),
        mach=float(options["mach"]),
        alphas=alphas,
    )
    print(f"{'alpha':>7} {'CL':>9} {'CD':>10} {'CM':>10}   备注")
    print("-" * 62)
    for row in rows:
        note = "外推" if row["extrapolates"] else ""
        if row["outside"]:
            note = (note + "  " + "; ".join(row["outside"])).strip()
        print(
            f"{row['alpha']:>7.2f} {row['CL']:>9.4f} "
            f"{row['CD']:>10.5f} {row['CM']:>10.5f}   {note}"
        )
    outside = sum(1 for row in rows if row["extrapolates"])
    print()
    if outside:
        print(
            f"注意: {outside}/{len(rows)} 行的输入超出训练范围，"
            f"这些预测是外推，不应直接采信。"
        )
    else:
        print("全部行的输入都在训练范围内（插值）。")
    if options["csv"]:
        import csv as csv_module

        path = Path(options["csv"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv_module.DictWriter(
                handle,
                fieldnames=[
                    "alpha",
                    "CL",
                    "CD",
                    "CM",
                    "extrapolates",
                    "outside",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        **row,
                        "outside": "; ".join(row["outside"]),
                    }
                )
        print(f"已写出   {path}")
    return 0


def command_info(argv: list) -> int:
    model_path = _latest_model()
    loaded = load_metadata(model_path)
    print(json.dumps(loaded, indent=2, ensure_ascii=False))
    return 0


def load_metadata(path: Path) -> dict:
    sidecar = Path(path).with_suffix(".metadata.json")
    if not sidecar.is_file():
        raise SystemExit(f"No metadata beside {path}")
    return json.loads(sidecar.read_text())


def main(argv: list) -> int:
    if len(argv) < 2 or argv[1] in {"-h", "--help"}:
        print(__doc__)
        return 0
    command, rest = argv[1], argv[2:]
    if command == "train":
        return command_train(rest)
    if command == "predict":
        return command_predict(rest)
    if command == "info":
        return command_info(rest)
    raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
