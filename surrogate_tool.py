"""Train the surrogate and ask it for numbers.

Commands:

    python surrogate_tool.py train [--backend=sklearn|mlx]
    python surrogate_tool.py predict --airfoil=naca2412 --re=5e5 --mach=0.1
    python surrogate_tool.py target-cl --airfoil=naca2412 --re=5e5 --mach=0.1 \
        --cl=0.75
    python surrogate_tool.py optimize --coordinates=examples/af.dat \
        --re=5e5 --mach=0.1 --alpha=4.0 --out=optimized.dat

``predict`` prints one row per angle and marks any row whose inputs fall
outside the values the model was trained on. A tree returns a number
everywhere, so without that mark an extrapolation is indistinguishable
from an interpolation. When using the ``mlx`` backend, predictions also
report ensemble uncertainty (``CD_std``) and support ``mx.grad`` inverse
queries (``target-cl`` and ``optimize``).
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
LATEST_MLX = "surrogate_mlx_latest.npz"


def _latest_model(backend: str = "sklearn") -> Path:
    pattern = (
        "surrogate_mlx_*.npz" if backend == "mlx" else "surrogate_*.joblib"
    )
    candidates = [
        p
        for p in sorted((DEFAULT_ROOT / MODEL_DIR).glob(pattern))
        if "latest" not in p.name
    ]
    if not candidates:
        latest = DEFAULT_ROOT / MODEL_DIR / (
            LATEST_MLX if backend == "mlx" else LATEST
        )
        if latest.is_file():
            return latest
        flag = " --backend=mlx" if backend == "mlx" else ""
        raise SystemExit(
            f"No trained surrogate. Run: python surrogate_tool.py train{flag}"
        )
    return candidates[-1]


def _resolve_backend(model_path: Path | None, backend_flag: str | None) -> str:
    if backend_flag in {"mlx", "sklearn"}:
        return backend_flag
    if model_path is not None and model_path.suffix == ".npz":
        return "mlx"
    return "sklearn"


def command_train(argv: list) -> int:
    from xfoil_mac.analysis import mlx_surrogate, surrogate
    from xfoil_mac.data import timestamp_label
    from xfoil_mac.storage.dataset import collect_rows

    root = DEFAULT_ROOT
    backend = "sklearn"
    shape_root = None
    for flag in argv:
        if flag.startswith("--root="):
            root = Path(flag.split("=", 1)[1])
        elif flag.startswith("--backend="):
            backend = flag.split("=", 1)[1]
        elif flag.startswith("--shape-root="):
            shape_root = Path(flag.split("=", 1)[1])

    print(f"读取 {root} ...")
    rows, _ = collect_rows(root)
    print(f"  {len(rows)} 行")
    report = surrogate.sufficiency(rows)
    print(f"  可训练 {report['usable_rows']} 行 / {report['runs']} run")
    if not report["fit_allowed"]:
        print("\n拒绝拟合:")
        print("  " + surrogate.refusal_reason(report))
        return 1

    if backend == "mlx":
        outcome = mlx_surrogate.fit(rows, shape_root=shape_root)
        ext = "npz"
        prefix = "surrogate_mlx_"
        latest_name = LATEST_MLX
        saver = mlx_surrogate.save
    else:
        outcome = surrogate.fit(rows, shape_root=shape_root)
        ext = "joblib"
        prefix = "surrogate_"
        latest_name = LATEST
        saver = surrogate.save

    if not outcome["fitted"]:
        print(f"\n拟合失败: {outcome['reason']}")
        return 1
    destination = (
        DEFAULT_ROOT / MODEL_DIR / f"{prefix}{timestamp_label()}.{ext}"
    )
    written = saver(outcome, destination)
    latest = DEFAULT_ROOT / MODEL_DIR / latest_name
    latest.write_bytes(Path(written["model"]).read_bytes())
    latest.with_suffix(".metadata.json").write_text(
        Path(written["metadata"]).read_text()
    )
    print(f"\n已保存: {written['model']}")
    print(f"         {written['metadata']}")
    print(f"         {latest}  (指向最新)")
    return 0


def command_predict(argv: list) -> int:
    from xfoil_mac.analysis import mlx_surrogate, surrogate

    options = {
        "airfoil": None,
        "coordinates": None,
        "re": None,
        "mach": None,
        "alphas": None,
        "model": None,
        "backend": None,
        "shape-root": None,
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

    explicit_model = Path(options["model"]) if options["model"] else None
    backend = _resolve_backend(explicit_model, options["backend"])
    model_path = explicit_model or _latest_model(backend)
    module = mlx_surrogate if backend == "mlx" else surrogate
    shape_root = Path(options["shape-root"]) if options["shape-root"] else None

    loaded = module.load(model_path)
    metadata = loaded.get("metadata") or {}
    print(f"模型     {model_path.name} ({backend})")
    print(
        f"特征     {metadata.get('feature_count', '?')} 个"
        f"  (含形状 {metadata.get('shape_count', '?')} 个, "
        f"{metadata.get('training_rows', '?')} 行训练)"
    )
    if not loaded["version_matches"]:
        lib = "MLX" if backend == "mlx" else "scikit-learn"
        print(f"警告     {lib} 版本与训练时不同，结果可能不可靠")
    target = options["airfoil"] or options["coordinates"]
    print(f"对象     {target}")
    print(f"工况     Re={float(options['re']):,.0f}  Mach={options['mach']}")
    print()
    rows = module.predict(
        loaded,
        airfoil=options["airfoil"],
        coordinates=(
            Path(options["coordinates"]) if options["coordinates"] else None
        ),
        re=float(options["re"]),
        mach=float(options["mach"]),
        alphas=alphas,
        shape_root=shape_root,
    )
    if backend == "mlx":
        print(
            f"{'alpha':>7} {'CL':>9} {'CD':>10} {'±CD_std':>9} "
            f"{'CM':>10}   备注"
        )
        print("-" * 72)
        for row in rows:
            note = "外推" if row["extrapolates"] else ""
            if row["outside"]:
                note = (note + "  " + "; ".join(row["outside"])).strip()
            print(
                f"{row['alpha']:>7.2f} {row['CL']:>9.4f} "
                f"{row['CD']:>10.5f} {row['CD_std']:>9.5f} "
                f"{row['CM']:>10.5f}   {note}"
            )
    else:
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
        fields = ["alpha", "CL", "CD", "CM", "extrapolates", "outside"]
        if backend == "mlx":
            fields = [
                "alpha",
                "CL",
                "CD",
                "CD_std",
                "CM",
                "ood_distance",
                "extrapolates",
                "outside",
            ]
        with path.open("w", newline="") as handle:
            writer = csv_module.DictWriter(
                handle, fieldnames=fields, extrasaction="ignore"
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


def command_target_cl(argv: list) -> int:
    from xfoil_mac.analysis import mlx_surrogate

    options = {
        "airfoil": None,
        "coordinates": None,
        "re": None,
        "mach": None,
        "cl": None,
        "model": None,
        "shape-root": None,
    }
    for flag in argv:
        if flag.startswith("--") and "=" in flag:
            name, _, value = flag[2:].partition("=")
            if name in options:
                options[name] = value
    if not options["airfoil"] and not options["coordinates"]:
        raise SystemExit("Give --airfoil=NAME or --coordinates=PATH")
    if (
        options["re"] is None
        or options["mach"] is None
        or options["cl"] is None
    ):
        raise SystemExit("Give --re=..., --mach=... and --cl=...")

    model_path = (
        Path(options["model"]) if options["model"] else _latest_model("mlx")
    )
    shape_root = Path(options["shape-root"]) if options["shape-root"] else None
    loaded = mlx_surrogate.load(model_path)
    res = mlx_surrogate.solve_alpha_for_cl(
        loaded,
        target_cl=float(options["cl"]),
        airfoil=options["airfoil"],
        coordinates=(
            Path(options["coordinates"]) if options["coordinates"] else None
        ),
        re=float(options["re"]),
        mach=float(options["mach"]),
        shape_root=shape_root,
    )
    print(
        f"目标 CL  {res['target_cl']:.4f}  -->  "
        f"反解 α* = {res['alpha']:.4f}° (dCL/dα = {res['dCL_dalpha']:.4f} /deg)"
    )
    print(
        f"预测气动 CL={res['CL']:.4f}  CD={res['CD']:.5f} (±{res['CD_std']:.5f})"
        f"  CM={res['CM']:.5f}  L/D={res['CL'] / res['CD']:.2f}"
    )
    if res["extrapolates"]:
        print(f"注意: 输入超出训练范围 ({'; '.join(res['outside'])})")
    return 0


def command_optimize(argv: list) -> int:
    from xfoil_mac.analysis import mlx_surrogate

    options = {
        "coordinates": None,
        "out": None,
        "re": None,
        "mach": None,
        "alpha": "4.0",
        "model": None,
    }
    for flag in argv:
        if flag.startswith("--") and "=" in flag:
            name, _, value = flag[2:].partition("=")
            if name in options:
                options[name] = value
    if (
        not options["coordinates"]
        or not options["out"]
        or options["re"] is None
        or options["mach"] is None
    ):
        raise SystemExit(
            "Give --coordinates=PATH --out=PATH --re=... --mach=..."
        )
    model_path = (
        Path(options["model"]) if options["model"] else _latest_model("mlx")
    )
    loaded = mlx_surrogate.load(model_path)
    outcome = mlx_surrogate.optimize_coordinates(
        loaded,
        coordinates=Path(options["coordinates"]),
        destination=Path(options["out"]),
        re=float(options["re"]),
        mach=float(options["mach"]),
        alpha=float(options["alpha"]),
    )
    base = outcome["baseline"]
    opt = outcome["optimized"]
    print(f"基准翼型 L/D = {base['L_D']:.2f} (LCB={base['L_D_lcb']:.2f})")
    print(f"优化翼型 L/D = {opt['L_D']:.2f} (LCB={opt['L_D_lcb']:.2f})")
    print(f"已写出   {outcome['destination']}")
    return 0


def command_info(argv: list) -> int:
    backend = "sklearn"
    for flag in argv:
        if flag == "--backend=mlx":
            backend = "mlx"
    model_path = _latest_model(backend)
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
    if command == "target-cl":
        return command_target_cl(rest)
    if command == "optimize":
        return command_optimize(rest)
    if command == "info":
        return command_info(rest)
    raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
