"""Pre-screen candidates with the surrogate, confirm the shortlist by solving.

Measured on this project's data, the tree surrogate ranks candidates well but
does not find the single best one. Across 1035 airfoils at one condition
the rank correlation between predicted and solved lift-to-drag was 0.88,
yet the best candidate it selected reached only 68% of the best solved
value, and widening the shortlist barely improved that. The shortlist is
therefore a filter, not an answer: it reliably removes candidates that
would not have won, and the ones it keeps still have to be solved.

With ``--backend=mlx``, screening uses the MLX deep ensemble and ranks
candidates by a conservative Lower Confidence Bound (``--lcb=1.0``) that
penalizes uncertain drag predictions in sparse regions of the shape space.

Usage:
    python prescreen.py --airfoils=coord_seligFmt --re=500000 --mach=0.10
                        --target-cl=0.8 --keep=40 [--backend=sklearn|mlx]
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_ROOT = Path("xfoil_runs")
MODEL = DEFAULT_ROOT / "models" / "surrogate_latest.joblib"
MODEL_MLX = DEFAULT_ROOT / "models" / "surrogate_mlx_latest.npz"


def _parse(argv: list) -> dict:
    options = {
        "airfoils": "coord_seligFmt",
        "re": None,
        "mach": None,
        "target-cl": None,
        "keep": "40",
        "alpha-min": "-4",
        "alpha-max": "14",
        "alpha-step": "0.5",
        "backend": "sklearn",
        "model": None,
        "lcb": "1.0",
        "csv": None,
        "solves": "0",
    }
    for item in argv:
        if item.startswith("--") and "=" in item:
            name, _, value = item[2:].partition("=")
            if name in options:
                options[name] = value
    return options


def screen(
    directory: Path,
    loaded: dict,
    *,
    re_value: float,
    mach_value: float,
    target_cl: float,
    alphas: list,
    lcb_beta: float = 0.0,
) -> list:
    """Predict a lift-to-drag at the target lift for every airfoil found."""
    from xfoil_mac.analysis import mlx_surrogate, surrogate

    is_mlx = loaded.get("kind") == "mlx_ensemble"
    module = mlx_surrogate if is_mlx else surrogate

    rows = []
    paths = sorted(Path(directory).glob("*.dat"))
    for path in paths:
        try:
            predictions = module.predict(
                loaded,
                coordinates=path,
                re=re_value,
                mach=mach_value,
                alphas=alphas,
            )
        except (ValueError, OSError):
            continue
        lift = [p["CL"] for p in predictions]
        if not lift or min(lift) > target_cl or max(lift) < target_cl:
            continue
        best = min(predictions, key=lambda p: abs(p["CL"] - target_cl))
        if abs(best["CL"] - target_cl) > 0.05 or best["CD"] <= 0:
            continue
        cd_std = float(best.get("CD_std", 0.0))
        conservative_cd = best["CD"] + lcb_beta * cd_std
        score = best["CL"] / max(conservative_cd, 1e-5)
        rows.append(
            {
                "airfoil": path.stem,
                "alpha": best["alpha"],
                "CL": best["CL"],
                "CD": best["CD"],
                "CD_std": cd_std,
                "lift_to_drag": best["CL"] / best["CD"],
                "score": score,
                "extrapolates": best["extrapolates"],
                "note": "; ".join(best["outside"]),
                "path": str(path),
            }
        )
    rows.sort(key=lambda r: -r["score"])
    return rows


def main(argv: list) -> int:
    import csv

    from xfoil_mac.analysis import mlx_surrogate, surrogate

    options = _parse(argv[1:])
    if options["re"] is None or options["mach"] is None:
        print(__doc__)
        return 0

    backend = options["backend"]
    if options["model"]:
        model_path = Path(options["model"])
        if model_path.suffix == ".npz":
            backend = "mlx"
    else:
        model_path = MODEL_MLX if backend == "mlx" else MODEL

    if not model_path.is_file():
        flag = " --backend=mlx" if backend == "mlx" else ""
        raise SystemExit(
            f"No trained model at {model_path}. "
            f"Run: python surrogate_tool.py train{flag}"
        )
    re_value = float(options["re"])
    mach_value = float(options["mach"])
    target_cl = float(options["target-cl"])
    keep = int(options["keep"])
    lcb_beta = float(options["lcb"]) if backend == "mlx" else 0.0
    alphas = [
        float(v)
        for v in __import__("numpy").arange(
            float(options["alpha-min"]),
            float(options["alpha-max"]) + 1e-9,
            float(options["alpha-step"]),
        )
    ]
    module = mlx_surrogate if backend == "mlx" else surrogate
    loaded = module.load(model_path)

    print(f"候选目录 {options['airfoils']}")
    print(
        f"工况     Re={re_value:,.0f}  Mach={mach_value}  目标 CL={target_cl}"
    )
    print(
        f"模型     {model_path.name} ({backend}), "
        f"{loaded['metadata'].get('feature_count', '?')} 个特征, "
        f"{loaded['metadata'].get('training_rows', '?')} 行训练"
    )
    print()
    rows = screen(
        Path(options["airfoils"]),
        loaded,
        re_value=re_value,
        mach_value=mach_value,
        target_cl=target_cl,
        alphas=alphas,
        lcb_beta=lcb_beta,
    )
    if not rows:
        print("没有任何候选落在目标升力范围内。")
        return 1
    trusted = [r for r in rows if not r["extrapolates"]]
    print(f"给出预测 {len(rows)} 个, 其中未外推 {len(trusted)} 个")
    shortlist = trusted[:keep] if trusted else rows[:keep]
    print(f"短名单    {len(shortlist)} 个")
    print()
    print(
        f"{'#':>3} {'翼型':<22} {'α':>6} {'CL':>7} {'CD':>9} "
        f"{'L/D':>7} {'外推':>5}"
    )
    print("-" * 68)
    for index, row in enumerate(shortlist, 1):
        print(
            f"{index:>3} {row['airfoil']:<22} {row['alpha']:>6.1f} "
            f"{row['CL']:>7.4f} {row['CD']:>9.5f} "
            f"{row['lift_to_drag']:>7.2f} "
            f"{'是' if row['extrapolates'] else '':>5}"
        )
    print()
    print("这是一份筛选结果，不是答案：仍需对短名单真实求解来定夺。")
    if options["csv"]:
        path = Path(options["csv"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(shortlist[0]), extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(shortlist)
        print(f"已写出   {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
