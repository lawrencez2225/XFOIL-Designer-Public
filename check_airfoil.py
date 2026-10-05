"""Predict, then check the prediction against anything already solved.

The useful question is not "what does the model say" but "how close was
it last time". This asks the surrogate for a curve and, when the same
airfoil and condition have already been solved, prints the solved values
beside it and the error between them.

Usage:
    python check_airfoil.py naca2412 --re=500000 --mach=0.10
    python check_airfoil.py my_airfoil.dat --re=800000 --mach=0.15
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_ROOT = Path("xfoil_runs")
MODEL = DEFAULT_ROOT / "models" / "surrogate_latest.joblib"
COORDINATES = Path("coord_seligFmt")


def _parse(argv: list) -> dict:
    options = {"target": None, "re": None, "mach": None, "alphas": None}
    for item in argv:
        if item.startswith("--") and "=" in item:
            name, _, value = item[2:].partition("=")
            if name in options:
                options[name] = value
        elif not item.startswith("--"):
            options["target"] = item
    return options


def _resolved(target: str) -> tuple:
    """Airfoil name and coordinate path for a target argument."""
    path = Path(target)
    if path.suffix == ".dat" and path.is_file():
        found = path
        if not found.is_absolute():
            found = Path.cwd() / found
        return path.stem, found
    candidate = COORDINATES / f"{target}.dat"
    if candidate.is_file():
        return target, candidate
    return target, None


def main(argv: list) -> int:
    from xfoil_mac.analysis import surrogate
    from xfoil_mac.storage.dataset import collect_rows

    options = _parse(argv[1:])
    if not options["target"]:
        print(__doc__)
        return 0
    if options["re"] is None or options["mach"] is None:
        raise SystemExit("Give --re=... and --mach=...")
    re_value = float(options["re"])
    mach_value = float(options["mach"])
    name, coordinates = _resolved(options["target"])
    if coordinates is None:
        raise SystemExit(f"No coordinates found for {name}")

    alphas = (
        [float(v) for v in options["alphas"].replace(",", " ").split()]
        if options["alphas"]
        else [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]
    )

    if not MODEL.is_file():
        raise SystemExit(
            "No trained model. Run: python surrogate_tool.py train"
        )
    loaded = surrogate.load(MODEL)
    print(f"翼型     {name}  ({coordinates})")
    print(f"工况     Re={re_value:,.0f}  Mach={mach_value}")
    print(
        f"模型     {loaded['metadata'].get('feature_count', '?')} 个特征, "
        f"{loaded['metadata'].get('training_rows', '?')} 行训练"
    )
    print()
    predictions = surrogate.predict(
        loaded,
        airfoil=name,
        coordinates=coordinates,
        re=re_value,
        mach=mach_value,
        alphas=alphas,
    )

    # Anything already solved for this airfoil and condition.
    rows, _ = collect_rows(DEFAULT_ROOT)
    solved = [
        row
        for row in rows
        if row.get("airfoil") == name
        and row.get("re")
        and row.get("mach")
        and abs(row["re"] - re_value) <= max(1.0, re_value * 1e-6)
        and abs(row["mach"] - mach_value) < 1e-6
        and all(row.get(k) is not None for k in ("CL", "CD", "CM"))
    ]
    by_alpha = {round(row["alpha"], 6): row for row in solved}

    header = f"{'alpha':>7} {'CL预测':>9} {'CL真值':>9} {'ΔCL':>8}"
    if by_alpha:
        header += f" {'CD预测':>10} {'CD真值':>10} {'ΔCD':>9}"
    print(header)
    print("-" * len(header))
    errors = {"CL": [], "CD": [], "CM": []}
    for entry in predictions:
        alpha = round(entry["alpha"], 6)
        truth = by_alpha.get(alpha)
        line = f"{entry['alpha']:>7.2f} {entry['CL']:>9.4f}"
        if truth:
            delta = entry["CL"] - truth["CL"]
            errors["CL"].append(abs(delta))
            errors["CD"].append(abs(entry["CD"] - truth["CD"]))
            errors["CM"].append(abs(entry["CM"] - truth["CM"]))
            line += f" {truth['CL']:>9.4f} {delta:>8.4f}"
            line += (
                f" {entry['CD']:>10.5f} {truth['CD']:>10.5f}"
                f" {entry['CD'] - truth['CD']:>9.5f}"
            )
        else:
            line += f" {'—':>9} {'—':>8}"
            if by_alpha:
                line += f" {'—':>10} {'—':>10} {'—':>9}"
        if entry["extrapolates"]:
            line += "   外推"
        print(line)

    print()
    if errors["CL"]:
        print(f"与 {len(errors['CL'])} 个已求解点对比:")
        for key in ("CL", "CD", "CM"):
            if errors[key]:
                mean = sum(errors[key]) / len(errors[key])
                print(
                    f"  {key}  平均绝对误差 {mean:.5f}   "
                    f"最大 {max(errors[key]):.5f}"
                )
        print()
        print("参考量级（预留测试集）：CL ≈ 0.02、CD ≈ 0.001、CM ≈ 0.004")
    else:
        print("该翼型在此工况下还没有求解记录，无法对照。")
        print("在工作台里对同一翼型和工况跑一次，再执行本命令即可对比。")
    outside = sum(1 for e in predictions if e["extrapolates"])
    if outside:
        print(f"\n注意: {outside}/{len(predictions)} 行是外推，不应直接采信。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
