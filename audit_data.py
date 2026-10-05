"""Check the archive for conditions that cannot all be true.

A solver accepts whatever pair of numbers it is handed. Reynolds number
and Mach number are independent dimensionless quantities -- Re scales
with chord and viscosity, Mach does not -- so they are only linked once
a chord and a fluid are fixed. A run that carried both in as free inputs
can therefore hold a combination no single chord and fluid could
produce, and every downstream number stays internally consistent, which
is what makes it hard to notice.

Usage:
    python audit_data.py
    python audit_data.py --root=xfoil_runs
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_ROOT = Path("xfoil_runs")
RELATIVE_TOLERANCE = 0.02
"""How far a stated Re may sit from the one its assumptions imply."""


def _assumption_sets(root: Path) -> dict:
    """Named (chord, fluid) assumptions seen in the archive."""
    found = {}
    for run in root.rglob("run.json"):
        try:
            document = json.loads(run.read_text())
        except (OSError, ValueError):
            continue
        assumptions = (document.get("config") or {}).get(
            "flow_assumptions"
        ) or {}
        chord = assumptions.get("chord_m")
        viscosity = assumptions.get("air_viscosity")
        density = assumptions.get("air_density")
        speed = assumptions.get("sound_speed")
        if not all((chord, viscosity, density, speed)):
            continue
        found.setdefault(
            (float(chord), float(density), float(viscosity), float(speed)), 0
        )
        found[
            (float(chord), float(density), float(viscosity), float(speed))
        ] += 1
    return found


def audit(root: Path) -> dict:
    relations = Counter()
    pairs = defaultdict(
        lambda: {"runs": 0, "sets": Counter(), "sources": Counter()}
    )
    unrecorded = 0
    mismatched = []
    declared = []
    for run in root.rglob("run.json"):
        try:
            document = json.loads(run.read_text())
        except (OSError, ValueError):
            continue
        config = document.get("config") or {}
        reynolds = config.get("re")
        mach = config.get("mach")
        if not reynolds or mach is None:
            continue
        relations[config.get("flow_relation") or "(not recorded)"] += 1
        assumptions = config.get("flow_assumptions") or {}
        chord = assumptions.get("chord_m")
        density = assumptions.get("air_density")
        viscosity = assumptions.get("air_viscosity")
        speed = assumptions.get("sound_speed")
        key = (round(reynolds), round(float(mach), 4))
        pairs[key]["runs"] += 1
        pairs[key]["sources"][str(run.relative_to(root)).split("/")[0]] += 1
        if not all((chord, density, viscosity, speed)):
            unrecorded += 1
            continue
        pairs[key]["sets"][
            (float(chord), float(density), float(viscosity), float(speed))
        ] += 1
        implied = density * float(mach) * speed * float(chord) / viscosity
        gap = abs(implied - float(reynolds)) / float(reynolds)
        if gap > RELATIVE_TOLERANCE:
            # A run that already records its pair as independently supplied
            # has said so on purpose. Computing a combination no single
            # chord and fluid produces is a legitimate thing to do, so it
            # is reported separately rather than counted as a fault.
            record = {
                "run": str(run.relative_to(root)),
                "re": float(reynolds),
                "mach": float(mach),
                "implied_re": implied,
                "off_by_percent": gap * 100,
            }
            if config.get("flow_relation") == "independent":
                declared.append(record)
            else:
                mismatched.append(record)
    return {
        "relations": relations,
        "pairs": pairs,
        "unrecorded": unrecorded,
        "mismatched": mismatched,
        "declared": declared,
    }


def report(document: dict) -> int:
    pairs = document["pairs"]
    print(f"=== 工况 (Re, Mach) 共 {len(pairs)} 个 ===")
    print(
        f"{'Re':>12} {'Mach':>7} {'run':>6}  {'流体假设(弦长,黏度)':<26} 来源"
    )
    print("-" * 92)
    for key in sorted(pairs, key=lambda k: (k[1], k[0])):
        entry = pairs[key]
        sets = (
            ", ".join(
                f"chord={c} mu={m}"
                for (c, _, m, _), _ in entry["sets"].most_common(2)
            )
            or "(未记录)"
        )
        source = ",".join(
            f"{s}×{n}" for s, n in entry["sources"].most_common(2)
        )
        print(
            f"{key[0]:>12,} {key[1]:>7} {entry['runs']:>6}  "
            f"{sets:<26} {source[:30]}"
        )
    print()
    print("=== Re 与 Mach 如何配对 ===")
    for name, count in document["relations"].most_common():
        note = ""
        if name == "independent":
            note = "  ← 两个独立给定，可能不自洽"
        elif name == "(not recorded)":
            note = "  ← 旧记录，无法判断"
        print(f"  {name:<24} {count:>6}{note}")
    print()
    if document["unrecorded"]:
        print(
            f"未记录流体假设的 run: {document['unrecorded']} "
            "(无法验证自洽性)"
        )
    declared = document.get("declared") or []
    if declared:
        print(f"声明为独立给定的 run: {len(declared)} (有意为之，非错误)")
        for item in declared[:5]:
            print(f"  {item['run'][:56]}")
            print(
                f"      Re={item['re']:,.0f} M={item['mach']} "
                f"→ 假设推出 Re={item['implied_re']:,.0f}"
            )
        print()
    bad = document["mismatched"]
    print(f"Re 与假设不符且未声明独立的 run: {len(bad)}")
    for item in bad[:10]:
        print(f"  {item['run'][:56]}")
        print(
            f"      Re={item['re']:,.0f} M={item['mach']} "
            f"→ 假设推出 Re={item['implied_re']:,.0f} "
            f"(差 {item['off_by_percent']:.1f}%)"
        )
    return 1 if bad else 0


def main(argv: list) -> int:
    root = DEFAULT_ROOT
    for item in argv[1:]:
        if item.startswith("--root="):
            root = Path(item.split("=", 1)[1])
    document = audit(root)
    status = report(document)
    if status:
        print()
        print(
            "有 run 的 Re 与它自己的假设不符。求解本身是有效的——XFOIL 按"
            "给定的两个数求解——但这些行描述的是一个物理上无法由所述弦长"
            "与流体同时成立的工况。"
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
