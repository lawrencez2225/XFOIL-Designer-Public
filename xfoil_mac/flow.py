"""XFOIL TYPE 1/2/3 with explicit reference-CL conventions."""

from __future__ import annotations
import math
from .data import positive_finite


def validate_flow(flow_type: int, reference_cl: float) -> None:
    if flow_type not in (1, 2, 3):
        raise ValueError("Flow type must be 1, 2 or 3")
    positive_finite(reference_cl, "Reference CL")


def solver_factors(config: dict) -> tuple[float, float]:
    kind, cl = config.get("flow_type", 1), config.get("reference_cl", 1.0)
    validate_flow(kind, cl)
    factor = math.sqrt(cl) if kind == 2 else cl if kind == 3 else 1.0
    values = config["re"] * factor, config["mach"] * (
        math.sqrt(cl) if kind == 2 else 1.0
    )
    if not all(math.isfinite(v) for v in values):
        raise ValueError(
            "Re/Mach reference factors overflow; use finite "
            "representable flow conditions"
        )
    return values


def actual_conditions(config: dict, cl: float) -> tuple[float, float] | None:
    kind = config.get("flow_type", 1)
    if kind == 1:
        return config["re"], config["mach"]
    if not math.isfinite(cl) or cl <= 0:
        return None
    re_factor, mach_factor = solver_factors(config)
    reynolds = re_factor / (math.sqrt(cl) if kind == 2 else cl)
    mach = mach_factor / math.sqrt(cl) if kind == 2 else mach_factor
    if not math.isfinite(reynolds) or not 0 <= mach < 1:
        return None
    return reynolds, mach


def condition_label(config: dict) -> str:
    label = f"Re={config['re']:g}, M={config['mach']:g}"
    if config.get("flow_type", 1) != 1:
        label += (
            f" at CL={config.get('reference_cl', 1):g} | TYPE "
            f"{config['flow_type']}"
        )
    return label
