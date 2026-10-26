"""Public entry point for MLX Multi-Paradigm Aero-Structural MDO & Bionic Wing-Box Topology Optimization.

Integrates into `XFOIL-Designer-Public`:
  1. 6D Asymmetric Supercritical CST Airfoil Manifold + Rectified Flow Generative Prior (`AirfoilRectifiedFlowMLX`)
  2. 3-Condition Flight Envelope Evaluator (`Cruise` + `Maneuver` + `Gust`) with Dual-Fidelity
     `surrogate_mlx_latest.npz` + Native Apple Silicon `XFOIL 6.996` + `trust.py` boundary-layer audit
  3. Body-Fitted Isoparametric Q4 Wing-Box Structural Topology Optimizer (`IsoparametricWingBoxFEM`)
     under Constant Mass Budget (`m = m_ref`)
  4. Three Optimization Paradigms:
     - Paradigm 1: Local Coupled Adjoint Gradient Descent (`run_coupled_adjoint_mdo`)
     - Paradigm 2: 6D Latent GP Bayesian Optimization with Matérn-5/2 UCB/LCB (`run_bayesian_optimization_mdo`)
     - Paradigm 3: Physics-Guided Diffusion / Rectified Flow ODE (`run_guided_flow_mdo`)
  5. Arbitrary `.dat` Coordinate Hicks-Henne `mx.grad` Optimization + XFOIL `Cp(x)` Bionic Rib Synthesis
  6. 3D Finite-Wing Spanwise Strip Aero-Structural Wing-Box Coupling (`Root` / `Mid-Span` / `Tip`)
  7. Unified High-Contrast Dark-Mode Engineering Visualizers & Interactive TUI Control Panel
"""

from __future__ import annotations

import sys
from pathlib import Path

_MDO_DIR = Path(__file__).resolve().parent / "mdo"
if str(_MDO_DIR) not in sys.path:
    sys.path.insert(0, str(_MDO_DIR))


def run_showcase(argv: list[str] | None = None) -> dict:
    """Run the full 3-paradigm + XFOIL-Designer-Public aero-structural MDO showcase."""
    from .mdo import run_showcase as _showcase

    old_argv = sys.argv
    try:
        if argv is not None:
            sys.argv = ["run_showcase.py", *argv]
        return _showcase.main()
    finally:
        sys.argv = old_argv


def run_interactive_cli() -> None:
    """Launch the interactive terminal control panel for 6D BO / Guided Flow / Wing-Box MDO."""
    from .mdo import interactive_cli as _cli

    _cli.main()


def optimize_airfoil_wingbox(
    coordinates: str | Path,
    out_dir: str | Path = "xfoil_runs/mdo_outputs",
    reynolds: float = 650000.0,
    mach: float = 0.14,
    alpha_cruise: float = 3.0,
    vol_frac: float = 0.36,
    simp_iters: int = 38,
) -> dict:
    """Optimize an arbitrary `.dat` airfoil with `mlx_surrogate` + XFOIL `Cp(x)` + `IsoparametricWingBoxFEM`."""
    from .mdo.xfoil_bridge import optimize_wingbox_from_xfoil_cp
    from .mdo.viz import save_xfoil_integration_figure

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result = optimize_wingbox_from_xfoil_cp(
        dat_path=Path(coordinates),
        reynolds=reynolds,
        mach=mach,
        alpha_cruise=alpha_cruise,
        vol_frac=vol_frac,
        simp_iters=simp_iters,
        out_dir=out_path,
    )
    return result
