"""End-to-end showcase for Multi-Paradigm Body-Fitted Aero-Structural MDO."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
import numpy as np

from airfoil_manifold import train_airfoil_flow_prior
from aero_panel import WeberGlauertPanelMLX, joukowski_exact_and_panel_verification
from isoparametric_fem import IsoparametricWingBoxFEM
from mdo_optimizers import (
    run_latent_gp_bo_mdo,
    run_local_adjoint_mdo,
    run_physics_guided_flow_mdo,
    run_single_vs_multipoint_study,
)
from viz import (
    save_html_report,
    save_multipoint_and_joukowski_panel,
    save_three_paradigms_panel,
    save_xfoil_3d_wing_integration_panel,
)
from xfoil_bridge import (
    evaluate_and_optimize_with_mlx_surrogate,
    optimize_arbitrary_dat_airfoil_wingbox,
    run_3d_wing_spanwise_aero_structural_mdo,
    run_native_xfoil_and_trust_audit,
)


def verify_isoparametric_adjoint(fem: IsoparametricWingBoxFEM) -> float:
    """Verify simultaneous shape + load isoparametric FEM adjoint against central FD."""
    rho = np.full((fem.nely, fem.nelx), 0.40, dtype=np.float64)
    op = fem.make_differentiable_multipoint_compliance(rho)
    inp_np = np.full((7, fem.nelx), 0.04, dtype=np.float32)
    inp_np[0] = 0.12

    _, grad_mx = mx.value_and_grad(lambda x: op(x))(mx.array(inp_np))
    mx.eval(grad_mx)
    g_adj = float(np.array(grad_mx)[0, fem.nelx // 2])

    eps = 1e-4
    p_plus = inp_np.copy()
    p_minus = inp_np.copy()
    p_plus[0, fem.nelx // 2] += eps
    p_minus[0, fem.nelx // 2] -= eps
    fd = (float(op(mx.array(p_plus)).item()) - float(op(mx.array(p_minus)).item())) / (2.0 * eps)
    return abs(g_adj - fd) / max(abs(fd), 1e-8)


import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLX Lab 7: 6D Aero-Structural MDO + XFOIL-Designer-Public Integration"
    )
    parser.add_argument("--nelx", type=int, default=36, help="Wing-box chordwise FEM elements (default: 36)")
    parser.add_argument("--nely", type=int, default=18, help="Wing-box vertical FEM elements (default: 18)")
    parser.add_argument(
        "--mass-budget", type=float, default=0.026, help="Constant structural mass budget (default: 0.026)"
    )
    parser.add_argument("--adjoint-steps", type=int, default=26, help="Local Adjoint gradient steps (default: 26)")
    parser.add_argument("--adjoint-lr", type=float, default=0.06, help="Local Adjoint learning rate (default: 0.06)")
    parser.add_argument("--bo-init", type=int, default=8, help="6D GP-BO initial LHS sample count (default: 8)")
    parser.add_argument("--bo-steps", type=int, default=20, help="6D GP-BO acquisition iterations (default: 20)")
    parser.add_argument(
        "--bo-kappa", type=float, default=3.6, help="6D GP-LCB exploration coefficient kappa (default: 3.6)"
    )
    parser.add_argument(
        "--bo-length-scale", type=float, default=1.65, help="Matérn-5/2 kernel length scale (default: 1.65)"
    )
    parser.add_argument(
        "--flow-particles", type=int, default=10, help="Physics-Guided Flow particle swarm size (default: 10)"
    )
    parser.add_argument(
        "--flow-steps", type=int, default=18, help="Physics-Guided Flow ODE integration steps (default: 18)"
    )
    parser.add_argument(
        "--flow-guidance", type=float, default=0.85, help="Adjoint guidance scale eta in Flow ODE (default: 0.85)"
    )
    parser.add_argument("--seed", type=int, default=108, help="Random seed (default: 108)")
    parser.add_argument(
        "--dat-file",
        type=str,
        default="",
        help="Optional path to an external Selig .dat airfoil file to optimize via XFOIL 6.996 + Isoparametric FEM",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "xfoil_runs" / "mdo_outputs"),
        help="Directory to write PNG/HTML/DAT outputs (default: xfoil_runs/mdo_outputs)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch interactive terminal control panel (TUI)",
    )
    return parser.parse_args()


def main() -> dict | None:
    args = parse_args()
    if args.interactive:
        from interactive_cli import MDOInteractiveConfig, run_interactive_loop

        cfg = MDOInteractiveConfig(
            mass_budget=args.mass_budget,
            nelx=args.nelx,
            nely=args.nely,
            bo_init=args.bo_init,
            bo_steps=args.bo_steps,
            bo_kappa=args.bo_kappa,
            bo_length_scale=args.bo_length_scale,
            flow_particles=args.flow_particles,
            flow_steps=args.flow_steps,
            flow_guidance=args.flow_guidance,
            adjoint_steps=args.adjoint_steps,
            adjoint_lr=args.adjoint_lr,
            seed=args.seed,
            custom_dat_path=args.dat_file,
        )
        run_interactive_loop(cfg)
        return None

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("MLX Lab 7: Deep Aero-Structural MDO + XFOIL-Designer-Public Integration")
    print("=" * 78)

    # 1. Physics & Adjoint Accuracy Verification
    print("\n[1/5] Verifying Physics Solver & Isoparametric Adjoint Sensitivities...")
    jouk = joukowski_exact_and_panel_verification(alpha_deg=4.0)
    fem = IsoparametricWingBoxFEM(nelx=args.nelx, nely=args.nely, mass_budget=args.mass_budget)
    panel = WeberGlauertPanelMLX()
    adj_rel_err = verify_isoparametric_adjoint(fem)
    print(
        f"  • Joukowski Exact Conformal CL : {jouk['cl_exact']:.4f} vs Panel CL: {jouk['cl_panel']:.4f} "
        f"(Rel Error: {jouk['cl_rel_error']*100:.2f}%)"
    )
    print(f"  • Isoparametric Shape Adjoint  : Rel Error vs Central FD = {adj_rel_err:.2e}")

    # 2. Train Rectified Flow Generative Prior on Multimodal Airfoil Manifold
    print("\n[2/5] Training Rectified Flow Prior on 3-Basin Airfoil Manifold...")
    flow_model, prior_samples = train_airfoil_flow_prior(n_samples=2400, iters=450)

    # 3. Run Head-to-Head Comparison of 3 Optimization Paradigms
    print("\n[3/5] Running 3 Optimization Paradigms under Constant Structural Mass Budget...")
    adj_res = run_local_adjoint_mdo(panel, fem, num_steps=args.adjoint_steps, lr=args.adjoint_lr)
    rho_ref = adj_res["rho_template"]

    bo_res = run_latent_gp_bo_mdo(
        panel,
        fem,
        rho_ref=rho_ref,
        n_init=args.bo_init,
        n_bo_steps=args.bo_steps,
        kappa_init=args.bo_kappa,
        length_scale=args.bo_length_scale,
        seed=args.seed,
    )
    flow_res = run_physics_guided_flow_mdo(
        flow_model,
        panel,
        fem,
        rho_ref=rho_ref,
        n_particles=args.flow_particles,
        ode_steps=args.flow_steps,
        guidance_scale=args.flow_guidance,
        seed=args.seed + 1918,
    )

    for res in (adj_res, bo_res, flow_res):
        d = res["final_design"]
        tc = float(np.max(d["yu"] - d["yl"]) * 100.0)
        ld = float(d["lds"][0])
        print(
            f"  • {res['method']:28s}: Loss={res['final_loss']:6.3f} | "
            f"t/c={tc:4.1f}% | Cruise L/D={ld:5.1f} | "
            f"3-Cond Comp={d['agg_compliance']:.4f} | Mass={d['structural_mass']:.4f}"
        )

    # 4. Run 3x3 Single-Point vs Multi-Point Flight Envelope Cross-Evaluation
    print("\n[4/5] Running Single-Point vs Multi-Point Flight Envelope Cross-Evaluation...")
    mp_study = run_single_vs_multipoint_study(flow_res["final_design"]["z"], panel, fem)
    print(
        f"  • Single-Point (Cruise Only)   : Cond A={mp_study['sp_compliances'][0]:.4f}, "
        f"Cond B (+2.5g)={mp_study['sp_compliances'][1]:.4f}, "
        f"Cond C (-1.2g)={mp_study['sp_compliances'][2]:.4f}"
    )
    print(
        f"  • Multi-Point (3-Cond Robust)  : Cond A={mp_study['mp_compliances'][0]:.4f}, "
        f"Cond B (+2.5g)={mp_study['mp_compliances'][1]:.4f} (-{mp_study['maneuver_gain_pct']:.1f}%), "
        f"Cond C (-1.2g)={mp_study['mp_compliances'][2]:.4f} (-{mp_study['gust_gain_pct']:.1f}%)"
    )

    # 5. Run XFOIL-Designer-Public Integration (All 3 Directions)
    print("\n[5/5] Running XFOIL-Designer-Public Integration (XFOIL 6.996 + MLX Surrogate + 3D Wing)...")
    d_fl = flow_res["final_design"]
    opt_dat_path = out_dir / "xfoil_mlx_optimized_airfoil.dat"
    surr_res = evaluate_and_optimize_with_mlx_surrogate(
        panel.xc_np,
        d_fl["yu"],
        d_fl["yl"],
        alpha_deg=3.0,
        reynolds=3.0e6,
        target_cl=0.75,
        opt_steps=16,
        export_opt_dat=opt_dat_path,
    )
    xfoil_audit = run_native_xfoil_and_trust_audit(
        panel.xc_np,
        surr_res["yu_opt"],
        surr_res["yl_opt"],
        alphas_deg=(3.0, 9.0, 14.0),
        reynolds=3.0e6,
        airfoil_name="MLX_HicksHenne_Supercritical",
    )
    c0, c1, c2 = xfoil_audit["conditions"][:3]
    print(
        f"  • [Dir 1] Native XFOIL 6.996 + trust.py Gate : "
        f"+3° ({c0['trust_verdict']}, CL={c0['cl']:.3f}, CD={c0['cd']:.5f}) | "
        f"+9° ({c1['trust_verdict']}, CL={c1['cl']:.3f}) | "
        f"+14° ({c2['trust_verdict']}, reasons={c2['trust_reasons']})"
    )
    print(
        f"  • [Dir 2] 31-Feature MLX Ensemble Inverse Opt: "
        f"Base L/D={surr_res['base_ld_mean']:.2f} -> Hicks-Henne Opt L/D={surr_res['opt_ld_mean']:.2f} | "
        f"Target-CL(0.75) Trim α*={surr_res['trim_alpha_deg']:.2f}° | Saved {opt_dat_path.name}"
    )
    if args.dat_file:
        custom_res = optimize_arbitrary_dat_airfoil_wingbox(args.dat_file, fem=fem)
        print(
            f"  • [Dir 2 Custom .dat] Loaded {Path(args.dat_file).name}: "
            f"XFOIL 3-Cond p-Norm Compliance={custom_res['wingbox']['pnorm_compliance']:.4f}"
        )

    wing_3d_res = run_3d_wing_spanwise_aero_structural_mdo(fem=fem, sim_steps=20)
    st_r, st_m, st_t = wing_3d_res["stations"]
    print(
        f"  • [Dir 3] 3D Finite-Wing Spanwise MDO        : "
        f"AR={wing_3d_res['aspect_ratio']:.2f}, 3D CL={wing_3d_res['CL_3D']:.3f}, "
        f"3D L/D={wing_3d_res['LD_3D']:.1f}, Oswald e={wing_3d_res['oswald_e']:.3f} | "
        f"Station Compliances (Root/Mid/Tip)={st_r['pnorm_compliance']:.2f} / "
        f"{st_m['pnorm_compliance']:.2f} / {st_t['pnorm_compliance']:.2f}"
    )

    p1_path = out_dir / "three_paradigms_comparison.png"
    p2_path = out_dir / "multipoint_envelope_matrix.png"
    p3_path = out_dir / "xfoil_3d_wing_integration.png"
    rep_path = out_dir / "report.html"

    save_three_paradigms_panel(p1_path, panel.xc_np, adj_res, bo_res, flow_res, prior_samples)
    save_multipoint_and_joukowski_panel(p2_path, panel.xc_np, mp_study, jouk)
    save_xfoil_3d_wing_integration_panel(p3_path, panel.xc_np, xfoil_audit, surr_res, wing_3d_res)

    d_adj = adj_res["final_design"]
    d_bo = bo_res["final_design"]

    metrics = {
        "jouk_cl_exact": jouk["cl_exact"],
        "jouk_cl_panel": jouk["cl_panel"],
        "jouk_rel_err_pct": jouk["cl_rel_error"] * 100.0,
        "adjoint_rel_err": adj_rel_err,
        "mass_budget": fem.mass_budget,
        "adj_loss": adj_res["final_loss"],
        "bo_loss": bo_res["final_loss"],
        "flow_loss": flow_res["final_loss"],
        "adj_tc": float(np.max(d_adj["yu"] - d_adj["yl"]) * 100.0),
        "bo_tc": float(np.max(d_bo["yu"] - d_bo["yl"]) * 100.0),
        "flow_tc": float(np.max(d_fl["yu"] - d_fl["yl"]) * 100.0),
        "adj_ld": float(d_adj["lds"][0]),
        "bo_ld": float(d_bo["lds"][0]),
        "flow_ld": float(d_fl["lds"][0]),
        "adj_comp": d_adj["agg_compliance"],
        "bo_comp": d_bo["agg_compliance"],
        "flow_comp": d_fl["agg_compliance"],
        "adj_mass": d_adj["structural_mass"],
        "bo_mass": d_bo["structural_mass"],
        "flow_mass": d_fl["structural_mass"],
        "sp_c0": float(mp_study["sp_compliances"][0]),
        "sp_c1": float(mp_study["sp_compliances"][1]),
        "sp_c2": float(mp_study["sp_compliances"][2]),
        "mp_c0": float(mp_study["mp_compliances"][0]),
        "mp_c1": float(mp_study["mp_compliances"][1]),
        "mp_c2": float(mp_study["mp_compliances"][2]),
        "sp_norm": mp_study["sp_envelope_norm"],
        "mp_norm": mp_study["mp_envelope_norm"],
        "maneuver_gain_pct": mp_study["maneuver_gain_pct"],
        "gust_gain_pct": mp_study["gust_gain_pct"],
        "envelope_gain_pct": mp_study["envelope_gain_pct"],
        "xfoil_cl_3": float(c0["cl"]),
        "xfoil_cd_3": float(c0["cd"]),
        "xfoil_verdict_9": f"{c1['trust_verdict'].upper()} (CL={c1['cl']:.2f})",
        "xfoil_verdict_14": f"{c2['trust_verdict'].upper()} ({','.join(c2['trust_reasons']) or 'sep'})",
        "xfoil_trim_alpha": float(surr_res["trim_alpha_deg"]),
        "xfoil_base_ld": float(surr_res["base_ld_mean"]),
        "xfoil_opt_ld": float(surr_res["opt_ld_mean"]),
        "wing3d_cl": float(wing_3d_res["CL_3D"]),
        "wing3d_ld": float(wing_3d_res["LD_3D"]),
        "wing3d_e": float(wing_3d_res["oswald_e"]),
        "wing3d_c_root": float(st_r["pnorm_compliance"]),
        "wing3d_c_mid": float(st_m["pnorm_compliance"]),
        "wing3d_c_tip": float(st_t["pnorm_compliance"]),
    }
    save_html_report(rep_path, metrics)
    print(f"\nSaved visual artifacts to {out_dir}:")
    print(f"  - {p1_path}")
    print(f"  - {p2_path}")
    print(f"  - {p3_path}")
    print(f"  - {opt_dat_path}")
    print(f"  - {rep_path}")
    return metrics


if __name__ == "__main__":
    main()

