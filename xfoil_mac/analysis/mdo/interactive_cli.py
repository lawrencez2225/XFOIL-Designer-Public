"""Interactive Terminal Control Center (TUI) for 6D Aero-Structural MDO in MLX."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aero_panel import FlightCondition, WeberGlauertPanelMLX, joukowski_exact_and_panel_verification
from airfoil_manifold import train_airfoil_flow_prior
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

# ANSI Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[96m"
C_GREEN = "\033[92m"
C_AMBER = "\033[93m"
C_ROSE = "\033[91m"
C_PURPLE = "\033[95m"
C_MUTED = "\033[90m"


@dataclass
class MDOInteractiveConfig:
    preset_name: str = "标准民航超临界翼盒 (Standard Transport +2.5g)"
    # 1. Structure & Mesh
    mass_budget: float = 0.026
    nelx: int = 36
    nely: int = 18
    # 2. 3-Condition Flight Envelope
    cruise_alpha: float = 3.0
    cruise_g: float = 1.0
    pullup_alpha: float = 9.0
    pullup_g: float = 2.5
    gust_alpha: float = -4.0
    gust_g: float = -1.2
    # 3. 6D Latent GP Bayesian Optimization
    bo_init: int = 8
    bo_steps: int = 20
    bo_kappa: float = 3.6
    bo_length_scale: float = 1.65
    # 4. Physics-Guided Flow / Diffusion ODE
    flow_particles: int = 10
    flow_steps: int = 18
    flow_guidance: float = 0.85
    # 5. Local Coupled Adjoint
    adjoint_steps: int = 26
    adjoint_lr: float = 0.06
    seed: int = 108
    # 6. XFOIL-Designer-Public & 3D Wing Integration
    xfoil_reynolds: float = 3.0e6
    xfoil_target_cl: float = 0.75
    wing_span_m: float = 14.0
    wing_sweep_deg: float = 18.0
    custom_dat_path: str = ""

    def build_envelope(self) -> tuple[FlightCondition, ...]:
        return (
            FlightCondition(
                name="cruise",
                title=f"Cond A: Nominal Cruise ({self.cruise_alpha:+.1f}°, {self.cruise_g:+.1f}g)",
                alpha_deg=self.cruise_alpha,
                load_factor_g=self.cruise_g,
                weight=0.40,
            ),
            FlightCondition(
                name="maneuver_pullup",
                title=f"Cond B: High-g Pull-Up ({self.pullup_alpha:+.1f}°, {self.pullup_g:+.1f}g)",
                alpha_deg=self.pullup_alpha,
                load_factor_g=self.pullup_g,
                weight=0.40,
            ),
            FlightCondition(
                name="negative_gust",
                title=f"Cond C: Negative Gust Dive ({self.gust_alpha:+.1f}°, {self.gust_g:+.1f}g)",
                alpha_deg=self.gust_alpha,
                load_factor_g=self.gust_g,
                weight=0.20,
            ),
        )


def _prompt_float(label: str, current: float, min_v: float = -50.0, max_v: float = 500.0) -> float:
    raw = input(f"  {label} [{C_CYAN}{current}{C_RESET}] (回车保持不变): ").strip()
    if not raw:
        return current
    try:
        val = float(raw)
        return max(min_v, min(max_v, val))
    except ValueError:
        print(f"  {C_ROSE}输入格式无效，保持原值 {current}{C_RESET}")
        return current


def _prompt_int(label: str, current: int, min_v: int = 2, max_v: int = 500) -> int:
    raw = input(f"  {label} [{C_CYAN}{current}{C_RESET}] (回车保持不变): ").strip()
    if not raw:
        return current
    try:
        val = int(raw)
        return max(min_v, min(max_v, val))
    except ValueError:
        print(f"  {C_ROSE}输入格式无效，保持原值 {current}{C_RESET}")
        return current


def print_dashboard(cfg: MDOInteractiveConfig) -> None:
    dat_disp = cfg.custom_dat_path if cfg.custom_dat_path else "默认自动导出最优 .dat"
    print("\n" + "=" * 78)
    print(
        f"{C_BOLD}{C_CYAN}  MLX Lab 07 交互式控制台 — 6D 气动-结构耦合 MDO + XFOIL 6.996 3D 机翼系统{C_RESET}"
    )
    print("=" * 78)
    print(f"  当前激活预设 : {C_GREEN}{C_BOLD}{cfg.preset_name}{C_RESET}")
    print("-" * 78)
    print(
        f"  {C_AMBER}[1]{C_RESET} 一键切换飞机设计预设   : "
        f"(a) 民航超临界 +2.5g  (b) 战斗机高过载 +6.0g  (c) 超轻滑翔机 +1.8g"
    )
    print(
        f"  {C_AMBER}[2]{C_RESET} 结构质量与有限元网格   : "
        f"质量预算 Mass={C_CYAN}{cfg.mass_budget:.4f}{C_RESET} | "
        f"贴体等参网格={C_CYAN}{cfg.nelx}x{cfg.nely}{C_RESET} ({cfg.nelx * cfg.nely} 单元)"
    )
    print(
        f"  {C_AMBER}[3]{C_RESET} 飞行包线三工况 (α, g)  : "
        f"巡航({C_CYAN}{cfg.cruise_alpha:+.1f}°,{cfg.cruise_g:+.1f}g{C_RESET}) | "
        f"机动拉起({C_CYAN}{cfg.pullup_alpha:+.1f}°,{cfg.pullup_g:+.1f}g{C_RESET}) | "
        f"负突风({C_CYAN}{cfg.gust_alpha:+.1f}°,{cfg.gust_g:+.1f}g{C_RESET})"
    )
    print(
        f"  {C_AMBER}[4]{C_RESET} 6D 贝叶斯优化 (GP-BO)  : "
        f"初始点={C_CYAN}{cfg.bo_init}{C_RESET} | 迭代步={C_CYAN}{cfg.bo_steps}{C_RESET} | "
        f"探索系数 κ={C_CYAN}{cfg.bo_kappa:.2f}{C_RESET} | 核长度尺度 l={C_CYAN}{cfg.bo_length_scale:.2f}{C_RESET}"
    )
    print(
        f"  {C_AMBER}[5]{C_RESET} 物理引导流扩散 (Flow)  : "
        f"粒子群={C_CYAN}{cfg.flow_particles}{C_RESET} | ODE步={C_CYAN}{cfg.flow_steps}{C_RESET} | "
        f"伴随引导强度 η={C_CYAN}{cfg.flow_guidance:.2f}{C_RESET}"
    )
    print(
        f"  {C_AMBER}[6]{C_RESET} 局部耦合伴随 (Adjoint) : "
        f"梯度步数={C_CYAN}{cfg.adjoint_steps}{C_RESET} | 学习率 lr={C_CYAN}{cfg.adjoint_lr:.3f}{C_RESET} | "
        f"随机种子={C_CYAN}{cfg.seed}{C_RESET}"
    )
    print(
        f"  {C_AMBER}[7]{C_RESET} XFOIL 6.996 与 3D 机翼 : "
        f"Re={C_CYAN}{cfg.xfoil_reynolds:.1e}{C_RESET} | 目标CL={C_CYAN}{cfg.xfoil_target_cl:.2f}{C_RESET} | "
        f"翼展={C_CYAN}{cfg.wing_span_m:.1f}m{C_RESET} 后掠={C_CYAN}{cfg.wing_sweep_deg:.1f}°{C_RESET} | "
        f".dat={C_CYAN}{dat_disp}{C_RESET}"
    )
    print("-" * 78)
    print(
        f"  {C_GREEN}{C_BOLD}[r 或 直接回车]{C_RESET} 立即运行全流程并刷新三张大图   "
        f"{C_PURPLE}[o]{C_RESET} 打开生成的图表/报告   "
        f"{C_ROSE}[q]{C_RESET} 退出"
    )
    print("=" * 78)


def execute_from_config(cfg: MDOInteractiveConfig) -> tuple[Path, Path, Path, Path]:
    from run_showcase import verify_isoparametric_adjoint

    out_dir = Path(__file__).resolve().parents[3] / "xfoil_runs" / "mdo_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{C_CYAN}>>> [1/5] 校验 Joukowski 解析共形映射基准与贴体等参有限元伴随灵敏度...{C_RESET}")
    jouk = joukowski_exact_and_panel_verification(alpha_deg=4.0)
    fem = IsoparametricWingBoxFEM(nelx=cfg.nelx, nely=cfg.nely, mass_budget=cfg.mass_budget)
    panel = WeberGlauertPanelMLX(envelope=cfg.build_envelope())
    adj_rel_err = verify_isoparametric_adjoint(fem)
    print(
        f"  • Joukowski 精确升力 CL={jouk['cl_exact']:.4f} vs MLX 面元法 CL={jouk['cl_panel']:.4f} "
        f"(相对误差: {C_GREEN}{jouk['cl_rel_error']*100:.2f}%{C_RESET})"
    )
    print(f"  • 等参有限元形状+载荷伴随梯度相对误差 = {C_GREEN}{adj_rel_err:.2e}{C_RESET}")

    print(f"\n{C_CYAN}>>> [2/5] 训练 6D 多模态翼型流形 Rectified Flow 生成先验网络...{C_RESET}")
    flow_model, prior_samples = train_airfoil_flow_prior(n_samples=2400, iters=450)

    print(
        f"\n{C_CYAN}>>> [3/5] 在恒定结构质量预算 (Mass={cfg.mass_budget:.4f}) 下运行三大优化范式...{C_RESET}"
    )
    adj_res = run_local_adjoint_mdo(panel, fem, num_steps=cfg.adjoint_steps, lr=cfg.adjoint_lr)
    rho_ref = adj_res["rho_template"]

    bo_res = run_latent_gp_bo_mdo(
        panel,
        fem,
        rho_ref=rho_ref,
        n_init=cfg.bo_init,
        n_bo_steps=cfg.bo_steps,
        kappa_init=cfg.bo_kappa,
        length_scale=cfg.bo_length_scale,
        seed=cfg.seed,
    )
    flow_res = run_physics_guided_flow_mdo(
        flow_model,
        panel,
        fem,
        rho_ref=rho_ref,
        n_particles=cfg.flow_particles,
        ode_steps=cfg.flow_steps,
        guidance_scale=cfg.flow_guidance,
        seed=cfg.seed + 1918,
    )

    for res in (adj_res, bo_res, flow_res):
        d = res["final_design"]
        tc = float((d["yu"] - d["yl"]).max() * 100.0)
        ld = float(d["lds"][0])
        print(
            f"  • {res['method']:28s}: Loss={C_BOLD}{res['final_loss']:6.3f}{C_RESET} | "
            f"相对厚度 t/c={tc:4.1f}% | 巡航 L/D={ld:5.1f} | "
            f"三工况综合柔度={d['agg_compliance']:.4f} | 实际质量={d['structural_mass']:.4f}"
        )

    print(f"\n{C_CYAN}>>> [4/5] 运行单点巡航 vs. 三工况飞行包线鲁棒性 3x3 交叉验证...{C_RESET}")
    mp_study = run_single_vs_multipoint_study(flow_res["final_design"]["z"], panel, fem)
    print(
        f"  • 单点巡航优化设计 (仅 {cfg.cruise_g:+.1f}g) : "
        f"巡航={mp_study['sp_compliances'][0]:.4f}, "
        f"机动({cfg.pullup_g:+.1f}g)={C_ROSE}{mp_study['sp_compliances'][1]:.4f}{C_RESET}, "
        f"突风({cfg.gust_g:+.1f}g)={C_ROSE}{mp_study['sp_compliances'][2]:.4f}{C_RESET}"
    )
    print(
        f"  • 多工况鲁棒设计 (三工况联合) : "
        f"巡航={mp_study['mp_compliances'][0]:.4f}, "
        f"机动({cfg.pullup_g:+.1f}g)={C_GREEN}{mp_study['mp_compliances'][1]:.4f} (-{mp_study['maneuver_gain_pct']:.1f}%){C_RESET}, "
        f"突风({cfg.gust_g:+.1f}g)={C_GREEN}{mp_study['mp_compliances'][2]:.4f} (-{mp_study['gust_gain_pct']:.1f}%){C_RESET}"
    )

    print(f"\n{C_CYAN}>>> [5/5] 运行 XFOIL 6.996 原生粘性求解器 + trust.py 门控 + 31维 MLX 代理与 3D 机翼多展向站位优化...{C_RESET}")
    d_fl = flow_res["final_design"]
    opt_dat_path = out_dir / "xfoil_mlx_optimized_airfoil.dat"
    surr_res = evaluate_and_optimize_with_mlx_surrogate(
        panel.xc_np,
        d_fl["yu"],
        d_fl["yl"],
        alpha_deg=cfg.cruise_alpha,
        reynolds=cfg.xfoil_reynolds,
        target_cl=cfg.xfoil_target_cl,
        opt_steps=16,
        export_opt_dat=opt_dat_path,
    )
    xfoil_audit = run_native_xfoil_and_trust_audit(
        panel.xc_np,
        surr_res["yu_opt"],
        surr_res["yl_opt"],
        alphas_deg=(cfg.cruise_alpha, cfg.pullup_alpha, 14.0),
        reynolds=cfg.xfoil_reynolds,
        airfoil_name="MLX_HicksHenne_Supercritical",
    )
    c0, c1, c2 = xfoil_audit["conditions"][:3]
    print(
        f"  • [方向1 原生XFOIL+trust.py] : "
        f"+{cfg.cruise_alpha:.1f}° ({c0['trust_verdict']}, CL={c0['cl']:.3f}, CD={c0['cd']:.5f}) | "
        f"+{cfg.pullup_alpha:.1f}° ({c1['trust_verdict']}, CL={c1['cl']:.3f}) | "
        f"+14.0° ({c2['trust_verdict']}, reasons={c2['trust_reasons']})"
    )
    print(
        f"  • [方向2 MLX代理逆向修型]    : "
        f"基准 L/D={surr_res['base_ld_mean']:.2f} -> Hicks-Henne 修型后 L/D={C_GREEN}{surr_res['opt_ld_mean']:.2f}{C_RESET} | "
        f"目标 CL={cfg.xfoil_target_cl:.2f} 配平迎角 α*={surr_res['trim_alpha_deg']:.2f}°"
    )
    if cfg.custom_dat_path:
        custom_res = optimize_arbitrary_dat_airfoil_wingbox(
            cfg.custom_dat_path,
            fem=fem,
            reynolds=cfg.xfoil_reynolds,
        )
        print(
            f"  • [外部 .dat 翼型翼盒优化]   : 已加载 {Path(cfg.custom_dat_path).name} -> "
            f"三工况 p-范数柔度={C_GREEN}{custom_res['wingbox']['pnorm_compliance']:.4f}{C_RESET}"
        )

    wing_3d_res = run_3d_wing_spanwise_aero_structural_mdo(
        span_m=cfg.wing_span_m,
        sweep_deg=cfg.wing_sweep_deg,
        reynolds_root=cfg.xfoil_reynolds,
        cruise_alpha_deg=cfg.cruise_alpha,
        fem=fem,
        sim_steps=20,
    )
    st_r, st_m, st_t = wing_3d_res["stations"]
    print(
        f"  • [方向3 3D有限机翼三站位]   : "
        f"展弦比 AR={wing_3d_res['aspect_ratio']:.2f}, 3D CL={wing_3d_res['CL_3D']:.3f}, "
        f"3D L/D={wing_3d_res['LD_3D']:.1f} | "
        f"翼根/中段/翼尖柔度={st_r['pnorm_compliance']:.2f} / {st_m['pnorm_compliance']:.2f} / {st_t['pnorm_compliance']:.2f}"
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
        "adj_tc": float((d_adj["yu"] - d_adj["yl"]).max() * 100.0),
        "bo_tc": float((d_bo["yu"] - d_bo["yl"]).max() * 100.0),
        "flow_tc": float((d_fl["yu"] - d_fl["yl"]).max() * 100.0),
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
    print(f"\n{C_GREEN}{C_BOLD}✔ 已更新全部可视化图像、翼型坐标与报告：{C_RESET}")
    print(f"  1) {p1_path}")
    print(f"  2) {p2_path}")
    print(f"  3) {p3_path}")
    print(f"  4) {opt_dat_path}")
    print(f"  5) {rep_path}")
    return p1_path, p2_path, p3_path, rep_path


def run_interactive_loop(cfg: MDOInteractiveConfig | None = None) -> None:
    if cfg is None:
        cfg = MDOInteractiveConfig()

    out_dir = Path(__file__).resolve().parent / "outputs"
    p1_path = out_dir / "three_paradigms_comparison.png"
    p2_path = out_dir / "multipoint_envelope_matrix.png"
    p3_path = out_dir / "xfoil_3d_wing_integration.png"
    rep_path = out_dir / "report.html"

    while True:
        print_dashboard(cfg)
        try:
            choice = input(f"请选择编号 {C_AMBER}[1-7 / r / o / q]{C_RESET} (直接回车立即运行): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C_MUTED}已退出交互式控制台。{C_RESET}")
            break

        if choice in ("q", "quit", "exit"):
            print(f"{C_MUTED}已退出交互式控制台。{C_RESET}")
            break

        elif choice in ("", "r", "run"):
            p1_path, p2_path, p3_path, rep_path = execute_from_config(cfg)

        elif choice == "o":
            for p in (p1_path, p2_path, p3_path, rep_path):
                if p.exists():
                    subprocess.run(["open", str(p)], check=False)
            print(f"  {C_GREEN}已在 macOS 中打开生成的三张大图与 HTML 报告。{C_RESET}")

        elif choice in ("1", "1a", "1b", "1c", "a", "b", "c"):
            sub = choice[-1] if choice in ("1a", "1b", "1c", "a", "b", "c") else ""
            if not sub:
                print(
                    f"\n  {C_AMBER}选择预设方案：{C_RESET}\n"
                    "    [a] 标准民航超临界翼盒 (Mass=0.026, 巡航+1.0g, 拉起+2.5g, 突风-1.2g)\n"
                    "    [b] 高机动战斗机重载翼盒 (Mass=0.034, 巡航+1.0g, 拉起+6.0g, 突风-2.5g)\n"
                    "    [c] 超轻高展弦比滑翔/太阳能机翼 (Mass=0.019, 巡航+1.0g, 拉起+1.8g, 突风-0.8g)"
                )
                sub = input("  请输入预设代号 [a/b/c]: ").strip().lower()
            if sub == "a":
                cfg.preset_name = "标准民航超临界翼盒 (Standard Transport +2.5g)"
                cfg.mass_budget = 0.026
                cfg.cruise_alpha, cfg.cruise_g = 3.0, 1.0
                cfg.pullup_alpha, cfg.pullup_g = 9.0, 2.5
                cfg.gust_alpha, cfg.gust_g = -4.0, -1.2
            elif sub == "b":
                cfg.preset_name = "高机动战斗机重载翼盒 (Fighter High-G +6.0g)"
                cfg.mass_budget = 0.034
                cfg.cruise_alpha, cfg.cruise_g = 2.5, 1.0
                cfg.pullup_alpha, cfg.pullup_g = 12.0, 6.0
                cfg.gust_alpha, cfg.gust_g = -5.0, -2.5
            elif sub == "c":
                cfg.preset_name = "超轻高展弦比滑翔/太阳能机翼 (Ultra-Light Glider +1.8g)"
                cfg.mass_budget = 0.019
                cfg.cruise_alpha, cfg.cruise_g = 4.5, 1.0
                cfg.pullup_alpha, cfg.pullup_g = 8.0, 1.8
                cfg.gust_alpha, cfg.gust_g = -3.0, -0.8

        elif choice == "2":
            print(f"\n{C_AMBER}--- 修改 [2] 结构质量与有限元网格 ---{C_RESET}")
            cfg.mass_budget = _prompt_float("恒定结构总质量预算 Mass Budget (推荐 0.015~0.045)", cfg.mass_budget, 0.010, 0.060)
            cfg.nelx = _prompt_int("翼盒弦向网格单元数 nelx (推荐 24~48)", cfg.nelx, 12, 72)
            cfg.nely = _prompt_int("翼盒法向网格单元数 nely (推荐 12~24)", cfg.nely, 8, 36)
            cfg.preset_name = "用户自定义配置 (Custom)"

        elif choice == "3":
            print(f"\n{C_AMBER}--- 修改 [3] 飞行包线三工况 (迎角 α & 过载因子 g) ---{C_RESET}")
            cfg.cruise_alpha = _prompt_float("工况 A 巡航迎角 alpha (度)", cfg.cruise_alpha, -5.0, 15.0)
            cfg.cruise_g = _prompt_float("工况 A 巡航过载因子 (+g)", cfg.cruise_g, 0.2, 5.0)
            cfg.pullup_alpha = _prompt_float("工况 B 机动拉起迎角 alpha (度)", cfg.pullup_alpha, 2.0, 18.0)
            cfg.pullup_g = _prompt_float("工况 B 机动拉起过载因子 (+g)", cfg.pullup_g, 1.0, 9.0)
            cfg.gust_alpha = _prompt_float("工况 C 负突风俯冲迎角 alpha (度)", cfg.gust_alpha, -15.0, 0.0)
            cfg.gust_g = _prompt_float("工况 C 负突风俯冲过载因子 (-g)", cfg.gust_g, -6.0, -0.2)
            cfg.preset_name = "用户自定义飞行包线 (Custom Envelope)"

        elif choice == "4":
            print(f"\n{C_AMBER}--- 修改 [4] 6D 隐空间高斯过程贝叶斯优化 (GP-BO) ---{C_RESET}")
            cfg.bo_init = _prompt_int("初始拉丁超立方采样点数 bo_init", cfg.bo_init, 4, 40)
            cfg.bo_steps = _prompt_int("贝叶斯采集迭代步数 bo_steps", cfg.bo_steps, 5, 60)
            cfg.bo_kappa = _prompt_float("GP-LCB 探索权重系数 kappa", cfg.bo_kappa, 0.5, 10.0)
            cfg.bo_length_scale = _prompt_float("Matérn-5/2 核长度尺度 length_scale", cfg.bo_length_scale, 0.4, 5.0)
            cfg.preset_name = "用户自定义配置 (Custom)"

        elif choice == "5":
            print(f"\n{C_AMBER}--- 修改 [5] 物理伴随引导流扩散模型 (Physics-Guided Flow) ---{C_RESET}")
            cfg.flow_particles = _prompt_int("并行采样粒子群数量 flow_particles", cfg.flow_particles, 4, 40)
            cfg.flow_steps = _prompt_int("ODE 积分步数 flow_steps", cfg.flow_steps, 6, 50)
            cfg.flow_guidance = _prompt_float("物理伴随梯度注入强度 η (flow_guidance)", cfg.flow_guidance, 0.0, 3.0)
            cfg.preset_name = "用户自定义配置 (Custom)"

        elif choice == "6":
            print(f"\n{C_AMBER}--- 修改 [6] 局部耦合伴随梯度与随机种子 ---{C_RESET}")
            cfg.adjoint_steps = _prompt_int("局部伴随梯度迭代步数 adjoint_steps", cfg.adjoint_steps, 5, 80)
            cfg.adjoint_lr = _prompt_float("局部伴随 Adam 学习率 adjoint_lr", cfg.adjoint_lr, 0.005, 0.3)
            cfg.seed = _prompt_int("全局随机种子 seed", cfg.seed, 1, 99999)
            cfg.preset_name = "用户自定义配置 (Custom)"

        elif choice == "7":
            print(f"\n{C_AMBER}--- 修改 [7] XFOIL 6.996 + MLX 代理 + 3D 机翼参数 ---{C_RESET}")
            cfg.xfoil_reynolds = _prompt_float("XFOIL 粘性雷诺数 Re", cfg.xfoil_reynolds, 2.0e5, 2.0e7)
            cfg.xfoil_target_cl = _prompt_float("目标配平升力系数 Target CL", cfg.xfoil_target_cl, 0.1, 1.5)
            cfg.wing_span_m = _prompt_float("3D 机翼全展长 Span (m)", cfg.wing_span_m, 4.0, 60.0)
            cfg.wing_sweep_deg = _prompt_float("3D 机翼四分之一弦线后掠角 Sweep (deg)", cfg.wing_sweep_deg, 0.0, 45.0)
            raw_p = input(f"  外部 .dat 翼型路径 [{C_CYAN}{cfg.custom_dat_path or '留空默认'}{C_RESET}]: ").strip()
            if raw_p:
                cfg.custom_dat_path = raw_p
            cfg.preset_name = "用户自定义 XFOIL + 3D 机翼配置 (Custom XFOIL/3D)"


if __name__ == "__main__":
    run_interactive_loop()

