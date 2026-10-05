"""One structured calculation entry point shared by CLI, history and web UI."""

import itertools
import json
import math
from datetime import datetime
from pathlib import Path

from ..data import alpha_values, atomic_json, timestamp_label
from ..runtime import FlowAssumptions, resolve_flow_inputs
from ..solver_settings import SolverSettings
from ..storage.catalog import save_last, write_index
from ..analysis.report import report_page

KINDS = {
    "polar",
    "lift",
    "panels",
    "directions",
    "benchmark",
    "wing",
    "design",
    "coupling",
    "compare",
}


def conditions(spec):
    flow = FlowAssumptions(**spec.get("flow", {}))
    if spec.get("speed") is not None:
        speed = float(spec["speed"])
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("速度必须为正数")
        if spec.get("re") is not None or spec.get("mach") is not None:
            raise ValueError("输入速度时无需同时填写 Re 和 Mach")
        return [
            (
                flow.air_density * speed * flow.chord_m / flow.air_viscosity,
                speed / flow.sound_speed,
            )
        ]
    return list(
        dict.fromkeys(
            resolve_flow_inputs(r, m, flow)[:2]
            for r in spec.get("re_values", [spec.get("re")])
            for m in spec.get("mach_values", [spec.get("mach")])
        )
    )


def preflight(spec):
    kind = spec.get("kind")
    if kind not in KINDS:
        raise ValueError("未知计算类型")
    FlowAssumptions(**spec.get("flow", {}))
    SolverSettings(**(spec.get("solver_settings") or {}))
    if not 1 <= float(spec.get("timeout", 120)) <= 1800:
        raise ValueError("每个算例时限需在 1–1800 秒")
    count = 1
    points = 1
    if kind in {"polar", "lift", "panels"}:
        cond = conditions(spec)
        for r, m in cond:
            resolve_flow_inputs(r, m)
        source = spec.get("source", {"naca": "2412"})
        sources = spec.get("nacas", [source.get("naca", "file")])
        if not sources or len(sources) > 50:
            raise ValueError("翼型数量需在 1–50")
        count = len(cond) * len(sources)
        points = (
            len(spec.get("targets", [0.5]))
            if kind == "lift"
            else len(
                spec.get("alpha_targets")
                or alpha_values(*spec.get("aseq", [-4, 12, 1]))
            )
        )
        if kind == "panels":
            count *= len(spec.get("panels", [120, 240, 360]))
    elif kind == "design":
        design = spec.get("design", {})
        count = math.prod(
            len(v)
            for v in design.get(
                "variables", {"camber": [0.8, 1.0, 1.2]}
            ).values()
        ) * len(design.get("conditions", [{}, {}]))
    elif kind == "wing":
        points = len(spec.get("model", {}).get("cases", []))
    if count > 200 or count * points > 10000 or count < 1:
        raise ValueError(
            "单个任务最多 200 个算例、10,000 个目标点，请缩小范围"
        )
    return {
        "cases": count,
        "requested_points": count * points,
        "solver_seconds_budget": count * float(spec.get("timeout", 120)),
        "note": "这是求解时间上限，不是耗时预测；生成图表另需时间。黏性机翼修正按当地翼型/Re 自动展开，最多 100 组极曲线。",
    }


def common_options(spec):
    source = spec.get("source", {"naca": "2412"})
    return dict(
        naca=str(source.get("naca", "2412")),
        airfoil_file=(
            Path(source["airfoil"]) if source.get("airfoil") else None
        ),
        assumptions=FlowAssumptions(**spec.get("flow", {})),
        solver_settings=(
            SolverSettings(**spec["solver_settings"])
            if spec.get("solver_settings") is not None
            else None
        ),
        iterations=int(spec.get("iterations", 200)),
        timeout=float(spec.get("timeout", 120)),
        retries=int(spec.get("retries", 2)),
        boundary_layer=bool(spec.get("boundary_layer", True)),
        show_xfoil_geometry=False,
        save_pressure_vectors=bool(spec.get("save_pressure_vectors", True)),
        quiet=True,
        flow_type=int(spec.get("flow_type", 1)),
        reference_cl=float(spec.get("reference_cl", 1)),
        retry_step=float(spec.get("retry_step", 0.5)),
        target_cl=spec.get("target_cl"),
        adaptive_rounds=int(spec.get("adaptive_rounds", 0)),
    )


def execute_spec(app, spec, root, resume=False):
    """Calculation data lives below data/; the user opens the single root
    report.
    """
    preflight(spec)
    kind = spec["kind"]
    data = root / "data"
    if kind in {"polar", "panels", "lift"}:
        from ..workflows import run_polar
        from ..analysis.lift import run_lift
        from ..analysis.studies import panel_study

        options = common_options(spec)
        sources = spec.get("nacas") or [options["naca"]]
        pairs = list(itertools.product(sources, conditions(spec)))
        folders = []
        records = []
        reports = []
        for index, (naca, (reynolds, mach)) in enumerate(pairs, 1):
            out = data / f"case_{index:03d}"
            kwargs = dict(
                options, naca=str(naca), reynolds=reynolds, mach=mach
            )
            if kind == "lift":
                for key in (
                    "show_xfoil_geometry",
                    "save_pressure_vectors",
                    "quiet",
                    "flow_type",
                    "reference_cl",
                    "retry_step",
                    "target_cl",
                    "adaptive_rounds",
                ):
                    kwargs.pop(key)
                kwargs["airfoil"] = kwargs.pop("airfoil_file")
                file = run_lift(
                    app,
                    spec.get("targets", [0.5]),
                    out,
                    resume=resume and (out / "lift_run.json").exists(),
                    **kwargs,
                )
            else:
                a, b, c = spec.get("aseq", [-4, 12, 1])
                kwargs.update(alpha_start=a, alpha_end=b, alpha_step=c)
                if spec.get("alpha_targets") is not None:
                    kwargs["alpha_targets"] = spec["alpha_targets"]
                if kind == "panels":
                    file = panel_study(
                        app,
                        out,
                        spec.get("panels", [120, 240, 360]),
                        resume=resume,
                        **kwargs,
                    )
                else:
                    file = run_polar(
                        app,
                        out_file=out / "polar.txt",
                        resume=resume and (out / "run.json").exists(),
                        **kwargs,
                    )
            manifest = out / (
                "lift_run.json" if kind == "lift" else "run.json"
            )
            saved = (
                json.loads(manifest.read_text())
                if manifest.exists()
                else json.loads((out / "study.json").read_text())
            )
            folders.append(str(out.relative_to(root)))
            label = f"{naca} · Re={reynolds:g} · M={mach:g}"
            records.append(
                {
                    "算例": label,
                    "状态": saved["status"],
                    "收敛摘要": saved.get("summary", {}),
                }
            )
            reports.append(
                (label, str((out / "report.html").relative_to(root)))
            )
        if kind == "polar":
            atomic_json(root / "study.json", {"folders": folders})
            from ..dashboard import write_dashboard

            write_dashboard([root], root / "explorer.html")
            reports.insert(
                0, ("交互查看极曲线 / Cp / 失败点", "explorer.html")
            )
            if len(pairs) > 1:
                from ..comparison import write_comparison

                write_comparison(
                    [root], root / "comparison", spec.get("target_cl")
                )
                reports.append(("对比数据", "comparison/comparison.csv"))
        status_labels = {
            "ok": "已完成",
            "partial_convergence": "部分收敛",
            "no_converged_points": "未收敛",
            "see_panel_results": "查看各面板结果",
            "needs_attention": "需要检查",
            "invalid_outputs": "输出无效",
            "missing_cp": "压力分布缺失",
            "timeout": "超时",
            "interrupted": "已中断",
            "cancelled": "已取消",
            "failed": "失败",
        }
        display_records = []
        for record in records:
            display = {
                "算例": record["算例"],
                "状态": status_labels.get(record["状态"], record["状态"]),
            }
            if kind == "polar":
                summary = record["收敛摘要"]
                missing = summary.get("failed_alphas")
                display.update(
                    {
                        "已收敛": summary.get("points"),
                        "请求点数": summary.get("requested_points"),
                        "缺失攻角": (
                            "未记录"
                            if missing is None
                            else ", ".join(f"{a:g}°" for a in missing) or "无"
                        ),
                    }
                )
            display_records.append(display)
        report_page(
            root / "report.html",
            spec.get("title")
            or {
                "polar": "翼型扫描",
                "lift": "按目标 CL 求解",
                "panels": "面板敏感性",
            }[kind],
            "选择下面的算例查看图、边界层和原始数据。data 是计算证据目录，日常无需逐个打开。",
            display_records,
            reports,
        )
        return (
            "ok"
            if all(r["状态"] in {"ok", "see_panel_results"} for r in records)
            else "needs_attention"
        )
    if kind == "wing":
        from ..avl import run_wing, find_avl
        from .wing import clean_model

        model = root / "inputs/model.json"
        atomic_json(model, clean_model(spec["model"]))
        file = run_wing(
            model,
            data,
            find_avl(app.app_root),
            spec.get("timeout", 120),
            resume and (data / "wing_run.json").exists(),
        )
    elif kind == "directions":
        from ..analysis.studies import check_directions

        file = check_directions(
            app, Path(spec["source_run"]), data, spec.get("angles")
        )
    elif kind == "benchmark":
        from ..analysis.studies import benchmark

        file = benchmark(
            app,
            data,
            grit=spec.get("grit", "80 grit"),
            trip=float(spec.get("trip", 0.05)),
            max_alpha=float(spec.get("max_alpha", 10.2)),
            resume=resume,
            panels=(spec.get("solver_settings") or {}).get("panels", 240),
        )
    elif kind == "design":
        from ..analysis.design import design_search

        file = design_search(app, spec["design"], data, resume)
    elif kind == "coupling":
        from ..analysis.coupling import coupled_drag

        file = coupled_drag(
            app,
            Path(spec["source_run"]),
            data,
            speed=float(spec["speed"]),
            iterations=int(spec.get("iterations", 200)),
            retries=int(spec.get("retries", 2)),
            boundary_layer=bool(spec.get("boundary_layer", False)),
            length_unit_m=float(spec.get("length_unit_m", 1)),
            flow=FlowAssumptions(**spec.get("flow", {})),
            solver_settings=SolverSettings(
                **(spec.get("solver_settings") or {})
            ),
            alpha_range=tuple(spec.get("aseq", [-4, 14, 0.5])),
            timeout=spec.get("timeout", 120),
            resume=resume,
        )
    elif kind == "compare":
        from ..comparison import write_comparison
        from ..dashboard import write_dashboard

        folders = [Path(p) for p in spec["folders"]]
        write_comparison(folders, data)
        write_dashboard(folders, data / "explorer.html")
        report_page(
            data / "report.html",
            "结果对比",
            "按 Re、Mach、转捩与面板设置分组。缺失点保留断线。",
            [],
            [("交互对比", "explorer.html"), ("数值汇总", "comparison.csv")],
            [(p.name, p.name) for p in sorted(data.glob("*.png"))],
        )
        file = data / "report.html"
    report_page(
        root / "report.html",
        spec.get("title") or kind,
        "本次计算的图表、结论与原始数据入口。",
        [],
        [("打开详细结果", "data/report.html")],
    )
    if kind == "directions":
        points = json.loads((data / "direction_check.json").read_text())[
            "points"
        ]
        return (
            "ok"
            if all(p["status"] == "consistent" for p in points)
            else "needs_attention"
        )
    if kind == "benchmark":
        comparison = json.loads((data / "experiment_report.json").read_text())
        return (
            "ok"
            if comparison["matched"] == comparison["measurements"]
            else "needs_attention"
        )
    if file.suffix == ".json":
        return json.loads(file.read_text()).get("status", "ok")
    return "ok"


def run_job(app, spec, root, resume=False):
    preflight(spec)
    root = Path(root).expanduser().resolve()
    job = root / "job.json"
    if root.exists() and any(root.iterdir()) and not (resume and job.exists()):
        raise ValueError("请使用新目录；已有任务通过历史列表继续计算")
    root.mkdir(parents=True, exist_ok=True)
    old = json.loads(job.read_text()) if job.exists() else {}
    if resume and old.get("spec") != spec:
        raise ValueError("设置与原任务不一致，请另存为新任务")
    saved = {
        "spec": spec,
        "title": spec.get("title") or f"{spec['kind']} {root.name}",
        "created": old.get("created", datetime.now().astimezone().isoformat()),
        "status": "running",
    }
    atomic_json(job, saved)
    save_last(app, spec)
    try:
        saved["status"] = execute_spec(app, spec, root, resume)
    except KeyboardInterrupt:
        saved["status"] = "cancelled"
        raise
    except Exception as error:
        saved.update(status="failed", error=str(error))
        report_page(
            root / "report.html",
            saved["title"],
            "计算未完成。已有原始数据保留。可以查看任务记录，修正设置后新建计算，或用完全相同的设置继续。",
            [{"错误": str(error)}],
            [("任务设置和状态", "job.json")],
        )
        raise
    finally:
        saved["updated"] = datetime.now().astimezone().isoformat()
        atomic_json(job, saved)
        write_index(app)
    return root / "report.html"


def new_job_root(app, spec):
    return app.run_root / "projects" / f"{timestamp_label()}_{spec['kind']}"
