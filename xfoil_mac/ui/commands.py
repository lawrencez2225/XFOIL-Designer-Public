"""Keep new workbench commands separate from the original compatibility CLI."""

import json
from pathlib import Path
from .service import run_job, new_job_root
from ..data import cl_values
from ..runtime import FlowAssumptions


def add_arguments(parser, mode):
    mode.add_argument(
        "--workbench", action="store_true", help="Open the local web workbench"
    )
    mode.add_argument(
        "--history", action="store_true", help="List saved calculations"
    )
    mode.add_argument(
        "--wing-wizard",
        action="store_true",
        help="Create or edit a wing interactively",
    )
    mode.add_argument(
        "--lift", action="store_true", help="Solve specified CL directly"
    )
    mode.add_argument(
        "--panel-study",
        type=int,
        nargs="+",
        help="Compare panel counts, e.g. 120 240 360",
    )
    mode.add_argument(
        "--check-directions",
        metavar="RUN",
        help="Independently approach key angles from both sides",
    )
    mode.add_argument(
        "--benchmark",
        action="store_true",
        help="Compare against real Ladson/NASA measurements",
    )
    mode.add_argument(
        "--design",
        metavar="JSON",
        help="Run a bounded multi-condition shape design search",
    )
    mode.add_argument(
        "--couple-wing",
        metavar="AVL_RUN",
        help="Add local XFOIL profile drag to an AVL wing run",
    )
    mode.add_argument(
        "--study-config",
        metavar="JSON",
        help="Run a saved workbench configuration",
    )
    mode.add_argument(
        "--job", metavar="FOLDER", help="Continue a saved workbench job"
    )
    parser.add_argument(
        "--cl-values",
        type=float,
        nargs="+",
        help="Direct CL targets; requires --lift",
    )
    parser.add_argument(
        "--cseq",
        type=float,
        nargs=3,
        metavar=("START", "END", "STEP"),
        help="CL target sequence; requires --lift",
    )
    parser.add_argument(
        "--check-angles",
        type=float,
        nargs="+",
        help="Angles for a direction check",
    )
    parser.add_argument(
        "--speed",
        type=float,
        help="Physical airspeed in m/s; derives both Re and Mach",
    )
    parser.add_argument("--ncrit", type=float)
    parser.add_argument("--xtr-top", type=float)
    parser.add_argument("--xtr-bottom", type=float)
    parser.add_argument("--panels", type=int)
    parser.add_argument("--panel-bunching", type=float)
    parser.add_argument("--te-le-ratio", type=float)
    parser.add_argument(
        "--boundary-layer",
        action="store_true",
        help="Save per-angle boundary-layer diagnostics",
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--length-unit-m",
        type=float,
        default=1.0,
        help="Metres per AVL model length unit",
    )


def settings_from_args(args):
    from ..solver_settings import SolverSettings

    values = {
        k: getattr(args, k)
        for k in (
            "ncrit",
            "xtr_top",
            "xtr_bottom",
            "panels",
            "panel_bunching",
            "te_le_ratio",
        )
        if getattr(args, k) is not None
    }
    return SolverSettings(**values) if values else None


def handle(app, args, parser):
    if args.workbench:
        from .server import serve

        serve(app, args.port, not args.no_browser)
        return 0
    if args.history:
        from ..storage.catalog import history, write_index

        for i, r in enumerate(history(app), 1):
            print(
                f"{i:3d}  {r['status']:18} {r['title']}\n     "
                f"{app.run_root/r['id']}"
            )
        print(f"结果首页：{write_index(app)}")
        return 0
    if args.wing_wizard:
        from .wing import wing_wizard

        wing_wizard(app)
        return 0
    if args.job:
        root = Path(args.job).expanduser().resolve()
        spec = json.loads((root / "job.json").read_text())["spec"]
        print(run_job(app, spec, root, args.resume))
        return 0
    mode = next(
        (
            m
            for m in (
                "lift",
                "panel_study",
                "check_directions",
                "benchmark",
                "design",
                "couple_wing",
                "study_config",
            )
            if getattr(args, m)
        ),
        None,
    )
    if not mode:
        return None
    if args.force:
        parser.error(
            "New studies preserve previous runs; use a new --outdir or "
            "--resume"
        )
    if args.resume and not args.outdir:
        parser.error("--resume requires --outdir")
    settings = settings_from_args(args)
    flow = FlowAssumptions.from_env(
        **{
            k: getattr(args, k)
            for k in ("chord_m", "air_density", "air_viscosity", "sound_speed")
        }
    )
    spec = {
        "kind": {
            "panel_study": "panels",
            "check_directions": "directions",
            "couple_wing": "coupling",
        }.get(mode, mode),
        "source": (
            {"airfoil": str(Path(args.airfoil).expanduser().resolve())}
            if args.airfoil
            else {"naca": args.naca}
        ),
        "re": args.reynolds,
        "mach": args.mach,
        "speed": args.speed,
        "flow": vars(flow),
        "solver_settings": settings.to_dict() if settings else None,
        "aseq": list(args.aseq),
        "iterations": args.iter,
        "timeout": args.timeout,
        "retries": args.retries,
        "boundary_layer": args.boundary_layer,
    }
    if args.nacas:
        spec["nacas"] = args.nacas
    if args.re_list:
        spec["re_values"] = args.re_list
    if args.mach_list:
        spec["mach_values"] = args.mach_list
    if mode == "lift":
        if args.cseq and args.cl_values:
            parser.error("Use --cseq or --cl-values")
        spec["targets"] = (
            cl_values(*args.cseq) if args.cseq else args.cl_values or [0.5]
        )
        spec["boundary_layer"] = True
    if mode == "panel_study":
        spec["panels"] = args.panel_study
    if mode == "check_directions":
        spec.update(
            source_run=str(Path(args.check_directions).expanduser().resolve()),
            angles=args.check_angles,
        )
    if mode == "couple_wing":
        if args.speed is None:
            parser.error("--couple-wing requires --speed")
        spec.update(
            source_run=str(Path(args.couple_wing).expanduser().resolve()),
            length_unit_m=args.length_unit_m,
        )
    if mode == "benchmark":
        spec["solver_settings"] = settings.to_dict() if settings else {}
    if mode == "design":
        spec["design"] = json.loads(Path(args.design).expanduser().read_text())
    if mode == "study_config":
        spec = json.loads(Path(args.study_config).expanduser().read_text())
    root = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else new_job_root(app, spec)
    )
    print(run_job(app, spec, root, args.resume))
    return 0


def history_menu(app):
    from ..storage.catalog import history, spec_from_history, within
    import webbrowser

    records = history(app)
    for i, r in enumerate(records, 1):
        print(f"{i:3d} {r['title']} · {r['status']}")
    text = input("选择结果编号（回车返回）: ").strip()
    if not text:
        return
    index = int(text) - 1
    if not 0 <= index < len(records):
        raise ValueError("编号超出范围")
    record = records[index]
    folder = within(app.run_root, record["id"])
    action = (
        input("1 查看，2 复制设置重跑，3 继续新版任务 [1]: ").strip() or "1"
    )
    if action == "1":
        report = folder / "report.html"
        if not report.exists():
            from ..analysis.diagnostics import diagnose

            if (folder / "run.json").exists():
                diagnose(folder)
                report = folder / "diagnostics.html"
        webbrowser.open(
            report.as_uri() if report.exists() else folder.as_uri()
        )
    elif action in {"2", "3"}:
        spec = spec_from_history(folder)
        if action == "3" and not (folder / "job.json").exists():
            raise ValueError("旧版本结果请复制设置重跑；原数据保持不动")
        print(
            run_job(
                app,
                spec,
                folder if action == "3" else new_job_root(app, spec),
                action == "3",
            )
        )
