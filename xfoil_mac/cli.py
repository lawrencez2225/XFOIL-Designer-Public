"""Command-line arguments and interactive menus; numerical work lives in
workflows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .comparison import write_comparison
from .data import (
    APP_VERSION,
    alpha_values,
    atomic_json,
    fingerprint,
    safe_label,
    timestamp_label,
    validate_solver_options,
)
from .runtime import (
    AppPaths,
    FlowAssumptions,
    configure_matplotlib_cache,
    die,
    discover_app,
    resolve_flow_inputs,
    start_xquartz,
    xfoil_env,
)
from .workflows import run_batch, run_polar


def prompt_float_blank(prompt: str) -> float | None:
    value = input(prompt).strip()
    return None if value == "" else float(value)


def prompt_flow_assumptions(
    defaults: FlowAssumptions | None = None,
) -> FlowAssumptions:
    if defaults is None:
        from .storage.catalog import presets

        last_flow = presets(discover_app()).get("last", {}).get("flow", {})
        defaults = (
            FlowAssumptions(**last_flow)
            if last_flow
            else FlowAssumptions.from_env()
        )
    print(
        "Physical conditions for Re/Mach conversion (Enter keeps each "
        "default):"
    )
    values = {}
    for field, label in (
        ("chord_m", "Chord length (m)"),
        ("air_density", "Air density (kg/m^3)"),
        ("air_viscosity", "Dynamic viscosity (Pa*s)"),
        ("sound_speed", "Speed of sound (m/s)"),
    ):
        default = getattr(defaults, field)
        text = input(f"  {label} [{default:g}]: ").strip()
        values[field] = float(text) if text else default
    result = FlowAssumptions(**values)
    from .storage.catalog import presets, save_last

    app = discover_app()
    last = presets(app).get("last", {})
    save_last(app, dict(last, flow=vars(result)))
    return result


def flow_arguments(assumptions: FlowAssumptions) -> list[str]:
    return [
        value
        for field, flag in (
            ("chord_m", "--chord"),
            ("air_density", "--air-density"),
            ("air_viscosity", "--air-viscosity"),
            ("sound_speed", "--sound-speed"),
        )
        for value in (flag, str(getattr(assumptions, field)))
    ]


def prompt_flow_inputs(
    defaults: FlowAssumptions | None = None,
) -> tuple[float, float, FlowAssumptions]:
    assumptions = prompt_flow_assumptions(defaults)
    print("Flow input:")
    print(
        "  Enter both Re and Mach, or leave one blank to estimate it "
        "from the other."
    )
    print(f"  Estimate assumptions: {assumptions.describe()}")
    re_value = prompt_float_blank(
        "Reynolds number Re (blank to estimate from Mach): "
    )
    mach = prompt_float_blank("Mach number (blank to estimate from Re): ")
    resolved_re, resolved_mach, _, _ = resolve_flow_inputs(
        re_value, mach, assumptions
    )
    print(f"  Using Re={resolved_re:g}, Mach={resolved_mach:.6g}")
    return resolved_re, resolved_mach, assumptions


def parse_alpha_input() -> tuple[float, float, float]:
    print("")
    print("Enter alpha as one angle or as a range:")
    print("  single angle: 4")
    print("  range:        -4 12 1")
    text = input("Alpha input [-4 12 1]: ").strip() or "-4 12 1"
    parts = text.split()
    if len(parts) == 1:
        value = float(parts[0])
        return value, value, 1.0
    if len(parts) == 3:
        return float(parts[0]), float(parts[1]), float(parts[2])
    die(
        "Alpha input must be one value, for example 4, or three values, "
        "for example -4 12 1"
    )


def prompt_analysis_options() -> dict:
    flow_type = int(
        input(
            "Flow type [1=fixed Re/M, 2=fixed lift/chord, 3=fixed "
            "lift/speed; default 1]: "
        ).strip()
        or "1"
    )
    reference_cl = (
        float(input("Reference CL for the supplied Re/M [1]: ").strip() or "1")
        if flow_type != 1
        else 1.0
    )
    adaptive = (
        input("Adaptively refine peaks and turning regions? [y/N]: ")
        .strip()
        .lower()
        == "y"
    )
    return dict(
        flow_type=flow_type,
        reference_cl=reference_cl,
        adaptive_rounds=2 if adaptive else 0,
    )


def choose_airfoil_and_run(
    app: AppPaths, defaults: FlowAssumptions | None = None
) -> None:
    print("Choose airfoil input:")
    print("  1  NACA airfoil number, for example 2412")
    print("  2  Airfoil coordinate file path")
    choice = input("Selection [1/2]: ").strip()
    reynolds, mach, assumptions = prompt_flow_inputs(defaults)
    iterations_text = input("XFOIL iteration limit [200]: ").strip()
    iterations = int(iterations_text or "200")
    alpha_start, alpha_end, alpha_step = parse_alpha_input()
    advanced = prompt_analysis_options()
    advanced["assumptions"] = assumptions
    if choice == "1":
        naca = input("Enter NACA number: ").strip()
        if not naca:
            die("NACA number cannot be empty")
        run_polar(
            app,
            naca=naca,
            reynolds=reynolds,
            mach=mach,
            iterations=iterations,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            alpha_step=alpha_step,
            keep_result_open=True,
            force=True,
            **advanced,
        )
    elif choice == "2":
        airfoil_path = Path(input("Airfoil file path: ").strip().strip("'\""))
        run_polar(
            app,
            airfoil_file=airfoil_path,
            reynolds=reynolds,
            mach=mach,
            iterations=iterations,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            alpha_step=alpha_step,
            keep_result_open=True,
            force=True,
            **advanced,
        )
    else:
        die("Selection must be 1 or 2")


def menu(app: AppPaths, defaults: FlowAssumptions | None = None) -> None:
    while True:
        print("Choose calculation mode:")
        print("  1  Single airfoil")
        print("  2  Batch airfoil database folder")
        print("  3  Open interactive XFOIL")
        print("  4  Quit")
        print("  5  Compare saved results")
        print("  6  Multiple NACA airfoils / Re / Mach sweep")
        print("  7  Screen airfoil database / saved results")
        print("  8  Performance maps")
        print("  9  Inspect / normalize coordinate file")
        print(" 10  Compare experimental CSV")
        print(" 11  Offline interactive analysis explorer")
        print(" 12  AVL wing analysis")
        print(" 13  Install official Apple Silicon AVL")
        print(" 14  Local web workbench (recommended / 本地工作台)")
        print(" 15  Calculation history / 计算历史")
        choice = input("Selection [1-15]: ").strip()
        try:
            if choice == "1":
                choose_airfoil_and_run(app, defaults)
            elif choice == "2":
                folder = Path(
                    input("Airfoil database folder: ").strip().strip("\"'")
                )
                reynolds, mach, assumptions = prompt_flow_inputs(defaults)
                iterations = int(
                    input("XFOIL iteration limit [200]: ").strip() or "200"
                )
                alpha_start, alpha_end, alpha_step = parse_alpha_input()
                run_batch(
                    app,
                    folder,
                    reynolds,
                    mach,
                    iterations,
                    alpha_start,
                    alpha_end,
                    alpha_step,
                    None,
                    False,
                    assumptions=assumptions,
                    **prompt_analysis_options(),
                )
            elif choice == "3":
                app.run_root.mkdir(parents=True, exist_ok=True)
                start_xquartz()
                subprocess.run(
                    [str(app.xfoil_bin)],
                    cwd=str(app.run_root),
                    env=xfoil_env(app),
                    check=False,
                )
            elif choice == "5":
                folder = Path(
                    input("Saved run, batch, or study folder: ")
                    .strip()
                    .strip("\"'")
                )
                target = prompt_float_blank("Target CL (blank to skip): ")
                destination = app.run_root / f"comparison_{timestamp_label()}"
                print(
                    f"Comparison: "
                    f"{write_comparison([folder], destination, target)}"
                )
            elif choice == "6":
                assumptions = prompt_flow_assumptions(defaults)
                nacas = input("NACA numbers [0012 2412 4412]: ").split() or [
                    "0012",
                    "2412",
                    "4412",
                ]
                re_text = input(
                    "Re values [1000000; type auto to calculate from Mach]: "
                ).strip()
                reynolds = (
                    []
                    if re_text.lower() == "auto"
                    else re_text.split() or ["1000000"]
                )
                machs = input(
                    "Mach values (blank: calculate from each Re): "
                ).split()
                if not reynolds and not machs:
                    raise ValueError(
                        "Enter Mach values when Re is set to auto"
                    )
                alphas = parse_alpha_input()
                target = input("Target CL (blank to skip): ").strip()
                advanced = prompt_analysis_options()
                arguments = [
                    "--polar",
                    "--nacas",
                    *nacas,
                    "--aseq",
                    *map(str, alphas),
                    "--headless",
                ]
                if reynolds:
                    arguments += ["--re-list", *reynolds]
                if machs:
                    arguments += ["--mach-list", *machs]
                arguments += flow_arguments(assumptions)
                arguments += [
                    "--flow-type",
                    str(advanced["flow_type"]),
                    "--reference-cl",
                    str(advanced["reference_cl"]),
                    "--adaptive-rounds",
                    str(advanced["adaptive_rounds"]),
                ]
                if target:
                    arguments += ["--target-cl", target]
                main(arguments)
            elif choice == "7":
                source = (
                    input("Coordinate database or saved results folder: ")
                    .strip()
                    .strip("\"'")
                )
                criteria = (
                    input(
                        "Selection criteria JSON (see "
                        "examples/selection.json): "
                    )
                    .strip()
                    .strip("\"'")
                )
                print(
                    "These physical settings apply to new coordinate "
                    "calculations; saved results keep their recorded "
                    "settings."
                )
                main(
                    [
                        "--select",
                        source,
                        "--criteria",
                        criteria,
                        *flow_arguments(prompt_flow_assumptions(defaults)),
                    ]
                )
            elif choice in {"8", "11"}:
                folder = input("Saved results folder: ").strip().strip("\"'")
                main(["--maps" if choice == "8" else "--dashboard", folder])
            elif choice == "9":
                source = input("Coordinate file: ").strip().strip("\"'")
                repair = (
                    input("Save normalized copy if geometry is valid? [y/N]: ")
                    .strip()
                    .lower()
                    == "y"
                )
                main(["--geometry", source] + (["--repair"] if repair else []))
            elif choice == "10":
                source = input("Experimental CSV: ").strip().strip("\"'")
                folder = (
                    input("Matching saved results folder: ")
                    .strip()
                    .strip("\"'")
                )
                main(["--experiment", source, "--results", folder])
            elif choice == "12":
                from .ui.wing import wing_wizard

                wing_wizard(app)
            elif choice == "13":
                main(["--install-avl"])
            elif choice == "14":
                main(["--workbench"])
            elif choice == "15":
                from .ui.commands import history_menu

                history_menu(app)
            elif choice in {"4", "q", "Q", "quit", "exit"}:
                break
            else:
                print("Selection must be 1 to 15.")
        except (ValueError, OSError, RuntimeError, SystemExit) as error:
            print(f"Calculation could not complete: {error}")
        print("")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XFOIL analysis with resumable runs and retained raw data"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--polar",
        action="store_true",
        help="Run one or more NACA airfoils or a coordinate file",
    )
    mode.add_argument(
        "--batch",
        metavar="AIRFOIL_FOLDER",
        help="Run .dat/.txt files independently",
    )
    mode.add_argument(
        "--compare",
        nargs="+",
        metavar="RESULT_FOLDER",
        help="Compare saved runs without recalculating",
    )
    mode.add_argument(
        "--interactive", "-i", action="store_true", help="Open native XFOIL"
    )
    mode.add_argument(
        "--commands", "-c", metavar="FILE", help="Run a raw XFOIL command file"
    )
    mode.add_argument(
        "--select",
        metavar="SOURCE_FOLDER",
        help="Rank saved runs or calculate and screen coordinate files",
    )
    mode.add_argument(
        "--maps",
        nargs="+",
        metavar="RESULT_FOLDER",
        help="Write observed Re/alpha heatmaps",
    )
    mode.add_argument(
        "--geometry",
        metavar="COORDINATE_FILE",
        help="Inspect geometry and preview normalization",
    )
    mode.add_argument(
        "--experiment",
        metavar="CSV",
        help="Compare reference data with --results",
    )
    mode.add_argument(
        "--dashboard",
        nargs="+",
        metavar="RESULT_FOLDER",
        help="Generate an offline interactive HTML explorer",
    )
    mode.add_argument(
        "--wing", metavar="JSON", help="Run an AVL wing/surface model"
    )
    mode.add_argument(
        "--install-avl",
        action="store_true",
        help="Install the tested official Apple Silicon AVL build",
    )
    mode.add_argument(
        "--install-airfoils",
        action="store_true",
        help=(
            "Download the UIUC airfoil coordinate database into "
            "coord_seligFmt/ (it is not redistributed with the project)"
        ),
    )
    mode.add_argument(
        "--airfoil-status",
        action="store_true",
        help="Report whether the airfoil coordinate database is installed",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--naca", default="2412")
    source.add_argument(
        "--nacas",
        nargs="+",
        help="NACA numbers for comparison, e.g. 0012 2412 4412",
    )
    source.add_argument("--airfoil")
    re_group = parser.add_mutually_exclusive_group()
    re_group.add_argument("--re", type=float, dest="reynolds")
    re_group.add_argument(
        "--re-list",
        type=float,
        nargs="+",
        help="Reynolds numbers for a Cartesian parameter sweep",
    )
    mach_group = parser.add_mutually_exclusive_group()
    mach_group.add_argument("--mach", type=float)
    mach_group.add_argument(
        "--mach-list",
        type=float,
        nargs="+",
        help="Mach numbers for a Cartesian parameter sweep",
    )
    parser.add_argument("--iter", type=int, default=200)
    parser.add_argument(
        "--aseq",
        nargs=3,
        type=float,
        metavar=("START", "END", "STEP"),
        default=(-4.0, 12.0, 1.0),
    )
    parser.add_argument("--out")
    parser.add_argument("--csv")
    parser.add_argument("--summary")
    parser.add_argument("--outdir")
    parser.add_argument("--airfoil-label")
    parser.add_argument(
        "--keep-result-open",
        action="store_true",
        help="Replay native plots after saving the calculation",
    )
    overwrite = parser.add_mutually_exclusive_group()
    overwrite.add_argument("--force", action="store_true")
    overwrite.add_argument(
        "--resume",
        action="store_true",
        help="Continue a matching saved run; requires --out or --outdir",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable XQuartz windows; still save PNG plots",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Total XFOIL seconds per case per invocation (default: 120)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Extra rounds for missing angles (0 to 5)",
    )
    parser.add_argument(
        "--retry-step",
        type=float,
        default=0.5,
        help="Initial continuation step in degrees, halved each round",
    )
    parser.add_argument(
        "--target-cl",
        type=float,
        help=(
            "Report CD at this CL, without extrapolating or crossing missing "
            "angles"
        ),
    )
    parser.add_argument("--workdir")
    parser.add_argument("--criteria", help="Selection criteria JSON")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Save a normalized copy of valid inspected geometry",
    )
    parser.add_argument(
        "--results", help="Saved result folder for experimental comparison"
    )
    parser.add_argument(
        "--re-tolerance",
        type=float,
        default=0.01,
        help="Relative Re tolerance in experiment matching",
    )
    parser.add_argument(
        "--mach-tolerance",
        type=float,
        default=0.002,
        help="Absolute Mach tolerance in experiment matching",
    )
    parser.add_argument("--avl-binary", help="Use an existing AVL executable")
    parser.add_argument("--flow-type", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--reference-cl",
        type=float,
        default=1.0,
        help="CL at which the supplied Re/Mach apply for TYPE 2/3",
    )
    parser.add_argument(
        "--chord",
        "--chord-m",
        dest="chord_m",
        type=float,
        help="Physical chord in metres for XFOIL Re/Mach conversion",
    )
    parser.add_argument(
        "--air-density", type=float, help="Air density in kg/m^3"
    )
    parser.add_argument(
        "--air-viscosity", type=float, help="Dynamic viscosity in Pa*s"
    )
    parser.add_argument(
        "--sound-speed", type=float, help="Speed of sound in m/s"
    )
    refinement = parser.add_mutually_exclusive_group()
    refinement.add_argument(
        "--adaptive",
        dest="adaptive_rounds",
        action="store_const",
        const=2,
        default=0,
        help="Refine important intervals for two rounds",
    )
    refinement.add_argument(
        "--adaptive-rounds", dest="adaptive_rounds", type=int, default=0
    )
    parser.add_argument("--adaptive-max-points", type=int, default=100)
    parser.add_argument("--adaptive-min-step", type=float, default=0.125)
    parser.add_argument("--version", action="version", version=APP_VERSION)
    from .ui.commands import add_arguments

    add_arguments(parser, mode)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    app = discover_app()
    configure_matplotlib_cache(app)
    from .ui.commands import handle, settings_from_args

    handled = handle(app, args, parser)
    if handled is not None:
        return handled
    if (args.cseq or args.cl_values) and not args.lift:
        parser.error("--cseq / --cl-values require --lift")
    overrides = {
        name: getattr(args, name)
        for name in ("chord_m", "air_density", "air_viscosity", "sound_speed")
    }
    has_flow_overrides = any(value is not None for value in overrides.values())
    if has_flow_overrides and (
        args.compare or args.interactive or args.commands or args.install_avl
    ):
        parser.error(
            "Physical flow options apply to XFOIL calculations/selection "
            "or defaults for the calculation menu"
        )
    if args.repair and not args.geometry:
        parser.error("--repair requires --geometry")
    if args.airfoil_status:
        from . import airfoils

        print(airfoils.describe(app.app_root))
        return 0
    if args.install_airfoils:
        from . import airfoils

        installed = airfoils.install_database(app.app_root, force=args.force)
        print(f"Database installed: {installed}")
        return 0
    if args.install_avl:
        from .avl import install_avl

        print(f"AVL installed: {install_avl(app.app_root)}")
        return 0
    post_mode = next(
        (
            name
            for name in (
                "select",
                "maps",
                "geometry",
                "experiment",
                "dashboard",
                "wing",
            )
            if getattr(args, name)
        ),
        None,
    )
    if post_mode:
        if has_flow_overrides and post_mode != "select":
            parser.error(
                "Physical flow options apply to XFOIL "
                "calculations/selection; AVL geometry and Mach come from "
                "its model JSON"
            )
        if args.out or args.csv or args.summary:
            parser.error("Analysis tools use --outdir for all outputs")
        if args.resume and post_mode not in {"select", "wing"}:
            parser.error(
                "--resume applies to calculations, database selection "
                "and AVL runs"
            )
        if args.select and args.force:
            parser.error(
                "Database selection supports --resume; use a new "
                "--outdir to start over"
            )
        if args.resume and not args.outdir:
            parser.error("--resume requires an explicit --outdir")
        destination = (
            Path(args.outdir).expanduser().resolve()
            if args.outdir
            else app.run_root / f"{post_mode}_{timestamp_label()}"
        )
        if args.select:
            if not args.criteria:
                parser.error("--select requires --criteria")
            from .selection import load_criteria, select_source

            result = select_source(
                app,
                Path(args.select),
                load_criteria(Path(args.criteria).expanduser()),
                destination,
                args.resume,
                assumptions=FlowAssumptions.from_env(**overrides),
            )
        elif args.maps:
            from .maps import write_maps

            result = write_maps([Path(p) for p in args.maps], destination)
        elif args.geometry:
            from .geometry import inspect_file

            result = inspect_file(
                Path(args.geometry), destination, args.repair
            )
        elif args.experiment:
            if not args.results:
                parser.error("--experiment requires --results")
            from .experiments import compare_experiment

            result = compare_experiment(
                Path(args.results),
                Path(args.experiment).expanduser(),
                destination,
                args.re_tolerance,
                args.mach_tolerance,
            )
        elif args.dashboard:
            from .dashboard import write_dashboard

            result = write_dashboard(
                [Path(p) for p in args.dashboard],
                destination / "explorer.html",
            )
        else:
            from .avl import find_avl, run_wing

            binary = find_avl(
                app.app_root,
                Path(args.avl_binary) if args.avl_binary else None,
            )
            result = run_wing(
                Path(args.wing),
                destination,
                binary,
                args.timeout,
                args.resume,
                args.force,
            )
        print(f"{post_mode}: {result}")
        if args.wing:
            return 0 if json.loads(result.read_text())["status"] == "ok" else 1
        if args.geometry:
            return 0 if json.loads(result.read_text())["valid"] else 1
        if args.experiment:
            report = json.loads(result.read_text())
            return 0 if report["matched"] == report["measurements"] else 1
        return 0
    alpha_start, alpha_end, alpha_step = args.aseq
    if args.compare:
        destination = (
            Path(args.outdir)
            if args.outdir
            else app.run_root / f"comparison_{timestamp_label()}"
        )
        result = write_comparison(
            [Path(p) for p in args.compare], destination, args.target_cl
        )
        print(f"Comparison: {result}")
        return 0
    if args.headless and args.keep_result_open:
        parser.error("--headless cannot be combined with --keep-result-open")
    if args.out and args.outdir:
        parser.error("Use only one of --out and --outdir")
    if args.polar or args.batch:
        assumptions = FlowAssumptions.from_env(**overrides)
        validate_solver_options(
            args.iter, args.timeout, args.retries, args.retry_step
        )
        alpha_values(*args.aseq)
        if args.target_cl is not None and not math.isfinite(args.target_cl):
            parser.error("--target-cl must be finite")
        if args.resume and not (args.out or args.outdir):
            parser.error("--resume requires --out or --outdir")
        if args.speed is not None:
            if (
                args.reynolds is not None
                or args.mach is not None
                or args.re_list
                or args.mach_list
            ):
                parser.error(
                    "--speed derives Re and Mach; omit separate Re/Mach inputs"
                )
            if not math.isfinite(args.speed) or args.speed <= 0:
                parser.error("--speed must be positive")
            args.reynolds = (
                assumptions.air_density
                * args.speed
                * assumptions.chord_m
                / assumptions.air_viscosity
            )
            args.mach = args.speed / assumptions.sound_speed
        conditions = list(
            dict.fromkeys(
                resolve_flow_inputs(reynolds, mach, assumptions)[:2]
                for reynolds in (args.re_list or [args.reynolds])
                for mach in (args.mach_list or [args.mach])
            )
        )
        nacas = list(dict.fromkeys(args.nacas or [args.naca]))
        multiple = len(conditions) > 1 or (args.polar and len(nacas) > 1)
        if multiple and any(
            (
                args.out,
                args.csv,
                args.summary,
                args.airfoil_label,
                args.keep_result_open,
            )
        ):
            parser.error(
                "Multiple cases use --outdir; --out, --csv, --summary, "
                "--airfoil-label and --keep-result-open apply to one case"
            )
        if args.batch and any(
            (args.out, args.csv, args.summary, args.nacas, args.airfoil)
        ):
            parser.error(
                "Batch input comes from the folder; use --outdir for its "
                "results"
            )
        destination = (
            Path(args.outdir).expanduser().resolve()
            if args.outdir
            else app.run_root / f"study_{timestamp_label()}"
        )
        common = dict(
            iterations=args.iter,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            alpha_step=alpha_step,
            force=args.force,
            resume=args.resume,
            timeout=args.timeout,
            retries=args.retries,
            retry_step=args.retry_step,
            target_cl=args.target_cl,
            flow_type=args.flow_type,
            reference_cl=args.reference_cl,
            adaptive_rounds=args.adaptive_rounds,
            adaptive_max_points=args.adaptive_max_points,
            adaptive_min_step=args.adaptive_min_step,
            assumptions=assumptions,
            solver_settings=settings_from_args(args),
            boundary_layer=args.boundary_layer,
        )
        result_folders = []
        incomplete = False
        if multiple:
            if (
                destination.exists()
                and any(destination.iterdir())
                and not (args.resume or args.force)
            ):
                raise ValueError(
                    f"Study directory is not empty: {destination}; use "
                    f"--resume or --force"
                )
            destination.mkdir(parents=True, exist_ok=True)
        for re_value, mach_value in conditions:
            condition_label = (
                f"Re{re_value:.10g}_M{mach_value:.10g}_"
                f"{fingerprint({'re': re_value, 'mach': mach_value})[:8]}"
            )
            if args.batch:
                folder = (
                    destination / condition_label
                    if multiple
                    else Path(args.outdir) if args.outdir else None
                )
                summary = run_batch(
                    app,
                    Path(args.batch),
                    re_value,
                    mach_value,
                    out_dir=folder,
                    headless=args.headless,
                    **common,
                )
                result_folders.append(summary.parent)
                with summary.open() as handle:
                    incomplete |= any(
                        r["status"] != "ok" for r in csv.DictReader(handle)
                    )
            else:
                for naca in nacas:
                    out = (
                        destination
                        / condition_label
                        / f"NACA{naca}"
                        / "polar.txt"
                        if multiple
                        else (
                            Path(args.out)
                            if args.out
                            else (
                                destination / "polar.txt"
                                if args.outdir
                                else None
                            )
                        )
                    )
                    if args.airfoil and multiple:
                        out = (
                            destination
                            / condition_label
                            / safe_label(Path(args.airfoil).stem)
                            / "polar.txt"
                        )
                    try:
                        polar = run_polar(
                            app,
                            naca=naca,
                            airfoil_file=(
                                Path(args.airfoil) if args.airfoil else None
                            ),
                            reynolds=re_value,
                            mach=mach_value,
                            out_file=out,
                            csv_file=Path(args.csv) if args.csv else None,
                            summary_file=(
                                Path(args.summary) if args.summary else None
                            ),
                            airfoil_label=args.airfoil_label,
                            keep_result_open=args.keep_result_open,
                            show_xfoil_geometry=not args.headless,
                            **common,
                        )
                    except (
                        ValueError,
                        OSError,
                        RuntimeError,
                        SystemExit,
                    ) as error:
                        if not multiple:
                            raise
                        incomplete = True
                        out.parent.mkdir(parents=True, exist_ok=True)
                        atomic_json(
                            out.parent / "failure.json",
                            {
                                "status": "failed",
                                "error": str(error),
                                "re": re_value,
                                "mach": mach_value,
                                "airfoil": (
                                    safe_label(Path(args.airfoil).stem)
                                    if args.airfoil
                                    else f"NACA{naca}"
                                ),
                                "input_file": (
                                    str(
                                        Path(args.airfoil)
                                        .expanduser()
                                        .resolve()
                                    )
                                    if args.airfoil
                                    else None
                                ),
                                "naca": None if args.airfoil else naca,
                            },
                        )
                        print(f"Case failed: {out.parent}: {error}")
                        result_folders.append(out.parent)
                        continue
                    (polar.parent / "failure.json").unlink(missing_ok=True)
                    result_folders.append(polar.parent)
                    saved = json.loads((polar.parent / "run.json").read_text())
                    incomplete |= saved["status"] != "ok"
        if multiple:
            atomic_json(
                destination / "study.json",
                {
                    "folders": [
                        str(p.relative_to(destination)) for p in result_folders
                    ],
                    "status": "needs_attention" if incomplete else "ok",
                },
            )
            comparison = write_comparison(
                result_folders, destination / "comparison", args.target_cl
            )
            print(f"Study comparison: {comparison}")
        return 1 if incomplete else 0
    if args.interactive:
        app.run_root.mkdir(parents=True, exist_ok=True)
        start_xquartz()
        return subprocess.run(
            [str(app.xfoil_bin)],
            cwd=str(app.run_root),
            env=xfoil_env(app),
            check=False,
        ).returncode
    if args.commands:
        command_file = Path(args.commands).expanduser().resolve()
        if not command_file.is_file():
            raise ValueError(f"Command file not found: {command_file}")
        workdir = (
            Path(args.workdir).expanduser().resolve()
            if args.workdir
            else app.run_root
        )
        workdir.mkdir(parents=True, exist_ok=True)
        with command_file.open() as handle:
            process = subprocess.run(
                [str(app.xfoil_bin)],
                cwd=str(workdir),
                env=xfoil_env(app),
                stdin=handle,
                check=False,
                timeout=args.timeout,
            )
        return process.returncode
    menu(
        app,
        FlowAssumptions.from_env(**overrides) if has_flow_overrides else None,
    )
    return 0


def run() -> int:
    """Shared exit/error policy for script and package entry points."""
    try:
        return main()
    except (
        ValueError,
        OSError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "Interrupted. Saved case data is retained; use --resume with "
            "the same settings and output path.",
            file=sys.stderr,
        )
        return 130
