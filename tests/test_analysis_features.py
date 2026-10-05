import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xfoil_mac import cli, workflows
from xfoil_mac.adaptive import refinement_candidates, validate_adaptive
from xfoil_mac.avl import (
    geometry_text,
    load_wing,
    parse_strips,
    parse_values,
    wing_commands,
)
from xfoil_mac.dashboard import write_dashboard
from xfoil_mac.data import atomic_json, sha256_file
from xfoil_mac.experiments import compare_experiment, load_experiment
from xfoil_mac.flow import actual_conditions
from xfoil_mac.geometry import inspect_file, inspect_points, load_coordinates
from xfoil_mac.maps import write_maps
from xfoil_mac.results import discover_runs, interpolate
from xfoil_mac.runtime import (
    FLOW_RELATIONS,
    FlowAssumptions,
    resolve_flow_inputs,
)
from xfoil_mac.selection import load_criteria, rank_runs
from tests.helpers import GEOMETRY, fake_solver

ROOT = Path(__file__).resolve().parents[1]
POINTS = [(1, 0), (0.5, 0.08), (0, 0), (0.5, -0.05), (1, 0)]


class GeometryTest(unittest.TestCase):
    def test_transformed_reversed_duplicate_contour_preserves_shape(self):
        angle = 0.3
        points = [
            (
                3 + 2 * (x * math.cos(angle) - y * math.sin(angle)),
                4 + 2 * (x * math.sin(angle) + y * math.cos(angle)),
            )
            for x, y in POINTS[::-1]
        ]
        points.insert(1, points[0])
        report, clean = inspect_points(points)
        self.assertTrue(report["valid"], report)
        self.assertTrue(report["reversed"])
        self.assertEqual(report["duplicate_count"], 1)
        self.assertAlmostEqual(report["thickness_ratio"], 0.13)
        for actual, expected in zip(clean, POINTS):
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b)

    def test_crossing_contour_is_not_repaired(self):
        report, _ = inspect_points(
            [(1, 0), (0.25, 0.1), (0.75, -0.1), (0, 0), (0.5, -0.05), (1, 0)]
        )
        self.assertFalse(report["valid"])
        self.assertTrue(report["intersections"])

    def test_preview_preserves_original_and_blocks_input_collision(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "shape.dat"
            source.write_text(GEOMETRY)
            before = source.read_bytes()
            report = inspect_file(source, root / "preview", True)
            self.assertTrue(json.loads(report.read_text())["valid"])
            self.assertEqual(source.read_bytes(), before)
            self.assertTrue((root / "preview/repaired.dat").is_file())
            with self.assertRaises(ValueError):
                inspect_file(
                    root / "preview/repaired.dat", root / "preview", True
                )

    def test_nonfinite_and_malformed_coordinates_are_not_silently_dropped(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "shape.dat"
            for text in ("nan 0", "broken coordinate"):
                source.write_text(GEOMETRY + text + "\n")
                with self.assertRaises(ValueError):
                    load_coordinates(source)


class FlowAdaptiveTest(unittest.TestCase):
    def test_menu_accepts_physical_conditions_and_preserves_blank_defaults(
        self,
    ):
        with patch("builtins.input", side_effect=["0.3", "", "2e-5", ""]):
            settings = cli.prompt_flow_assumptions(FlowAssumptions())
        self.assertEqual(settings, FlowAssumptions(0.3, 1.225, 2e-5, 340.3))

    def test_multicase_menu_allows_mach_to_follow_re_and_chord(self):
        answers = [
            "6",
            ".3",
            "",
            "",
            "",
            "0012 2412",
            "1000000",
            "",
            "0 4 1",
            "",
            "1",
            "n",
            "4",
        ]
        with (
            patch("builtins.input", side_effect=answers),
            patch.object(cli, "main") as launch,
        ):
            cli.menu(cli.discover_app())
        arguments = launch.call_args.args[0]
        self.assertNotIn("--mach-list", arguments)
        self.assertIn("--re-list", arguments)
        self.assertEqual(arguments[arguments.index("--chord") + 1], "0.3")

    def test_physical_conversion_and_independent_pair_notice(self):
        settings = FlowAssumptions(0.2, 1.1, 2e-5, 330)
        re, mach, note, relation = resolve_flow_inputs(None, 0.1, settings)
        self.assertAlmostEqual(re, 363000)
        self.assertEqual(relation, "derived_re_from_mach")
        re2, mach2, _, relation2 = resolve_flow_inputs(re, None, settings)
        self.assertAlmostEqual(mach2, mach)
        self.assertEqual(relation2, "derived_mach_from_re")
        self.assertEqual(
            resolve_flow_inputs(1e6, 0.1, settings)[:2], (1e6, 0.1)
        )
        self.assertIn(
            "supplied independently",
            resolve_flow_inputs(1e6, 0.1, settings)[2],
        )
        self.assertEqual(resolve_flow_inputs(re, mach, settings)[2], "")
        self.assertEqual(resolve_flow_inputs(1e6, 0, settings)[:2], (1e6, 0))

    def test_the_pairing_is_recorded_so_a_bad_pair_is_visible(self):
        """Re and Mach are independent numbers, so their pairing is a fact.

        XFOIL solves whatever pair it is handed without complaint, so a
        run that carried both in as free inputs can hold a combination no
        single chord and fluid could produce. Recording how the pair was
        formed is what lets that be seen afterwards.
        """
        settings = FlowAssumptions(0.2, 1.1, 2e-5, 330)
        consistent = 363000.0
        self.assertEqual(
            resolve_flow_inputs(None, 0.1, settings)[3],
            "derived_re_from_mach",
        )
        self.assertEqual(
            resolve_flow_inputs(consistent, 0.1, settings)[3],
            "both_matched",
        )
        mismatch = resolve_flow_inputs(1e6, 0.1, settings)
        self.assertEqual(mismatch[3], "independent")
        self.assertEqual(len(FLOW_RELATIONS), len(set(FLOW_RELATIONS)))
        self.assertIn(mismatch[3], FLOW_RELATIONS)

    def test_physical_input_validation_and_explicit_override_precedence(self):
        for key in ("chord_m", "air_density", "air_viscosity", "sound_speed"):
            for value in (0, -1, math.nan, math.inf):
                with (
                    self.subTest(key=key, value=value),
                    self.assertRaises(ValueError),
                ):
                    FlowAssumptions(**{key: value})
        with patch.dict("os.environ", {"XFOIL_CHORD_M": "0.7"}):
            self.assertEqual(FlowAssumptions.from_env().chord_m, 0.7)
            self.assertEqual(
                FlowAssumptions.from_env(chord_m=0.2).chord_m, 0.2
            )

    def test_type_two_and_three_reference_invariants(self):
        for kind in (1, 2, 3):
            config = {
                "re": 1e6,
                "mach": 0.1,
                "flow_type": kind,
                "reference_cl": 0.5,
            }
            self.assertEqual(actual_conditions(config, 0.5), (1e6, 0.1))
            re_value, mach = actual_conditions(config, 2.0)
            if kind == 2:
                self.assertAlmostEqual(re_value, 500000)
                self.assertAlmostEqual(mach, 0.05)
            elif kind == 3:
                self.assertAlmostEqual(re_value, 250000)
                self.assertAlmostEqual(mach, 0.1)
            else:
                self.assertEqual((re_value, mach), (1e6, 0.1))
        self.assertIsNone(actual_conditions(dict(config, flow_type=2), 0))
        self.assertIsNone(
            actual_conditions(dict(config, flow_type=2, mach=0.9), 0.01)
        )

    def test_adaptive_target_priority_cap_and_minimum_spacing(self):
        rows = [
            {"alpha": a, "CL": a * 0.1, "CD": 0.01 + (a - 2) ** 2 * 0.002}
            for a in (0, 2, 4, 6)
        ]
        chosen = refinement_candidates(
            rows, [0, 2, 4, 6], 5, 0.5, target_cl=0.5
        )
        self.assertEqual(chosen, [5.0])
        self.assertEqual(refinement_candidates(rows, [0, 2, 4, 6], 9, 1.1), [])
        with self.assertRaises(ValueError):
            validate_adaptive(2, 3, 0.1, 4)


class SavedAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        binary = self.root / "xfoil"
        binary.write_text("fake")
        self.app = workflows.AppPaths(
            self.root,
            self.root,
            binary,
            self.root,
            self.root / "runs",
            self.root / "single",
            self.root / "batch",
        )
        self.patches = [
            patch("xfoil_mac.execution.execute", side_effect=fake_solver),
            patch.object(workflows, "write_geometry_plot"),
            patch.object(
                workflows,
                "write_cp_polar_outputs",
                return_value={"result_plots": [], "pressure_vectors": []},
            ),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def run_case(self, name="case", **kwargs):
        options = dict(
            app=self.app,
            naca="2412",
            reynolds=1e6,
            mach=0.1,
            alpha_start=0,
            alpha_end=4,
            alpha_step=2,
            out_file=self.root / name / "polar.txt",
            show_xfoil_geometry=False,
            retries=0,
            quiet=True,
        )
        options.update(kwargs)
        workflows.run_polar(**options)
        return self.root / name

    def criteria(self):
        path = self.root / "criteria.json"
        path.write_text(
            json.dumps(
                {
                    "reynolds": [1e6],
                    "mach": [0.1],
                    "target_cls": [0.1, 0.3],
                    "min_thickness": 0.1,
                    "max_abs_cm": 0.03,
                }
            )
        )
        return load_criteria(path)

    def test_adaptive_manifest_keeps_fingerprint_and_original_attempt_angles(
        self,
    ):
        folder = self.run_case(
            adaptive_rounds=2,
            adaptive_max_points=8,
            adaptive_min_step=0.25,
            target_cl=0.3,
        )
        manifest = json.loads((folder / "run.json").read_text())
        self.assertEqual(manifest["config"]["alphas"], [0.0, 2.0, 4.0])
        self.assertEqual(manifest["attempts"][0]["alphas"], [0.0, 2.0, 4.0])
        self.assertGreater(len(manifest["requested_alphas"]), 3)
        self.assertLessEqual(len(manifest["requested_alphas"]), 8)
        self.assertEqual(
            manifest["summary"]["requested_points"],
            len(manifest["requested_alphas"]),
        )
        before = sha256_file(folder / "polar.txt")
        with patch(
            "xfoil_mac.execution.execute",
            side_effect=AssertionError("Unexpected recalculation"),
        ):
            self.run_case(
                adaptive_rounds=2,
                adaptive_max_points=8,
                adaptive_min_step=0.25,
                target_cl=0.3,
                resume=True,
            )
        self.assertEqual(before, sha256_file(folder / "polar.txt"))

    def test_cli_physical_options_reach_solver_and_saved_config(self):
        destination = self.root / "physical"
        settings = FlowAssumptions(0.2, 1.1, 2e-5, 330)
        with patch.object(cli, "discover_app", return_value=self.app):
            status = cli.main(
                [
                    "--polar",
                    "--re",
                    "1000000",
                    "--aseq",
                    "0",
                    "2",
                    "1",
                    "--headless",
                    "--outdir",
                    str(destination),
                    *cli.flow_arguments(settings),
                ]
            )
        self.assertEqual(status, 0)
        config = json.loads((destination / "run.json").read_text())["config"]
        self.assertEqual(config["flow_assumptions"], vars(settings))
        # This case hands in a Re and a Mach that the stated assumptions
        # already make consistent, and the run says so.
        self.assertEqual(config["flow_relation"], "both_matched")
        self.assertAlmostEqual(config["mach"], 1e6 * 2e-5 / (1.1 * 0.2 * 330))
        commands = (
            (destination / "attempts/0001/commands.txt")
            .read_text()
            .splitlines()
        )
        self.assertAlmostEqual(
            float(commands[commands.index("MACH") + 1]), config["mach"]
        )

    def test_batch_inherits_this_run_physical_settings(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        (inputs / "foil.dat").write_text(GEOMETRY)
        settings = FlowAssumptions(0.2, 1.1, 2e-5, 330)
        destination = self.root / "physical_batch"
        with (
            patch.object(cli, "discover_app", return_value=self.app),
            patch.object(workflows, "write_comparison"),
        ):
            status = cli.main(
                [
                    "--batch",
                    str(inputs),
                    "--mach",
                    ".1",
                    "--aseq",
                    "0",
                    "2",
                    "1",
                    "--headless",
                    "--outdir",
                    str(destination),
                    *cli.flow_arguments(settings),
                ]
            )
        self.assertEqual(status, 0)
        config = json.loads((destination / "foil/run.json").read_text())[
            "config"
        ]
        self.assertEqual(config["flow_assumptions"], vars(settings))
        self.assertEqual(config["re"], 363000)

    def test_resume_refuses_changed_physical_settings(self):
        self.run_case(assumptions=FlowAssumptions(chord_m=0.3))
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            self.run_case(
                assumptions=FlowAssumptions(chord_m=0.4), resume=True
            )

    def test_adaptive_resume_recovers_interrupted_new_point(self):
        calls = 0

        def interrupt(binary, env, work, commands, timeout):
            nonlocal calls
            calls += 1
            result = fake_solver(binary, env, work, commands, timeout)
            return ("interrupted", -9) if calls == 2 else result

        with (
            patch("xfoil_mac.execution.execute", side_effect=interrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.run_case(adaptive_rounds=2, adaptive_max_points=8)
        folder = self.run_case(
            adaptive_rounds=2, adaptive_max_points=8, resume=True
        )
        manifest = json.loads((folder / "run.json").read_text())
        self.assertEqual(manifest["status"], "ok")
        self.assertEqual(
            manifest["summary"]["points"], len(manifest["requested_alphas"])
        )
        self.assertLessEqual(len(manifest["requested_alphas"]), 8)

    def test_interpolation_does_not_bridge_a_missing_requested_point(self):
        folder = self.run_case()
        run = discover_runs([folder])[0]
        run.rows = [r for r in run.rows if r["alpha"] != 2]
        self.assertEqual(interpolate(run, "CL", 0.1), (None, "not_bracketed"))
        self.assertEqual(interpolate(run, "alpha", 1), (None, "not_bracketed"))

    def test_selection_requires_every_condition_and_moment_limit(self):
        folder = self.run_case()
        c = self.criteria()
        result = rank_runs([folder], c, self.root / "ranking")
        report = json.loads(result.read_text())
        self.assertEqual(report["eligible"], 1)
        self.assertTrue(report["candidates"][0]["pareto"])
        c["reynolds"].append(500000)
        report = json.loads(
            rank_runs([folder], c, self.root / "ranking").read_text()
        )
        self.assertEqual(report["eligible"], 0)
        self.assertTrue(
            any(
                "missing_condition" in r
                for r in report["candidates"][0]["reasons"]
            )
        )
        c["reynolds"] = [1e6]
        c["max_abs_cm"] = 0.01
        report = json.loads(
            rank_runs([folder], c, self.root / "ranking").read_text()
        )
        self.assertTrue(
            any(
                "Pitching moment" in r
                for r in report["candidates"][0]["reasons"]
            )
        )

    def test_selection_does_not_cherry_pick_duplicate_conditions(self):
        a = self.run_case("a")
        b = self.run_case("b")
        report = json.loads(
            rank_runs(
                [a, b], self.criteria(), self.root / "ranking"
            ).read_text()
        )
        self.assertEqual(report["eligible"], 0)
        self.assertTrue(
            any(
                "duplicate_condition" in r
                for r in report["candidates"][0]["reasons"]
            )
        )

    def test_selection_keeps_failed_candidates_without_run_manifests(self):
        atomic_json(self.root / "batch.json", {"cases": ["bad"]})
        atomic_json(
            self.root / "bad.failure.json",
            {"airfoil": "bad", "error": "Invalid coordinates"},
        )
        report = json.loads(
            rank_runs(
                [self.root], self.criteria(), self.root / "ranking"
            ).read_text()
        )
        self.assertEqual(report["eligible"], 0)
        self.assertEqual(len(report["candidates"]), 1)
        self.assertIn(
            "Invalid coordinates", report["candidates"][0]["reasons"][0]
        )

    def test_selection_rejects_malformed_scan_settings_before_calculation(
        self,
    ):
        base = self.criteria()
        path = self.root / "criteria.json"
        for values in (
            {"alpha_range": [0, 4]},
            {"alpha_range": [0, 4, 0]},
            {"timeout": "slow"},
            {"adaptive_rounds": 9},
            {"iterations": 0},
        ):
            with self.subTest(values=values):
                atomic_json(path, dict(base, **values))
                with self.assertRaises(ValueError):
                    load_criteria(path)

    def test_selection_rejects_mixed_solver_settings_across_conditions(self):
        a = self.run_case("a")
        b = self.run_case("b", reynolds=500000, iterations=50)
        criteria = self.criteria()
        criteria["reynolds"].append(500000)
        report = json.loads(
            rank_runs([a, b], criteria, self.root / "ranking").read_text()
        )
        self.assertEqual(report["eligible"], 0)
        self.assertTrue(
            any(
                "Mixed solver" in r for r in report["candidates"][0]["reasons"]
            )
        )

    def test_failed_batch_manifest_does_not_reappear_in_postprocessing(self):
        self.run_case("a")
        self.run_case("b", naca="0012")
        atomic_json(self.root / "batch.json", {"cases": ["a", "b"]})
        atomic_json(self.root / "a.failure.json", {"status": "failed"})
        self.assertEqual(
            [r.airfoil for r in discover_runs([self.root])], ["NACA0012"]
        )

    def test_maps_reject_duplicate_cells_and_preserve_missing_status(self):
        folder = self.run_case()
        discover_runs([folder])[0]
        polar = folder / "polar.txt"
        polar.write_text(
            "\n".join(
                line
                for line in polar.read_text().splitlines()
                if not line.startswith("2 ")
            )
            + "\n"
        )
        report = write_maps([folder], self.root / "maps")
        with (report.parent / "performance_map.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            [r["status"] for r in rows], ["converged", "missing", "converged"]
        )
        duplicate = self.run_case("duplicate")
        with self.assertRaises(ValueError):
            write_maps([folder, duplicate], self.root / "other_maps")

    def test_experiment_conditions_known_error_and_no_extrapolation(
        self,
    ):
        folder = self.run_case()
        source = self.root / "experiment.csv"
        source.write_text(
            "alpha,CL,CD,CM,Re,Mach\n1,.11,.012,-.03,1000000,.1\n"
            "5,.5,.01,-.02,1000000,.1\n1,.1,.01,-.02,500000,.1\n"
        )
        report = json.loads(
            compare_experiment(
                folder, source, self.root / "experiment"
            ).read_text()
        )
        self.assertEqual(report["matched"], 1)
        self.assertAlmostEqual(report["metrics"]["CL"]["bias"], -0.01)
        self.assertAlmostEqual(report["metrics"]["CD"]["rmse"], 0.002)
        self.assertAlmostEqual(report["metrics"]["CM"]["mae"], 0.01)

    def test_experiment_missing_flow_metadata_fails(self):
        source = self.root / "bad.csv"
        source.write_text("alpha,CL,CD\n1,.1,.01\n")
        with self.assertRaises(ValueError):
            load_experiment(source)

    def test_experiment_rejects_duplicate_columns_and_allows_optional_values(
        self,
    ):
        source = self.root / "measurements.csv"
        source.write_text("alpha,CL,CD,Re,Mach,CL\n1,.1,.01,1000000,.1,.2\n")
        with self.assertRaises(ValueError):
            load_experiment(source)
        source.write_text(
            "alpha,CL,CD,Re,Mach,CM,airfoil\n1,.1,.01,1000000,.1\n"
        )
        row = load_experiment(source)[0]
        self.assertEqual(row["airfoil"], "")
        self.assertNotIn("CM", row)

    def test_dashboard_embeds_data_safely_and_retains_missing_samples(self):
        folder = self.run_case()
        path = folder / "run.json"
        manifest = json.loads(path.read_text())
        manifest["airfoil"] = "</script><script>alert(1)</script>"
        atomic_json(path, manifest)
        result = write_dashboard([folder], self.root / "viewer.html")
        html = result.read_text()
        self.assertIn("\\u003c/script\\u003e", html)
        self.assertNotIn("</script><script>alert(1)</script>", html)
        self.assertNotIn("https://", html)
        self.assertIn('id="alpha"', html)

    def test_cli_dispatches_new_modes_and_rejects_incomplete_experiment(self):
        folder = self.run_case()
        with patch.object(cli, "discover_app", return_value=self.app):
            self.assertEqual(
                cli.main(
                    [
                        "--dashboard",
                        str(folder),
                        "--outdir",
                        str(self.root / "ui"),
                    ]
                ),
                0,
            )
            with self.assertRaises(SystemExit):
                cli.main(["--experiment", "missing.csv"])
            with self.assertRaises(SystemExit):
                cli.main(["--repair"])


class AVLTest(unittest.TestCase):
    def test_derivative_is_not_overwritten_by_spiral_ratio(self):
        result = parse_values(
            "Cnb = .002114\nClb Cnr / Clr Cnb = .284038\nCLtot = 3.2D-1"
        )
        self.assertAlmostEqual(result["Cnb"], 0.002114)
        self.assertAlmostEqual(result["CLtot"], 0.32)
        self.assertNotIn("CLa", parse_values("CLa = 1e999"))

    def test_wing_model_and_command_set_actual_mach_and_target_lift(self):
        config = load_wing(ROOT / "examples/wing.json")
        commands = wing_commands(config, {"cl": 0.5, "beta": 0})
        self.assertIn("MN 0.1", commands)
        self.assertIn("A C 0.5", commands)
        with tempfile.TemporaryDirectory() as folder:
            text = geometry_text(config, Path(folder))
        self.assertIn("YDUPLICATE\n0", text)
        self.assertIn("NACA\n0012", text)

    def test_avl_strip_parser_respects_printed_column_names(self):
        rows = parse_strips(
            "Surface # 1 Wing\n j Xle Yle Zle Chord Area c_cl ai cl_norm "
            "cl cd "
            "cdv cm_c/4 cm_LE C.P.x/c\n 1 0 1 0 2 .4 1 .01 .5 .5 .002 0 "
            "-.01 -.1 .25\n"
        )
        self.assertEqual(rows[0]["Yle"], 1)
        self.assertEqual(rows[0]["c_cl"], 1)
        self.assertEqual(rows[0]["cm_c/4"], -0.01)

    def test_invalid_reference_and_duplicate_sections_are_rejected(self):
        original = json.loads((ROOT / "examples/wing.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wing.json"
            for change in ("reference", "sections"):
                config = json.loads(json.dumps(original))
                if change == "reference":
                    config["reference"]["area"] = -1
                else:
                    config["surfaces"][0]["sections"][1]["y"] = 0
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_wing(path)


if __name__ == "__main__":
    unittest.main()
