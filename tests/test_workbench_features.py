"""Scientific edge cases and the storage boundaries of the new workbench."""

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from xfoil_mac.analysis.boundary_layer import read_boundary_layer
from xfoil_mac.analysis.coupling import local_profile
from xfoil_mac.analysis.design import make_shape, pareto_front
from xfoil_mac.analysis.diagnostics import log_evidence
from xfoil_mac.execution import solver_commands
from xfoil_mac.runtime import discover_app, FlowAssumptions
from xfoil_mac.solver_settings import SolverSettings
from xfoil_mac.storage.catalog import (
    within,
    save_preset,
    presets,
    export_report,
)
from xfoil_mac.ui.service import conditions, preflight
from xfoil_mac.ui.wing import simple_wing, wing_mesh


class WorkbenchFeatureTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.app = replace(discover_app(), run_root=self.root / "runs")

    def test_settings_validation_and_command_order(self):
        for changes in (
            {"panels": 601},
            {"panels": True},
            {"xtr_top": 1.1},
            {"ncrit": float("nan")},
        ):
            with self.assertRaises(ValueError):
                SolverSettings(**changes)
        config = dict(
            input={"kind": "naca", "naca": "2412"},
            re=1e6,
            mach=0.05,
            iterations=100,
            solver_settings=SolverSettings(panels=160, xtr_top=0.3).to_dict(),
            boundary_layer=True,
        )
        commands = solver_commands(config, [0, 2])
        self.assertIn("N 160", commands)
        self.assertIn("XTR 0.3 1", commands)
        self.assertLess(commands.index("XTR 0.3 1"), commands.index("PACC"))
        self.assertEqual(sum(c.startswith("DUMP ") for c in commands), 2)

    def test_speed_keeps_re_and_mach_physically_consistent(self):
        flow = FlowAssumptions(chord_m=0.3)
        re, mach = conditions({"speed": 20, "flow": vars(flow)})[0]
        self.assertAlmostEqual(
            re, flow.air_density * 20 * 0.3 / flow.air_viscosity
        )
        self.assertAlmostEqual(mach, 20 / flow.sound_speed)
        with self.assertRaises(ValueError):
            conditions({"speed": 20, "re": 1e6})

    def test_preflight_cartesian_product_and_cap(self):
        spec = {
            "kind": "polar",
            "nacas": ["0012", "2412"],
            "re_values": [5e5, 1e6],
            "mach_values": [0.03, 0.05],
            "aseq": [0, 4, 2],
        }
        self.assertEqual(preflight(spec)["requested_points"], 24)
        with self.assertRaises(ValueError):
            preflight(dict(spec, re_values=list(range(1, 100))))

    def test_bl_surface_and_wake_are_distinct(self):
        file = self.root / "dump.txt"
        file.write_text(
            "# s x y Ue Dstar Theta Cf H\n"
            + "".join(
                f"{i} {x} {y} 1 .01 .005 {cf} 2\n"
                for i, (x, y, cf) in enumerate(
                    [
                        (1, 0.1, 0.001),
                        (0, 0, 0),
                        (1, -0.1, -0.001),
                        (1.1, 0, 0),
                    ]
                )
            )
        )
        data = read_boundary_layer(file, [(1, 0.1), (0, 0), (1, -0.1)])
        self.assertEqual(
            [r["surface"] for r in data], ["upper", "upper", "lower", "wake"]
        )
        self.assertEqual(data[2]["Cf"], -0.001)

    def test_oscillation_label_uses_residual_evidence(self):
        file = self.root / "log"
        file.write_text(
            "Convergence failed\n"
            + "".join(f"rms: {v:.2e}\n" for v in [0.1, 0.2] * 6)
        )
        self.assertEqual(
            log_evidence(file, "finished")["reason"], "oscillating_iterations"
        )
        self.assertEqual(log_evidence(file, "timeout")["reason"], "timeout")

    def test_tapered_wing_reference_geometry(self):
        model = simple_wing(span=8, root_chord=2, tip_chord=1, sweep=10)
        self.assertAlmostEqual(model["reference"]["area"], 12)
        self.assertAlmostEqual(model["reference"]["chord"], 14 / 9)
        self.assertGreater(model["surfaces"][0]["sections"][1]["x"], 0)
        mesh = wing_mesh(model)
        self.assertGreater(np.ptp(np.asarray(mesh["triangles"])[:, :, 2]), 0.2)

    def test_local_profile_mirrors_and_rejects_extrapolation(self):
        s = simple_wing()["surfaces"][0]
        np.testing.assert_allclose(local_profile(s, -2), local_profile(s, 2))
        with self.assertRaises(ValueError):
            local_profile(s, 5)

    def test_thickness_and_blend_are_saved_without_changing_source(self):
        file, metrics = make_shape(
            self.app,
            self.root / "shape",
            {"naca": "0012"},
            thickness=1.2,
            blend_source={"naca": "2412"},
            blend=0.5,
        )
        self.assertTrue(file.is_file())
        self.assertAlmostEqual(metrics["thickness_ratio"], 0.144, delta=0.001)
        self.assertGreater(metrics["max_camber"], 0.008)

    def test_pareto_filters_infeasible_and_dominated_candidates(self):
        rows = [
            dict(candidate=1, feasible=True, mean_CD=0.01, worst_LD=50),
            dict(candidate=2, feasible=True, mean_CD=0.02, worst_LD=60),
            dict(candidate=3, feasible=True, mean_CD=0.02, worst_LD=40),
            dict(candidate=4, feasible=False, mean_CD=0.001, worst_LD=100),
        ]
        self.assertEqual(pareto_front(rows), [1, 2])

    def test_results_paths_cannot_escape_through_symlinks(self):
        (self.root / "runs").mkdir()
        (self.root / "runs/link").symlink_to(self.root)
        for value in ["../secrets", "link/secrets", "/etc/passwd"]:
            with self.assertRaises(ValueError):
                within(self.app.run_root, value)

    def test_presets_round_trip_and_export_relative_entries(self):
        spec = {"kind": "polar", "speed": 20, "flow": {"chord_m": 0.3}}
        save_preset(self.app, "风洞设置", spec)
        self.assertEqual(presets(self.app)["presets"]["风洞设置"], spec)
        folder = self.root / "result"
        folder.mkdir()
        (folder / "report.html").write_text("report")
        (folder / "raw.txt").write_text("data")
        (folder / "outside").symlink_to("/etc/passwd")
        output = export_report(folder, folder / "report.zip")
        import zipfile

        with zipfile.ZipFile(output) as z:
            self.assertEqual(set(z.namelist()), {"report.html", "raw.txt"})


@unittest.skipUnless(
    os.environ.get("XFOIL_NATIVE_TESTS") == "1",
    "Opt-in native scientific checks",
)
class NewNativeFeaturesTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.app = replace(discover_app(), run_root=self.root / "runs")

    def test_direct_cl_matches_target_and_resume_does_not_recompute(self):
        from xfoil_mac.analysis.lift import run_lift

        file = run_lift(
            self.app,
            [0.2, 0.5],
            self.root / "lift",
            reynolds=1e6,
            mach=0.05,
            boundary_layer=True,
        )
        saved = json.loads(file.read_text())
        self.assertEqual(saved["status"], "ok")
        for point in saved["points"]:
            self.assertAlmostEqual(
                point["target_cl"], point["result"]["CL"], delta=0.00015
            )
            self.assertTrue(
                (
                    file.parent
                    / point["attempts"][-1]["directory"]
                    / "boundary_layer.txt"
                ).is_file()
            )
        with patch(
            "xfoil_mac.analysis.lift.execute",
            side_effect=AssertionError("unexpected rerun"),
        ):
            run_lift(
                self.app,
                [0.2, 0.5],
                self.root / "lift",
                reynolds=1e6,
                mach=0.05,
                boundary_layer=True,
                resume=True,
            )

    def test_transition_panel_and_boundary_layer_are_applied(self):
        from xfoil_mac.workflows import run_polar
        from xfoil_mac.data import parse_polar_file, read_pairs

        kwargs = dict(
            app=self.app,
            naca="2412",
            reynolds=1e6,
            mach=0.05,
            alpha_start=0,
            alpha_end=2,
            alpha_step=2,
            out_file=self.root / "polar.txt",
            solver_settings=SolverSettings(
                panels=160, xtr_top=0.3, xtr_bottom=0.5, ncrit=7
            ),
            boundary_layer=True,
            show_xfoil_geometry=False,
            quiet=True,
            save_pressure_vectors=False,
        )
        run_polar(**kwargs)
        rows = parse_polar_file(self.root / "polar.txt")
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(read_pairs(self.root / "geometry.dat")), 160)
        self.assertAlmostEqual(rows[0]["Top_Xtr"], 0.3, places=3)
        self.assertAlmostEqual(rows[0]["Bot_Xtr"], 0.5, places=3)
        bl = read_boundary_layer(
            self.root / "boundary_layer/bl_alpha_0.txt",
            read_pairs(self.root / "geometry.dat"),
        )
        self.assertEqual(sum(r["surface"] != "wake" for r in bl), 160)
        self.assertTrue(any(r["surface"] == "wake" for r in bl))
        with patch(
            "xfoil_mac.execution.execute",
            side_effect=AssertionError("unexpected rerun"),
        ):
            run_polar(**kwargs, resume=True)

    def test_native_flap_generation_is_valid(self):
        file, metrics = make_shape(
            self.app, self.root / "flap", {"naca": "0012"}, flap=5
        )
        self.assertTrue(file.exists())
        self.assertGreater(abs(metrics["max_camber"]), 0.005)
        self.assertIn("Flap hinge:", (file.parent / "xfoil.log").read_text())
