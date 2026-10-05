"""Reject incomplete AVL loads before reporting integrated wing drag."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xfoil_mac.analysis.coupling import coupled_drag
from xfoil_mac.avl import parse_strips, run_wing
from xfoil_mac.data import sha256_file
from xfoil_mac.runtime import FlowAssumptions

COLUMNS = (
    "j Xle Yle Zle Chord Area c_cl ai cl_norm cl cd cdv cm_c/4 cm_LE C.P.x/c"
)


def wing_model():
    """Four unit-area strips on a mirrored wing with dimensions in metres."""
    return {
        "name": "Coverage test",
        "mach": 0.1,
        "profile_drag": 0.0,
        "reference": {
            "area": 4.0,
            "chord": 1.0,
            "span": 4.0,
            "point": [0.25, 0, 0],
        },
        "surfaces": [
            {
                "name": "Wing",
                "mirror": True,
                "chord_panels": 2,
                "span_panels": 2,
                "sections": [
                    {
                        "x": 0,
                        "y": 0,
                        "z": 0,
                        "chord": 1,
                        "twist": 0,
                        "naca": "0012",
                    },
                    {
                        "x": 0,
                        "y": 2,
                        "z": 0,
                        "chord": 1,
                        "twist": 0,
                        "naca": "0012",
                    },
                ],
            }
        ],
        "cases": [{"alpha": 2}],
    }


def strip_output():
    blocks = []
    for surface, sign in [(1, 1), (2, -1)]:
        first = 2 * surface - 1
        lines = [
            f"Surface # {surface} Wing",
            f"# Chordwise = 2 # Spanwise = 2 First strip = {first}",
            "Surface area Ssurf = 2.000000 Ave. chord Cave = 1.000000",
            COLUMNS,
        ]
        for offset, y in enumerate((0.5, 1.5)):
            lines.append(
                f"{first + offset} 0 {sign * y} 0 1 1.0000 "
                ".5 .01 .5 .5 .002 0 -.01 -.1 .25"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n------------------\n"


class StripIntegrityTest(unittest.TestCase):
    def test_complete_table_and_fortran_exponents_are_preserved(self):
        text = strip_output().replace("1.0000", "1.0000D+00")
        rows = parse_strips(text, wing_model())
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(row["Area"] for row in rows), 4.0)
        self.assertEqual(rows[-1]["Yle"], -1.5)
        self.assertEqual(rows[0]["cl"], 0.5)

    def test_damage_never_silently_discards_a_strip(self):
        original = strip_output()
        first_row = original.splitlines()[4]
        corruptions = {
            "missing row": original.replace(first_row + "\n", ""),
            "short row": original.replace(first_row, "1 0 .5 0 1"),
            "overflow": original.replace(
                first_row, first_row.replace(".002", "*****")
            ),
            "nan": original.replace(
                first_row, first_row.replace(".002", "NaN")
            ),
            "negative area": original.replace(
                first_row, first_row.replace("1.0000", "-1.0000")
            ),
            "missing surface": original.split("Surface # 2")[0],
            "truncated end": original.rsplit("4 0 -1.5", 1)[0],
            "wrong mesh": original.replace("# Spanwise = 2", "# Spanwise = 4"),
            "wrong summed area": original.replace("1.0000 .5", ".5000 .5"),
            "wrong geometry area": original.replace(
                "1.0000 .5", ".5000 .5"
            ).replace("Ssurf = 2.000000", "Ssurf = 1.000000"),
            "missing dimensions": original.replace(
                "Surface area Ssurf = 2.000000", "removed"
            ),
            "empty": "",
        }
        for name, text in corruptions.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_strips(text, wing_model())

    def test_native_rounding_of_strip_areas_is_allowed(self):
        text = strip_output().replace("1.0000 .5", "0.99995 .5")
        self.assertEqual(len(parse_strips(text, wing_model())), 4)

    def test_missing_surface_is_detected_even_if_table_is_self_consistent(
        self,
    ):
        text = strip_output().split("Surface # 2")[0]
        self.assertEqual(len(parse_strips(text)), 2)
        with self.assertRaisesRegex(ValueError, "every model surface"):
            parse_strips(text, wing_model())

    def test_run_and_resume_mark_bad_outputs_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "wing.json"
            source.write_text(json.dumps(wing_model()))
            binary = folder / "avl"
            binary.write_text("mock executable identity")
            output = folder / "run"

            def fake_avl(binary, env, directory, commands, timeout, **kwargs):
                (directory / "totals.txt").write_text(
                    "CLtot = .5 CDind = .002 Cmtot = -.01 "
                    "Alpha = 2 Mach = .1 Beta = 0"
                )
                (directory / "stability.txt").write_text("CLa = 5")
                (directory / "strips.txt").write_text(
                    strip_output().split("Surface # 2")[0]
                )
                return "finished", None

            with (
                patch("xfoil_mac.execution.execute", side_effect=fake_avl),
                patch("xfoil_mac.avl._plot_wing"),
                patch("xfoil_mac.ui.wing.write_wing_interactive"),
            ):
                result = run_wing(source, output, binary)
            saved = json.loads(result.read_text())
            self.assertEqual(saved["status"], "needs_attention")
            self.assertEqual(saved["cases"][0]["status"], "invalid_outputs")
            self.assertIn(
                "every model surface", saved["cases"][0]["strip_error"]
            )
            self.assertNotIn(
                "0.5", (output / "spanwise_loads.csv").read_text()
            )
            # Simulate a legacy manifest that trusted this same damaged file.
            saved["cases"][0]["status"] = "ok"
            result.write_text(json.dumps(saved))
            with (
                patch(
                    "xfoil_mac.execution.execute",
                    side_effect=AssertionError("Unexpected solve"),
                ),
                patch("xfoil_mac.avl._plot_wing"),
                patch("xfoil_mac.ui.wing.write_wing_interactive"),
            ):
                resumed = json.loads(
                    run_wing(source, output, binary, resume=True).read_text()
                )
            self.assertEqual(resumed["cases"][0]["status"], "invalid_outputs")


class CouplingFailureTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "wing"
        self.source.mkdir()
        self.flow = FlowAssumptions(
            chord_m=1,
            air_density=1.225,
            air_viscosity=1.7894e-5,
            sound_speed=340.3,
        )

    def save_cases(self, statuses, text=None):
        cases = []
        for index, status in enumerate(statuses, 1):
            case = {
                "case": index,
                "status": status,
                "target_beta": 0,
                "Alpha": 2,
                "CLtot": 0.5,
                "CDind": 0.002,
            }
            if status == "ok":
                folder = self.source / f"case_{index:03d}"
                folder.mkdir()
                strips = folder / "strips.txt"
                strips.write_text(strip_output() if text is None else text)
                case["raw_sha256"] = {"strips.txt": sha256_file(strips)}
            cases.append(case)
        model = wing_model()
        model["cases"] *= len(cases)
        saved = {
            "config": model,
            "fingerprint": "test AVL configuration",
            "cases": cases,
        }
        (self.source / "wing_run.json").write_text(json.dumps(saved))

    def calculate(self):
        result = coupled_drag(
            None,
            self.source,
            self.root / "coupled",
            speed=34.03,
            flow=self.flow,
        )
        self.assertTrue((result.parent / "report.html").is_file())
        self.assertTrue((result.parent / "wing_drag.csv").is_file())
        return json.loads(result.read_text())

    def test_all_failed_avl_cases_are_saved_without_missing_manifest_error(
        self,
    ):
        self.save_cases(["timeout", "invalid_outputs"])
        with patch(
            "xfoil_mac.workflows.run_polar",
            side_effect=AssertionError("Unexpected polar"),
        ):
            payload = self.calculate()
        self.assertEqual(payload["status"], "needs_attention")
        self.assertEqual(len(payload["cases"]), 2)
        self.assertEqual(
            [case["avl_status"] for case in payload["cases"]],
            ["timeout", "invalid_outputs"],
        )
        self.assertTrue(
            all(case["CD_total"] is None for case in payload["cases"])
        )
        self.assertTrue(
            all(case["coverage"] is None for case in payload["cases"])
        )

    def test_trailing_avl_failure_stays_in_manifest_and_success_is_unchanged(
        self,
    ):
        self.save_cases(["ok", "timeout"])
        with (
            patch("xfoil_mac.workflows.run_polar"),
            patch(
                "xfoil_mac.analysis.coupling.discover_runs",
                return_value=[object()],
            ),
            patch(
                "xfoil_mac.analysis.coupling.interpolate",
                return_value=({"CD": 0.01}, "exact"),
            ),
        ):
            payload = self.calculate()
        self.assertEqual(len(payload["cases"]), 2)
        good = payload["cases"][0]
        self.assertEqual(good["status"], "ok")
        self.assertEqual(good["coverage"], 1.0)
        self.assertAlmostEqual(good["CD_profile"], 0.01)
        self.assertAlmostEqual(good["CD_total"], 0.012)
        self.assertEqual(payload["cases"][1]["status"], "AVL_not_converged")
        self.assertEqual(payload["status"], "needs_attention")

    def test_legacy_partial_and_empty_outputs_do_not_claim_complete_coverage(
        self,
    ):
        for text in (strip_output().split("Surface # 2")[0], ""):
            with (
                self.subTest(empty=not bool(text)),
                tempfile.TemporaryDirectory() as temporary,
            ):
                previous = self.root, self.source
                self.root = Path(temporary)
                self.source = self.root / "wing"
                self.source.mkdir()
                self.save_cases(["ok"], text)
                with patch(
                    "xfoil_mac.workflows.run_polar",
                    side_effect=AssertionError("Unexpected polar"),
                ):
                    payload = self.calculate()
                case = payload["cases"][0]
                self.assertEqual(case["status"], "invalid_AVL_outputs")
                self.assertIsNone(case["coverage"])
                self.assertIsNone(case["CD_total"])
                self.root, self.source = previous

    def test_changed_or_missing_saved_strip_file_is_recorded(self):
        self.save_cases(["ok", "ok"])
        (self.source / "case_001/strips.txt").write_text("damaged after solve")
        (self.source / "case_002/strips.txt").unlink()
        with patch(
            "xfoil_mac.workflows.run_polar",
            side_effect=AssertionError("Unexpected polar"),
        ):
            payload = self.calculate()
        self.assertEqual(
            [case["status"] for case in payload["cases"]],
            ["invalid_AVL_outputs"] * 2,
        )
        self.assertIn("changed", payload["cases"][0]["reason"])

    def test_missing_polar_match_leaves_integrated_drag_unavailable(self):
        self.save_cases(["ok"])
        with (
            patch("xfoil_mac.workflows.run_polar"),
            patch(
                "xfoil_mac.analysis.coupling.discover_runs",
                return_value=[object()],
            ),
            patch(
                "xfoil_mac.analysis.coupling.interpolate",
                side_effect=[({"CD": 0.01}, "exact")] * 3
                + [(None, "missing")],
            ),
        ):
            payload = self.calculate()
        case = payload["cases"][0]
        self.assertEqual(case["status"], "incomplete_strip_coverage")
        self.assertEqual(case["coverage"], 0.75)
        self.assertIsNone(case["CD_profile"])
        self.assertIsNone(case["CD_total"])


if __name__ == "__main__":
    unittest.main()
