"""Wing reference quantities stay explicit across preview and editing flows."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from xfoil_mac.ui.server import Workbench, make_handler
from xfoil_mac.ui.wing import reference_geometry, simple_wing, wing_wizard


class WingReferenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = SimpleNamespace(app_root=self.root, run_root=self.root)
        self.model = simple_wing()
        self.model["reference"] = {
            "area": 12.0,
            "chord": 1.5,
            "span": 9.0,
            "point": [0.6, 0.2, -0.1],
        }

    def preview(self, **options):
        """Exercise the actual preview handler without binding a port."""
        workbench = SimpleNamespace(app=self.app, token="test-token")
        workbench.check_paths = lambda value: Workbench.check_paths(
            workbench, value
        )
        handler_type = make_handler(workbench)
        handler = handler_type.__new__(handler_type)
        handler.server = SimpleNamespace(server_port=8765)
        handler.path = "/api/preview"
        payload = json.dumps({"model": self.model, **options}).encode()
        handler.headers = {
            "Host": "127.0.0.1:8765",
            "X-Xfoil-Token": workbench.token,
            "Content-Length": str(len(payload)),
        }
        handler.rfile = io.BytesIO(payload)
        handler.send = Mock()
        handler.do_POST()
        args = handler.send.call_args.args
        self.assertEqual(args[1] if len(args) > 1 else 200, 200, args[0])
        return args[0]["model"]

    def test_preview_preserves_all_explicit_reference_quantities(self):
        result = self.preview()
        self.assertEqual(result["reference"], self.model["reference"])

    def test_preview_generates_reference_only_when_absent(self):
        del self.model["reference"]
        result = self.preview()
        self.assertEqual(result["reference"], simple_wing()["reference"])

    def test_explicit_recalculation_preserves_moment_reference_point(self):
        result = self.preview(recalculate_reference=True)
        self.assertEqual(result["reference"]["area"], 8)
        self.assertEqual(result["reference"]["chord"], 1)
        self.assertEqual(result["reference"]["span"], 8)
        self.assertEqual(result["reference"]["point"], [0.6, 0.2, -0.1])

    def test_recalculation_preserves_avl_origin_when_point_was_omitted(self):
        del self.model["reference"]["point"]
        result = self.preview(recalculate_reference=True)
        self.assertEqual(result["reference"]["point"], [0, 0, 0])

    def test_explicit_point_reset_uses_new_quarter_chord(self):
        result = self.preview(
            recalculate_reference=True, reset_reference_point=True
        )
        self.assertEqual(result["reference"], simple_wing()["reference"])

    def test_offset_mirrored_roots_do_not_shorten_tip_to_tip_span(self):
        surface = simple_wing()["surfaces"][0]
        surface["sections"][0]["y"] = 1
        reference = reference_geometry(surface)
        self.assertEqual(reference["span"], 8)
        self.assertEqual(reference["area"], 6)
        self.assertEqual(reference["chord"], 1)

    def test_rectangular_and_tapered_reference_values_are_unchanged(self):
        self.assertEqual(
            simple_wing()["reference"],
            {
                "area": 8.0,
                "chord": 1.0,
                "span": 8.0,
                "point": [0.25, 0.0, 0.0],
            },
        )
        tapered = simple_wing(root_chord=2, tip_chord=1)["reference"]
        self.assertEqual(tapered["area"], 12)
        self.assertAlmostEqual(tapered["chord"], 14 / 9)
        self.assertEqual(tapered["span"], 8)
        self.assertAlmostEqual(tapered["point"][0], 7 / 18)

    def test_unmirrored_span_uses_actual_section_extent(self):
        surface = simple_wing()["surfaces"][0]
        surface["mirror"] = False
        surface["sections"][0]["y"] = 1
        reference = reference_geometry(surface)
        self.assertEqual(reference["span"], 3)
        self.assertEqual(reference["area"], 3)

    def test_cli_preserves_reference_unless_recalculation_is_selected(self):
        source = self.root / "input.json"
        source.write_text(json.dumps(self.model))
        for recalculate, reset_point in (
            (False, False),
            (True, False),
            (True, True),
        ):
            with self.subTest(
                recalculate=recalculate, reset_point=reset_point
            ):

                def answer(prompt):
                    if prompt.startswith("载入已有机翼"):
                        return str(source)
                    if prompt.startswith("按修改后的主翼"):
                        return "y" if recalculate else ""
                    if prompt.startswith("同时重置力矩参考点"):
                        return "y" if reset_point else ""
                    if prompt.startswith("开始 AVL"):
                        return "n"
                    return ""

                with (
                    patch("builtins.input", side_effect=answer),
                    patch("builtins.print"),
                    patch("webbrowser.open"),
                ):
                    result = json.loads(wing_wizard(self.app).read_text())
                expected = (
                    simple_wing()["reference"]
                    if recalculate
                    else dict(self.model["reference"])
                )
                if recalculate and not reset_point:
                    expected["point"] = self.model["reference"]["point"]
                self.assertEqual(result["reference"], expected)


if __name__ == "__main__":
    unittest.main()
