"""Scientific and share-output regressions using synthetic local evidence."""

import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from xfoil_mac.analysis import shape, surrogate, trust
from xfoil_mac.storage import dataset
from xfoil_mac.storage.catalog import export_report
from tests.test_boundary_integrity import GEOMETRY, VALID_DUMP, DUMP_LINES
from tests.test_surrogate import SHAPE_ROOT, sufficient_rows


class AlgorithmRegressionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_dataset_records_actual_cl_dependent_conditions(self):
        folder = self.root / "solve"
        folder.mkdir()
        config = {
            "re": 1e6,
            "mach": 0.1,
            "reference_cl": 0.5,
            "input": {"kind": "coordinates", "path": "synthetic.dat"},
        }
        (folder / "polar.csv").write_text(
            "alpha,CL,CD,CM\n0,0.125,0.01,-0.02\n"
        )
        for kind, expected in (
            (1, (1e6, 0.1)),
            (2, (2e6, 0.2)),
            (3, (4e6, 0.1)),
        ):
            with self.subTest(kind=kind):
                config["flow_type"] = kind
                (folder / "run.json").write_text(
                    json.dumps(
                        {
                            "config": config,
                            "requested_alphas": [0, 1],
                            "summary": {"failed_alphas": [1]},
                        }
                    )
                )
                rows, _ = dataset.collect_rows(self.root)
                solved, failed = rows
                self.assertAlmostEqual(solved["re"], expected[0])
                self.assertAlmostEqual(solved["mach"], expected[1])
                self.assertEqual(solved["re_reference"], 1e6)
                self.assertEqual(solved["mach_reference"], 0.1)
                self.assertEqual(solved["condition_status"], "ok")
                self.assertEqual(solved["coordinate_file"], "synthetic.dat")
                self.assertIsNone(failed["re"])
                self.assertEqual(failed["condition_status"], "missing_lift")
        (folder / "polar.csv").write_text("alpha,CL,CD,CM\n0,0,0.01,-0.02\n")
        rows, _ = dataset.collect_rows(self.root)
        self.assertIsNone(rows[0]["re"])
        self.assertEqual(rows[0]["condition_status"], "invalid_condition")

    def test_corrupt_dump_is_invalid_even_when_polar_converged(self):
        (self.root / "geometry.dat").write_text(
            "\n".join(f"{x} {y}" for x, y in GEOMETRY)
        )
        layer = self.root / "boundary_layer"
        layer.mkdir()
        path = layer / "bl_alpha_0.txt"
        corruptions = (
            "",
            "\n".join(DUMP_LINES[:4]),
            "\n".join(DUMP_LINES[:3] + DUMP_LINES[4:]),
            VALID_DUMP.replace(".001", "nan"),
            VALID_DUMP.replace("0.08", "0.09"),
        )
        for contents in corruptions:
            with self.subTest(contents=contents):
                path.write_text(contents)
                point = trust.sweep_report(self.root, [0])["points"][0]
                self.assertTrue(point["converged"])
                self.assertEqual(point["verdict"], "invalid")
                self.assertTrue(point["output_error"])
        path.write_text(VALID_DUMP)
        self.assertEqual(
            trust.sweep_report(self.root, [0])["points"][0]["verdict"], "ok"
        )
        path.unlink()
        self.assertEqual(
            trust.sweep_report(self.root, [0])["points"][0]["verdict"],
            "unchecked",
        )

    def test_separation_length_is_independent_of_panel_density(self):
        grids = (
            np.linspace(0, 1, 101),
            np.r_[np.linspace(0, 0.04, 41), np.linspace(0.05, 1, 60)],
        )
        for grid in grids:
            for xs in (grid, grid[::-1]):
                records = [
                    {"x": float(x), "Cf": float(x - 0.04), "H": 2} for x in xs
                ]
                evidence = trust.surface_evidence(records)
                self.assertAlmostEqual(evidence["longest_run_fraction"], 0.04)
                self.assertAlmostEqual(evidence["separated_fraction"], 0.04)
                self.assertEqual(
                    trust.point_verdict(evidence=evidence)[0], "ok"
                )

    def test_filtering_rechecks_training_and_evaluation_sufficiency(self):
        rows = sufficient_rows()
        # Only one run retains a resolvable shape; input counts are sufficient.
        rows = [
            (
                row
                if row["run_id"] == "run_00"
                else dict(row, airfoil="missing_shape")
            )
            for row in rows
        ]
        self.assertTrue(surrogate.sufficiency(rows)["fit_allowed"])
        fitted = surrogate.fit(rows, kind="ridge", shape_root=SHAPE_ROOT)
        evaluated = surrogate.evaluate(
            rows, surrogate.ridge_factory, shape_root=SHAPE_ROOT
        )
        for outcome in (fitted, evaluated):
            self.assertFalse(outcome["fitted"])
            self.assertEqual(outcome["sufficiency"]["usable_rows"], 4)
            self.assertEqual(outcome["sufficiency"]["runs"], 1)
        held = surrogate.holdout_evaluate(
            rows, surrogate.ridge_factory, shape_root=SHAPE_ROOT
        )
        self.assertFalse(held["reported"])
        self.assertEqual(held["train_runs"], 1)

    def test_old_descriptor_models_require_retraining(self):
        model = self.root / "model.joblib"
        model.with_suffix(".metadata.json").write_text(
            json.dumps(
                {
                    "descriptor_version": shape.DESCRIPTOR_VERSION - 1,
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "retrain"):
            surrogate.load(model)


class SharePrivacyTest(unittest.TestCase):
    def test_export_redacts_content_and_excludes_local_only_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary).resolve() / "result"
            folder.mkdir()
            external = "/Users/test-user/private project/shape.dat"
            internal = str(folder / "geometry.dat")
            document = {
                "source": external,
                "geometry": internal,
                "CL": 0.5,
                "error": f"Cannot read {external}",
                "windows": "C:\\Users\\test-user\\private\\shape.dat",
                "file_uri": "file:///Users/test-user/private/shape.dat",
                "escaped": "\\/Users\\/test-user\\/private\\/shape.dat",
            }
            original = json.dumps(document)
            (folder / "run.json").write_text(original)
            (folder / "report.html").write_text(
                f'<html><p>{external}</p><a href="run.json">data</a></html>'
            )
            (folder / "geometry.dat").write_text("1 0\n0 0\n1 0\n")
            (folder / "summary.csv").write_text(
                'source,CL\n"' + external + '",0.5\n'
            )
            for name in (".env", "private.txt", "avl.log", "model.joblib"):
                (folder / name).write_text("PRIVATE_SENTINEL")
            settings = folder / "settings"
            settings.mkdir()
            (settings / "run.json").write_text(original)
            outside = folder.parent / "outside"
            outside.mkdir()
            (outside / "polar.txt").write_text("PRIVATE_SENTINEL")
            (folder / "linked").symlink_to(outside, target_is_directory=True)
            output = export_report(folder, folder / "report.zip")
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                self.assertEqual(
                    names,
                    {
                        "run.json",
                        "report.html",
                        "geometry.dat",
                        "summary.csv",
                        "SHARING.txt",
                    },
                )
                for name in names:
                    text = archive.read(name).decode()
                    self.assertNotIn("test-user", text)
                    self.assertNotIn("PRIVATE_SENTINEL", text)
                    self.assertNotIn(str(folder), text)
                saved = json.loads(archive.read("run.json"))
                self.assertEqual(saved["geometry"], "geometry.dat")
                self.assertEqual(saved["CL"], 0.5)
                rows = list(
                    csv.DictReader(
                        io.StringIO(archive.read("summary.csv").decode())
                    )
                )
                self.assertEqual(rows[0]["CL"], "0.5")
            self.assertEqual((folder / "run.json").read_text(), original)
            self.assertTrue((folder / "avl.log").is_file())


if __name__ == "__main__":
    unittest.main()
