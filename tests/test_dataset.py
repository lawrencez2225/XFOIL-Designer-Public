"""Build one labelled table from run directories, with nothing dropped."""

import csv
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from xfoil_mac.storage import dataset as dataset_module
from xfoil_mac.storage import (
    COLUMNS,
    collect_rows,
    discover_runs,
    summarise,
    write_dataset,
)

POLAR_HEADER = "alpha,CL,CD,CDp,CM,Top_Xtr,Bot_Xtr,Top_Itr,Bot_Itr"
SOLVER_SETTINGS = {
    "ncrit": 9,
    "xtr_top": 1,
    "xtr_bottom": 1,
    "panels": 24,
    "panel_bunching": 1,
    "te_le_ratio": 0.15,
}
STATIONS_PER_SURFACE = 24
THICKNESS = 0.12
LE_STATIONS = 3
TE_STATIONS = 3


def _cosine_positions(count):
    return [
        0.5 * (1 - math.cos(math.pi * i / (count - 1))) for i in range(count)
    ]


def geometry_points():
    """A closed finite-thickness contour, upper TE to lower TE."""
    positions = list(reversed(_cosine_positions(STATIONS_PER_SURFACE)))
    upper = []
    for x in positions:
        y = (
            5
            * THICKNESS
            * (
                0.2969 * math.sqrt(x)
                - 0.1260 * x
                - 0.3516 * x**2
                + 0.2843 * x**3
                - 0.1015 * x**4
            )
        )
        upper.append((x, y))
    lower = [(x, -y) for x, y in reversed(upper[1:-1])]
    return upper + lower


def leading_station(points):
    return min(range(len(points)), key=lambda i: points[i][0])


def dump_text(points, *, leading_separated, trailing_separated):
    """A DUMP with separation placed at named ends of each surface.

    Stations run upper trailing edge to leading edge, then lower leading
    edge to trailing edge, so the leading edge is an interior index while
    both trailing edges are file ends.
    """
    leading = leading_station(points)
    surfaces = ((0, leading + 1), (leading + 1, len(points)))
    lines = []
    for start, stop in surfaces:
        span = stop - start
        for offset in range(span):
            index = start + offset
            x, y = points[index]
            separated = (leading_separated and offset < LE_STATIONS) or (
                trailing_separated and offset >= span - TE_STATIONS
            )
            Cf = -0.003 if separated else 0.002
            lines.append(
                " ".join(
                    f"{v:g}"
                    for v in (
                        index * 0.01,
                        x,
                        y,
                        1.0,
                        0.01,
                        0.005,
                        Cf,
                        2.0,
                    )
                )
            )
    return "\n".join(lines) + "\n"


def write_run(
    root,
    name,
    *,
    alphas=(),
    failed=(),
    boundary_layer,
    dumps,
    unreadable=False,
):
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True)
    if unreadable:
        (run_dir / "run.json").write_text("{ not json")
        return run_dir
    points = geometry_points()
    (run_dir / "geometry.dat").write_text(
        "\n".join(f"{x:.7f} {y:.7f}" for x, y in points) + "\n"
    )
    (run_dir / "polar.csv").write_text(
        POLAR_HEADER
        + "\n"
        + "\n".join(
            f"{a} {a * 0.1:.4f} 0.01 0.003 -0.02 0.5 0.6 40 60" for a in alphas
        )
        + "\n"
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "airfoil": "NACA2412",
                "requested_alphas": list(alphas) + list(failed),
                "summary": {"failed_alphas": list(failed)},
                "config": {
                    "input": {"kind": "naca", "naca": "2412"},
                    "re": 367500.0,
                    "mach": 0.0588,
                    "boundary_layer": boundary_layer,
                    "solver_settings": SOLVER_SETTINGS,
                },
            }
        )
    )
    if boundary_layer:
        layer = run_dir / "boundary_layer"
        layer.mkdir()
        for alpha, ends in dumps.items():
            label = f"{alpha:g}".replace("-", "m").replace(".", "p")
            (layer / f"bl_alpha_{label}.txt").write_text(
                dump_text(
                    points,
                    leading_separated=ends.get("leading", False),
                    trailing_separated=ends.get("trailing", False),
                )
            )
    return run_dir


TRAILING_ONLY = {"leading": False, "trailing": True}
BOTH_ENDS = {"leading": True, "trailing": True}


class FlowRelationTest(unittest.TestCase):
    """The dataset must expose how a run's Re and Mach were paired.

    Re and Mach are independent numbers, so a pair can be one that no
    chord and fluid could produce. Recording that is only useful if the
    table a reader works from carries it.
    """

    def test_the_pairing_is_a_column(self):
        self.assertIn("flow_relation", dataset_module.COLUMNS)
        self.assertEqual(
            dataset_module.COLUMNS[
                dataset_module.COLUMNS.index("flow_relation") - 1
            ],
            "mach",
        )

    def _case(self, *, with_field):
        folder = Path(tempfile.mkdtemp()) / "run"
        self.addCleanup(shutil.rmtree, folder.parent, ignore_errors=True)
        folder.mkdir(parents=True)
        config = {
            "input": {"kind": "naca", "naca": "2412"},
            "re": 698895.0,
            "mach": 0.1,
            "boundary_layer": False,
            "solver_settings": dict(SOLVER_SETTINGS),
        }
        if with_field:
            config["flow_relation"] = "derived_re_from_mach"
        (folder / "run.json").write_text(
            json.dumps(
                {
                    "airfoil": "NACA2412",
                    "config": config,
                    "requested_alphas": [0.0],
                    "summary": {},
                }
            )
        )
        (folder / "polar.csv").write_text(
            POLAR_HEADER + "\n0.0,0.2,0.01,0.001,0.0,0.5,0.5,10,10\n"
        )
        return dataset_module.load_run(folder, folder.parent)

    def test_the_pairing_survives_into_the_rows(self):
        run = self._case(with_field=True)
        self.assertEqual(run["flow_relation"], "derived_re_from_mach")
        row = dataset_module._row_for(run, 0.0)
        self.assertEqual(row["flow_relation"], "derived_re_from_mach")

    def test_a_run_without_the_field_stays_empty_not_guessed(self):
        run = self._case(with_field=False)
        self.assertIsNone(run["flow_relation"])
        row = dataset_module._row_for(run, 0.0)
        self.assertIsNone(row["flow_relation"])


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_discovers_run_directories_only(self):
        run_dir = write_run(
            self.root,
            "case_001",
            alphas=[0.0],
            boundary_layer=False,
            dumps={},
        )
        (run_dir / "attempts" / "0001").mkdir(parents=True)
        (run_dir / "attempts" / "0001" / "geometry.dat").write_text("1 0\n")
        self.assertEqual(discover_runs(self.root), [run_dir])

    def test_trailing_separation_alone_is_not_flagged(self):
        write_run(
            self.root,
            "case_001",
            alphas=[0.0],
            boundary_layer=True,
            dumps={0.0: TRAILING_ONLY},
        )
        rows, _ = collect_rows(self.root)
        self.assertEqual(rows[0]["verdict"], "ok")
        self.assertEqual(rows[0]["run_count"], 1)

    def test_separation_at_both_ends_is_flagged(self):
        write_run(
            self.root,
            "case_001",
            alphas=[12.0],
            boundary_layer=True,
            dumps={12.0: BOTH_ENDS},
        )
        rows, _ = collect_rows(self.root)
        self.assertEqual(rows[0]["verdict"], "suspect")
        self.assertEqual(rows[0]["run_count"], 2)
        self.assertIn("simultaneous_separated_regions", rows[0]["reasons"])

    def test_failed_angle_keeps_its_row_with_empty_values(self):
        write_run(
            self.root,
            "case_001",
            alphas=[-4.0, -3.5, -2.75],
            failed=[-3.0],
            boundary_layer=True,
            dumps={-4.0: TRAILING_ONLY},
        )
        rows, _ = collect_rows(self.root)
        self.assertEqual(len(rows), 4)
        failed = next(r for r in rows if r["alpha"] == -3.0)
        self.assertEqual(failed["CL"], None)
        self.assertEqual(failed["verdict"], "retry")
        self.assertEqual(failed["converged"], False)

    def test_run_without_boundary_layer_is_unchecked_not_passed(self):
        write_run(
            self.root,
            "case_001",
            alphas=[0.0, 4.0],
            boundary_layer=False,
            dumps={},
        )
        rows, _ = collect_rows(self.root)
        self.assertEqual({r["verdict"] for r in rows}, {"unchecked"})
        self.assertTrue(all(r["has_boundary_layer"] is False for r in rows))

    def test_geometry_descriptors_describe_the_solved_contour(self):
        write_run(
            self.root,
            "case_001",
            alphas=[0.0],
            boundary_layer=False,
            dumps={},
        )
        rows, _ = collect_rows(self.root)
        row = rows[0]
        self.assertTrue(row["geometry_valid"])
        self.assertAlmostEqual(row["thickness_ratio"], THICKNESS, places=3)
        self.assertAlmostEqual(row["max_camber"], 0.0, places=3)

    def test_summary_counts_every_row(self):
        write_run(
            self.root,
            "case_001",
            alphas=[0.0, 12.0],
            boundary_layer=True,
            dumps={0.0: TRAILING_ONLY, 12.0: BOTH_ENDS},
        )
        write_run(
            self.root,
            "case_002",
            alphas=[1.0],
            boundary_layer=False,
            dumps={},
        )
        rows, _ = collect_rows(self.root)
        counts = summarise(rows)
        self.assertEqual(counts["rows"], 3)
        self.assertEqual(counts["runs"], 2)
        self.assertEqual(counts["verdicts"]["ok"], 1)
        self.assertEqual(counts["verdicts"]["suspect"], 1)
        self.assertEqual(counts["verdicts"]["unchecked"], 1)
        self.assertEqual(counts["runs_without_boundary_layer"], 1)

    def test_written_table_round_trips_with_provenance(self):
        write_run(
            self.root,
            "case_001",
            alphas=[0.0, 12.0],
            boundary_layer=True,
            dumps={0.0: TRAILING_ONLY, 12.0: BOTH_ENDS},
        )
        destination = self.root / "out" / "table.csv"
        summary = write_dataset(self.root, destination)
        with destination.open(newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(tuple(reader.fieldnames), COLUMNS)
            written = list(reader)
        self.assertEqual(len(written), 2)
        self.assertEqual(written[0]["verdict"], "ok")
        self.assertEqual(written[1]["verdict"], "suspect")
        document = json.loads(
            destination.with_suffix(".provenance.json").read_text()
        )
        self.assertEqual(document["schema_version"], 2)
        self.assertEqual(document["summary"], summary)
        self.assertEqual(len(document["runs"]), 1)
        self.assertTrue(document["runs"][0]["run_json_sha256"])

    def test_unreadable_run_is_recorded_not_dropped(self):
        write_run(
            self.root,
            "case_001",
            boundary_layer=False,
            dumps={},
            unreadable=True,
        )
        rows, provenance = collect_rows(self.root)
        self.assertEqual(rows, [])
        self.assertEqual(provenance[0]["status"], "unreadable")


if __name__ == "__main__":
    unittest.main()
