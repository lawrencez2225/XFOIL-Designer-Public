"""The validity boundary as a table, derived only from what is on disk."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from xfoil_mac.analysis.domain import summarise_root, write_summary

POLAR_HEADER = "alpha,CL,CD,CDp,CM,Top_Xtr,Bot_Xtr,Top_Itr,Bot_Itr"
STATIONS = 20


def _dump(separated_ends):
    """Stations from the upper trailing edge to the lower trailing edge."""
    lines = []
    for index in range(2 * STATIONS):
        offset = index if index < STATIONS else 2 * STATIONS - 1 - index
        separated = (
            (offset < 2 or offset >= STATIONS - 2) if separated_ends else False
        )
        Cf = -0.003 if separated else 0.002
        x = 1.0 - offset / (STATIONS - 1)
        lines.append(f"{index * 0.01} {x} 0.0 1.0 0.01 0.005 {Cf} 2.0")
    return "\n".join(lines) + "\n"


def write_run(root, name, *, alphas, ends):
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True)
    # The evidence reader validates every DUMP against its saved panels.
    geometry = [line.split()[1:3] for line in _dump(False).splitlines()]
    (run_dir / "geometry.dat").write_text(
        "\n".join(" ".join(pair) for pair in geometry) + "\n"
    )
    (run_dir / "polar.csv").write_text(
        POLAR_HEADER
        + "\n"
        + "\n".join(
            f"{a},{a * 0.1:.4f},0.01,0.003,-0.02,0.5,0.6,40,60" for a in alphas
        )
        + "\n"
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "airfoil": "NACA2412",
                "requested_alphas": list(alphas),
                "summary": {"failed_alphas": []},
                "config": {
                    "mach": 0.1,
                    "re": 500000.0,
                    "boundary_layer": True,
                    "solver_settings": {"panels": 240},
                },
            }
        )
    )
    layer = run_dir / "boundary_layer"
    layer.mkdir()
    for alpha, both in ends.items():
        label = f"{alpha:g}".replace("-", "m").replace(".", "p")
        (layer / f"bl_alpha_{label}.txt").write_text(_dump(both))
    return run_dir


class DomainTableTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_boundary_stops_before_the_first_flagged_angle(self):
        write_run(
            self.root,
            "NACA2412_M0.1_Re500000",
            alphas=[0.0, 4.0, 8.0, 12.0],
            ends={0.0: False, 4.0: False, 8.0: True, 12.0: True},
        )
        rows, distribution = summarise_root(self.root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["last_trusted_alpha"], 4.0)
        self.assertEqual(rows[0]["first_suspect_alpha"], 8.0)
        self.assertEqual(distribution["median"], 4.0)

    def test_configuration_is_recovered_from_the_directory_name(self):
        write_run(
            self.root,
            "NACA4412_M0.25_Re1747237",
            alphas=[0.0],
            ends={0.0: False},
        )
        rows, _ = summarise_root(self.root)
        self.assertEqual(rows[0]["airfoil"], "NACA4412")
        self.assertEqual(rows[0]["mach"], 0.25)
        self.assertEqual(rows[0]["re"], 1747237.0)

    def test_lift_maximum_is_read_from_the_polar(self):
        write_run(
            self.root,
            "NACA2412_M0.1_Re500000",
            alphas=[0.0, 10.0],
            ends={0.0: False, 10.0: False},
        )
        rows, _ = summarise_root(self.root)
        self.assertAlmostEqual(rows[0]["cl_max"], 1.0)
        self.assertEqual(rows[0]["alpha_at_cl_max"], 10.0)

    def test_a_run_with_no_evidence_is_still_listed(self):
        write_run(
            self.root,
            "NACA2412_M0.1_Re500000",
            alphas=[0.0],
            ends={},
        )
        rows, _ = summarise_root(self.root)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["last_trusted_alpha"])

    def test_a_directory_without_a_manifest_is_ignored(self):
        (self.root / "stray").mkdir()
        rows, distribution = summarise_root(self.root)
        self.assertEqual(rows, [])
        self.assertEqual(distribution["n"], 0)

    def test_written_table_round_trips(self):
        write_run(
            self.root,
            "NACA2412_M0.1_Re500000",
            alphas=[0.0, 8.0],
            ends={0.0: False, 8.0: True},
        )
        destination = self.root / "out" / "domain.csv"
        report = write_summary(self.root, destination)
        self.assertEqual(report["rows"], 1)
        with destination.open(newline="") as handle:
            written = list(csv.DictReader(handle))
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["airfoil"], "NACA2412")
        self.assertEqual(written[0]["last_trusted_alpha"], "0.0")


if __name__ == "__main__":
    unittest.main()
