"""The batch analysis pass, exercised against a small synthetic batch."""

import json
import tempfile
import unittest
from pathlib import Path

import analyse_batch
import audit_data
import shutil

POLAR_HEADER = "alpha,CL,CD,CDp,CM,Top_Xtr,Bot_Xtr,Top_Itr,Bot_Itr"
STATIONS = 20


def _dump(separated, points):
    lines = []
    leading = min(range(len(points)), key=lambda i: points[i][0])
    for index, (x, y) in enumerate(points):
        ends = (
            index < 2 or abs(index - leading) < 2 or index >= len(points) - 2
        )
        Cf = -0.003 if (separated and ends) else 0.002
        lines.append(f"{index * 0.01} {x} {y} 1.0 0.01 0.005 {Cf} 2.0")
    return "\n".join(lines) + "\n"


def write_run(root, name, *, thickness, camber, alphas, separated_at):
    run_dir = Path(root) / "uiuc_test" / "M0.1_Re500000" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "polar.csv").write_text(
        POLAR_HEADER
        + "\n"
        + "\n".join(
            f"{a},{a * 0.1:.4f},0.01,0.003,-0.02,0.5,0.6,40,60" for a in alphas
        )
        + "\n"
    )
    (run_dir / "geometry.dat").write_text(
        "\n".join(f"{x:.7f} {y:.7f}" for x, y in _contour(thickness, camber))
        + "\n"
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "airfoil": name,
                "status": "ok",
                "requested_alphas": list(alphas),
                "summary": {"failed_alphas": []},
                "config": {
                    "input": {"kind": "file"},
                    "re": 500000.0,
                    "mach": 0.10,
                    "boundary_layer": True,
                    "solver_settings": {"panels": 240},
                },
            }
        )
    )
    layer = run_dir / "boundary_layer"
    layer.mkdir(exist_ok=True)
    for alpha in alphas:
        label = f"{alpha:g}".replace("-", "m").replace(".", "p")
        (layer / f"bl_alpha_{label}.txt").write_text(
            _dump(alpha >= separated_at, _contour(thickness, camber))
        )
    return run_dir


def _contour(thickness, camber):
    import math

    count = 24
    xs = [
        0.5 * (1 - math.cos(math.pi * i / (count - 1))) for i in range(count)
    ][::-1]
    upper = []
    for x in xs:
        y = (
            5
            * thickness
            * (
                0.2969 * math.sqrt(x)
                - 0.1260 * x
                - 0.3516 * x**2
                + 0.2843 * x**3
                - 0.1015 * x**4
            )
        )
        upper.append((x, y + camber * math.sin(math.pi * x)))
    lower = [
        (x, -y + camber * math.sin(math.pi * x))
        for x, y in reversed(upper[1:-1])
    ]
    return upper + lower


class AnalyseBatchTest(unittest.TestCase):
    """The pass is expensive, so it runs once and every test reads it."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        # Several runs with varied geometry so a fit is permitted.
        for index, (thickness, camber) in enumerate(
            [(0.09, 0.0), (0.12, 0.02), (0.15, 0.04), (0.18, 0.06)]
        ):
            write_run(
                cls.root,
                f"airfoil{index}",
                thickness=thickness,
                camber=camber,
                alphas=[0.0, 4.0, 8.0, 12.0],
                separated_at=8.0,
            )
        cls.summary = analyse_batch.analyse(cls.root, folds=2, root=cls.root)
        cls.report = analyse_batch.format_report(cls.summary)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_domain_step_reads_every_run(self):
        summary = self.summary
        self.assertEqual(summary["domain"]["runs"], 4)

    def test_boundary_is_the_last_unflagged_angle(self):
        summary = self.summary
        values = [c["min"] for c in summary["domain"]["conditions"]]
        self.assertEqual(values, [4.0])

    def test_dataset_step_counts_rows(self):
        summary = self.summary
        self.assertEqual(summary["dataset"]["rows"], 16)

    def test_summary_carries_every_step(self):
        summary = self.summary
        for key in ("domain", "dataset", "sufficiency", "holdout"):
            self.assertIn(key, summary)

    def test_report_names_the_steps(self):
        text = self.report
        for fragment in ("步骤 1", "步骤 2", "步骤 3", "步骤 4"):
            self.assertIn(fragment, text)

    def test_analysis_names_the_batch_it_read(self):
        self.assertEqual(self.summary["batch"], str(self.root / "uiuc_test"))


class DataAuditTest(unittest.TestCase):
    """A pair of numbers the solver accepted is not a pair that is true.

    Re and Mach are independent, so a run can hold a combination no
    chord and fluid could produce. XFOIL solves it anyway, which is why
    the pairing has to be checked rather than assumed.
    """

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def _write(self, name, *, re, mach, assumptions):
        folder = self.root / name
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(
            json.dumps(
                {
                    "config": {
                        "re": re,
                        "mach": mach,
                        "flow_assumptions": assumptions,
                    },
                }
            )
        )

    def test_a_consistent_pair_is_not_reported(self):
        self._write(
            "ok",
            re=698_895,
            mach=0.1,
            assumptions={
                "chord_m": 0.3,
                "air_density": 1.225,
                "air_viscosity": 1.7894e-05,
                "sound_speed": 340.3,
            },
        )
        document = audit_data.audit(self.root)
        self.assertEqual(document["mismatched"], [])

    def test_a_pair_no_fluid_could_produce_is_reported(self):
        self._write(
            "bad",
            re=500_000,
            mach=0.1,
            assumptions={
                "chord_m": 0.3,
                "air_density": 1.225,
                "air_viscosity": 2e-05,
                "sound_speed": 340.3,
            },
        )
        document = audit_data.audit(self.root)
        self.assertEqual(len(document["mismatched"]), 1)
        found = document["mismatched"][0]
        self.assertAlmostEqual(found["implied_re"], 625_301, delta=2)
        self.assertGreater(found["off_by_percent"], 20)

    def test_a_run_without_assumptions_is_counted_not_guessed(self):
        self._write("bare", re=500_000, mach=0.1, assumptions={})
        document = audit_data.audit(self.root)
        self.assertEqual(document["unrecorded"], 1)
        self.assertEqual(document["mismatched"], [])


class ReportFormatTest(unittest.TestCase):
    def test_missing_interval_renders_as_na(self):
        self.assertEqual(analyse_batch._format_interval({}), "n/a")
        self.assertEqual(analyse_batch._format_interval({"mae": None}), "n/a")

    def test_point_estimate_without_bounds_still_renders(self):
        rendered = analyse_batch._format_interval(
            {"mae": 0.5, "low": None, "high": None}
        )
        self.assertEqual(rendered, "0.50000")

    def test_interval_renders_both_bounds(self):
        rendered = analyse_batch._format_interval(
            {"mae": 0.5, "low": 0.4, "high": 0.6}
        )
        self.assertIn("0.40000", rendered)
        self.assertIn("0.60000", rendered)

    def test_a_report_without_a_holdout_says_so(self):
        text = analyse_batch.format_report(
            {
                "batch": "x",
                "seconds": 1.0,
                "holdout": {"reported": False, "reason": "too few runs"},
            }
        )
        self.assertIn("未出报告", text)
        self.assertIn("too few runs", text)

    def test_warnings_are_listed(self):
        text = analyse_batch.format_report(
            {
                "batch": "x",
                "seconds": 1.0,
                "holdout": {},
                "warnings": ["something to note"],
            }
        )
        self.assertIn("something to note", text)


if __name__ == "__main__":
    unittest.main()
