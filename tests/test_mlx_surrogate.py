"""Tests for the differentiable MLX ensemble surrogate and inverse tools."""

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from xfoil_mac.analysis import mlx_surrogate

SHAPE_ROOT = Path(tempfile.mkdtemp(prefix="mlx_shapes_"))
HAS_MLX = mlx_surrogate.mlx_version() is not None


def _write_contour(name: str, thickness: float, camber: float) -> Path:
    count = 30
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
    SHAPE_ROOT.mkdir(parents=True, exist_ok=True)
    path = SHAPE_ROOT / f"{name}.dat"
    path.write_text(
        "\n".join(f"{x:.7f} {y:.7f}" for x, y in upper + lower) + "\n"
    )
    return path


CONDITIONS = (
    (0.09, 0.26, 0.00, 2.0e5, 0.05),
    (0.10, 0.27, 0.01, 5.0e5, 0.10),
    (0.11, 0.28, 0.02, 1.0e6, 0.15),
    (0.12, 0.29, 0.03, 2.0e6, 0.20),
    (0.13, 0.30, 0.04, 3.5e5, 0.25),
    (0.14, 0.31, 0.05, 8.0e5, 0.30),
    (0.15, 0.32, 0.06, 1.5e6, 0.35),
    (0.16, 0.33, 0.07, 3.0e6, 0.40),
)


def sufficient_rows():
    rows = []
    for index, (t, tx, cam, re, mach) in enumerate(CONDITIONS):
        name = f"af{t:.3f}_{cam:.3f}".replace(".", "p")
        _write_contour(name, t, cam)
        for alpha in (-2.0, 0.0, 2.0, 4.0):
            cl = 0.11 * alpha + 8.0 * cam
            cd = 0.007 + 0.015 * (cl**2) + 0.02 * t
            cm = -0.02 - 1.5 * cam
            rows.append(
                {
                    "run_id": f"run_{index:02d}",
                    "airfoil": name,
                    "verdict": "ok",
                    "thickness_ratio": t,
                    "thickness_x": tx,
                    "max_camber": cam,
                    "re": re,
                    "mach": mach,
                    "alpha": alpha,
                    "CL": cl,
                    "CD": cd,
                    "CM": cm,
                }
            )
    return rows


def one_condition_rows():
    t, cam = 0.12, 0.02
    name = "af0p120_0p020"
    _write_contour(name, t, cam)
    return [
        {
            "run_id": "run_00",
            "airfoil": name,
            "verdict": "ok",
            "thickness_ratio": t,
            "thickness_x": 0.30,
            "max_camber": cam,
            "re": 5.0e5,
            "mach": 0.10,
            "alpha": alpha,
            "CL": 0.1 * alpha,
            "CD": 0.01,
            "CM": -0.03,
        }
        for alpha in (-4.0, -2.0, 0.0, 2.0, 4.0, 6.0)
    ]


class LazyImportMlxTest(unittest.TestCase):
    def test_importing_the_module_does_not_load_mlx_or_numpy(self):
        script = (
            "import sys; "
            "import xfoil_mac.analysis.mlx_surrogate; "
            "print('mlx' in sys.modules, 'numpy' in sys.modules)"
        )
        outcome = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        self.assertEqual(outcome.returncode, 0, outcome.stderr)
        self.assertEqual(outcome.stdout.strip(), "False False")


class AirfoilSplitTest(unittest.TestCase):
    def test_split_airfoils_keeps_geometries_disjoint(self):
        rows = sufficient_rows()
        # Add a second flow condition for the same airfoils
        rows_extra = [
            dict(r, run_id=f"{r['run_id']}_b2", re=r["re"] * 1.5)
            for r in rows
        ]
        partition = mlx_surrogate.split_airfoils(rows + rows_extra, seed=42)
        train_afs = {r["airfoil"] for r in partition["train"]}
        test_afs = {r["airfoil"] for r in partition["test"]}
        self.assertEqual(train_afs & test_afs, set())
        self.assertTrue(len(test_afs) >= 1)


@unittest.skipUnless(HAS_MLX, "mlx is not installed in this environment")
class MlxSurrogateEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = sufficient_rows()
        cls.fitted = mlx_surrogate.fit(
            cls.rows,
            ensemble_size=2,
            steps=120,
            hidden_dim=64,
            batch_size=16,
            shape_root=SHAPE_ROOT,
        )

    def test_refuses_single_condition(self):
        outcome = mlx_surrogate.fit(
            one_condition_rows(), shape_root=SHAPE_ROOT
        )
        self.assertFalse(outcome["fitted"])
        self.assertIn("not attempted", outcome["reason"])

    def test_fit_save_and_load_roundtrip(self):
        self.assertTrue(self.fitted["fitted"], self.fitted.get("reason"))
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "surrogate_mlx.npz"
            written = mlx_surrogate.save(self.fitted, dest)
            self.assertTrue(Path(written["model"]).is_file())
            meta = json.loads(Path(written["metadata"]).read_text())
            self.assertEqual(meta["kind"], "mlx_ensemble")
            loaded = mlx_surrogate.load(dest)
            self.assertTrue(loaded["version_matches"])
            preds = mlx_surrogate.predict(
                loaded,
                airfoil=self.rows[0]["airfoil"],
                re=5.0e5,
                mach=0.10,
                alphas=[0.0, 2.0],
                shape_root=SHAPE_ROOT,
            )
            self.assertEqual(len(preds), 2)
            for row in preds:
                self.assertGreater(row["CD"], 0.0)
                self.assertGreaterEqual(row["CD_std"], 0.0)
                self.assertIn("ood_distance", row)

    def test_solve_alpha_for_cl_and_shape_sensitivity(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "surrogate_mlx.npz"
            mlx_surrogate.save(self.fitted, dest)
            loaded = mlx_surrogate.load(dest)
            inv = mlx_surrogate.solve_alpha_for_cl(
                loaded,
                target_cl=0.25,
                airfoil=self.rows[0]["airfoil"],
                re=5.0e5,
                mach=0.10,
                max_iter=6,
                shape_root=SHAPE_ROOT,
            )
            self.assertIn("alpha", inv)
            self.assertIn("dCL_dalpha", inv)

            sens = mlx_surrogate.shape_sensitivity(
                loaded,
                airfoil=self.rows[0]["airfoil"],
                re=5.0e5,
                mach=0.10,
                alpha=2.0,
                shape_root=SHAPE_ROOT,
            )
            self.assertEqual(len(sens["sensitivities"]), 22)

    def test_optimize_coordinates_writes_valid_contour(self):
        with tempfile.TemporaryDirectory() as folder:
            dest_model = Path(folder) / "surrogate_mlx.npz"
            mlx_surrogate.save(self.fitted, dest_model)
            loaded = mlx_surrogate.load(dest_model)
            source = SHAPE_ROOT / f"{self.rows[0]['airfoil']}.dat"
            out_dat = Path(folder) / "optimized.dat"
            outcome = mlx_surrogate.optimize_coordinates(
                loaded,
                coordinates=source,
                destination=out_dat,
                re=5.0e5,
                mach=0.10,
                alpha=2.0,
                steps=4,
            )
            self.assertTrue(out_dat.is_file())
            self.assertIn("baseline", outcome)
            self.assertIn("optimized", outcome)


if __name__ == "__main__":
    unittest.main()
