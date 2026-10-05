import math
import tempfile
import unittest
from pathlib import Path

from xfoil_mac import plotting


class PlotOutputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = [
            (1.0, 0.05),
            (0.5, 0.08),
            (0.0, 0.0),
            (0.5, -0.05),
            (1.0, 0.0),
        ]

    def test_pressure_vectors_follow_signed_xfoil_direction(self) -> None:
        cp_rows = [
            (1.0, -1.0),
            (0.5, -2.0),
            (0.0, -1.0),
            (0.5, 1.0),
            (1.0, 0.0),
        ]
        rotated, xs, ys, us, vs = plotting.pressure_vector_components(
            cp_rows,
            self.geometry,
            alpha=0.0,
            max_vectors=85,
        )

        upper = 0
        self.assertAlmostEqual(xs[upper], rotated[1][0])
        self.assertAlmostEqual(ys[upper], rotated[1][1])
        self.assertGreater(vs[upper], 0.0)

        lower = 2
        self.assertLess(ys[lower], rotated[3][1])
        self.assertGreater(vs[lower], 0.0)
        self.assertAlmostEqual(xs[lower] + us[lower], rotated[3][0])
        self.assertAlmostEqual(ys[lower] + vs[lower], rotated[3][1])

    def test_pressure_vector_geometry_rotates_clockwise_with_alpha(
        self,
    ) -> None:
        rotated, *_ = plotting.pressure_vector_components(
            [(x, -1.0) for x, _ in self.geometry],
            self.geometry,
            alpha=90.0,
            max_vectors=85,
        )

        self.assertAlmostEqual(rotated[-1][0], 0.0, places=7)
        self.assertAlmostEqual(rotated[-1][1], -1.0, places=7)

    def test_lift_to_drag_omits_zero_and_nonfinite_drag(self) -> None:
        rows = [
            {"alpha": 0.0, "CL": 0.2, "CD": 0.01},
            {"alpha": 1.0, "CL": 0.3, "CD": 0.0},
            {"alpha": 2.0, "CL": math.inf, "CD": 0.02},
        ]

        self.assertEqual(
            plotting.finite_lift_to_drag_rows(rows), [(0.0, 20.0)]
        )

    def test_dense_cp_labels_use_separate_staggered_columns(self) -> None:
        surface_series = []
        for alpha in range(-5, 13):
            surface_series.extend(
                [
                    {
                        "alpha": alpha,
                        "surface": "upper",
                        "points": [(0.18, -1.0 + alpha * 0.01)],
                        "color": "blue",
                    },
                    {
                        "alpha": alpha,
                        "surface": "lower",
                        "points": [(0.18, alpha * 0.01)],
                        "color": "red",
                    },
                ]
            )

        upper = plotting.cp_alpha_label_positions(
            surface_series, label_surface="upper"
        )
        lower = plotting.cp_alpha_label_positions(
            surface_series, label_surface="lower"
        )
        upper_columns = sorted({label["xytext"][0] for label in upper})
        lower_columns = sorted({label["xytext"][0] for label in lower})

        self.assertEqual(len(upper_columns), 2)
        self.assertEqual(len(lower_columns), 2)
        self.assertLess(max(upper_columns), min(lower_columns))
        for labels, columns in (
            (upper, upper_columns),
            (lower, lower_columns),
        ):
            for column in columns:
                y_values = sorted(
                    label["xytext"][1]
                    for label in labels
                    if label["xytext"][0] == column
                )
                self.assertTrue(
                    all(
                        next_y - y >= 0.159
                        for y, next_y in zip(y_values, y_values[1:])
                    )
                )

    def test_result_plots_are_six_standalone_pngs(self) -> None:
        cp_rows = [
            (1.0, -0.1),
            (0.5, -1.0),
            (0.0, -0.5),
            (0.5, 0.2),
            (1.0, 0.0),
        ]
        polar_rows = [{"alpha": 5.0, "CL": 0.84, "CD": 0.0066, "CM": -0.052}]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs = plotting.write_cp_polar_outputs(
                [(5.0, cp_rows)],
                self.geometry,
                polar_rows,
                root,
                "NACA2412",
                save_pressure_vectors=False,
            )
            names = sorted(path.name for path in outputs["result_plots"])

            self.assertEqual(
                names,
                sorted(
                    [
                        "cp_distribution.png",
                        "cl_vs_alpha.png",
                        "cd_vs_alpha.png",
                        "cm_vs_alpha.png",
                        "cl_vs_cd.png",
                        "lift_to_drag_vs_alpha.png",
                    ]
                ),
            )
            self.assertTrue(all((root / name).is_file() for name in names))
            self.assertFalse((root / "cp_polar_overview.png").exists())


if __name__ == "__main__":
    unittest.main()
