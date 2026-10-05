import copy
import unittest

import numpy as np

from xfoil_mac.wing_geometry import (
    section_profile,
    section_ring,
    surface_lofts,
    loft_triangles,
)


class WingGeometryTest(unittest.TestCase):
    def section(self):
        return dict(x=0, y=0, z=0, chord=1, twist=0, naca="0012")

    def test_naca0012_has_twelve_percent_thickness_and_finite_trailing_edge(
        self,
    ):
        points = section_profile(self.section(), samples=101)
        upper, lower = points[:101][::-1], points[100:]
        np.testing.assert_allclose(upper[:, 0], lower[:, 0])
        np.testing.assert_allclose(upper[:, 1], -lower[:, 1])
        self.assertAlmostEqual(
            np.max(upper[:, 1] - lower[:, 1]), 0.12, delta=0.0001
        )
        self.assertAlmostEqual(points[0, 1] - points[-1, 1], 0.00252)
        np.testing.assert_allclose(points[100], [0, 0])

    def test_cambered_and_file_sections_keep_their_own_profiles(self):
        points = section_profile(
            dict(self.section(), naca="2412"), samples=101
        )
        upper, lower = points[:101][::-1], points[100:]
        self.assertAlmostEqual(
            np.max((upper[:, 1] + lower[:, 1]) / 2), 0.02, delta=0.0001
        )
        section = dict(
            self.section(),
            coordinates=[
                (1, 0.01),
                (0.5, 0.1),
                (0, 0),
                (0.5, -0.04),
                (1, -0.01),
            ],
        )
        del section["naca"]
        points = section_profile(section, samples=3)
        self.assertAlmostEqual(points[1, 1], 0.1)
        self.assertAlmostEqual(points[3, 1], -0.04)

    def test_loft_preserves_span_chord_mirror_washout_and_input(
        self,
    ):
        surface = dict(
            mirror=True,
            sections=[self.section(), dict(self.section(), y=4, twist=-1)],
        )
        before = copy.deepcopy(surface)
        right, left = surface_lofts(surface)
        self.assertEqual(surface, before)
        np.testing.assert_allclose(left, right * np.array([1, -1, 1]))
        self.assertEqual(np.max(right[:, :, 1]) - np.min(left[:, :, 1]), 8)
        trailing_midpoint = (right[-1, 0] + right[-1, -1]) / 2 - [0, 4, 0]
        self.assertAlmostEqual(np.linalg.norm(trailing_midpoint), 1)
        self.assertAlmostEqual(
            np.degrees(
                np.arctan2(-trailing_midpoint[2], trailing_midpoint[0])
            ),
            -1,
        )
        for loft, reflected in ((right, False), (left, True)):
            face = loft_triangles(loft, reflected)[0]
            self.assertGreater(
                np.cross(face[1] - face[0], face[2] - face[0])[2], 0
            )

    def test_vertical_surface_thickness_follows_its_local_normal(self):
        axis = np.array([5.0, 0, -1])
        ring = section_ring(self.section(), axis)
        np.testing.assert_array_equal(axis, [5, 0, -1])
        self.assertGreater(np.ptp(ring[:, 1]), 0.119)
        np.testing.assert_allclose(ring[:, 2], 0)


if __name__ == "__main__":
    unittest.main()
