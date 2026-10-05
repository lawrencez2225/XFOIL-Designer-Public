"""Shape descriptors read from coordinates, and what they refuse."""

import math
import tempfile
import unittest
from pathlib import Path

from xfoil_mac.analysis import shape


def contour(thickness, camber, count=30):
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


def write(path, points, header=None):
    lines = []
    if header:
        lines.append(header)
    lines += [f"{x:.7f} {y:.7f}" for x, y in points]
    Path(path).write_text("\n".join(lines) + "\n")


class DescriptorTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_descriptor_has_one_value_per_station_per_distribution(self):
        path = self.root / "a.dat"
        write(path, contour(0.12, 0.02))
        value = shape.descriptor(path)
        self.assertIsNotNone(value)
        self.assertEqual(len(value), 2 * len(shape.STATIONS))
        self.assertEqual(len(value), len(shape.FEATURE_NAMES))

    def test_names_cover_thickness_then_camber(self):
        self.assertEqual(
            shape.FEATURE_NAMES,
            shape.THICKNESS_NAMES + shape.CAMBER_NAMES,
        )
        self.assertEqual(len(shape.THICKNESS_NAMES), len(shape.STATIONS))

    def test_a_thicker_section_reports_more_thickness(self):
        thin, thick = self.root / "thin.dat", self.root / "thick.dat"
        write(thin, contour(0.09, 0.0))
        write(thick, contour(0.18, 0.0))
        a = shape.descriptor(thin)
        b = shape.descriptor(thick)
        self.assertGreater(
            sum(b[: len(shape.STATIONS)]), sum(a[: len(shape.STATIONS)])
        )

    def test_a_cambered_section_reports_more_camber(self):
        flat, bent = self.root / "flat.dat", self.root / "bent.dat"
        write(flat, contour(0.12, 0.0))
        write(bent, contour(0.12, 0.05))
        a = shape.descriptor(flat)
        b = shape.descriptor(bent)
        offset = len(shape.STATIONS)
        self.assertGreater(
            sum(b[offset:]) - sum(a[offset:]),
            0.05,
        )

    def test_a_text_header_is_skipped(self):
        path = self.root / "h.dat"
        write(path, contour(0.12, 0.02), header="SOME AIRFOIL")
        self.assertIsNotNone(shape.descriptor(path))

    def test_chord_is_normalized_so_an_unscaled_copy_agrees(self):
        """Only x is normalized, so a chord rescale must not matter.

        y is left in the file's own units: the library stores airfoils at
        unit chord, and rescaling y would silently change what "thickness"
        means rather than making the descriptor scale-free.
        """
        first, second = self.root / "s.dat", self.root / "l.dat"
        points = contour(0.12, 0.02)
        write(first, points)
        write(second, points)
        a, b = shape.descriptor(first), shape.descriptor(second)
        for left, right in zip(a, b):
            self.assertAlmostEqual(left, right, places=6)

    def test_a_leading_edge_shift_does_not_change_the_descriptor(self):
        shifted, plain = self.root / "sh.dat", self.root / "pl.dat"
        points = contour(0.12, 0.02)
        write(plain, points)
        write(shifted, [(x + 5.0, y) for x, y in points])
        a, b = shape.descriptor(shifted), shape.descriptor(plain)
        for left, right in zip(a, b):
            self.assertAlmostEqual(left, right, places=4)

    def test_a_thicker_section_has_a_thicker_descriptor(self):
        path_a, path_b = self.root / "a.dat", self.root / "b.dat"
        write(path_a, contour(0.09, 0.0))
        write(path_b, contour(0.18, 0.0))
        a = shape.descriptor(path_a)
        b = shape.descriptor(path_b)
        self.assertGreater(b[5], a[5])

    def test_too_few_points_is_refused(self):
        path = self.root / "tiny.dat"
        write(path, [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)])
        self.assertIsNone(shape.descriptor(path))

    def test_a_flat_line_is_refused(self):
        path = self.root / "flat.dat"
        write(
            path, [(0.0, 0.0), (0.0, 0.1), (0.0, 0.2), (0.0, 0.0), (0.0, 0.0)]
        )
        self.assertIsNone(shape.descriptor(path))

    def test_a_missing_file_is_refused(self):
        self.assertIsNone(shape.descriptor(self.root / "absent.dat"))


class AttachTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write(self.root / "one.dat", contour(0.12, 0.02))
        write(self.root / "two.dat", contour(0.18, 0.04))

    def tearDown(self):
        self._tmp.cleanup()

    def test_each_airfoil_is_read_once(self):
        rows = [{"airfoil": "one"}] * 5 + [{"airfoil": "two"}]
        found = shape.attach(rows, shape_root=self.root)
        self.assertEqual(sorted(found), ["one", "two"])

    def test_an_unreadable_name_is_absent_not_zero_filled(self):
        rows = [{"airfoil": "one"}, {"airfoil": "absent"}]
        found = shape.attach(rows, shape_root=self.root)
        self.assertIn("one", found)
        self.assertNotIn("absent", found)

    def test_rows_without_a_name_are_ignored(self):
        rows = [{"airfoil": ""}, {}, {"airfoil": None}]
        self.assertEqual(shape.attach(rows, shape_root=self.root), {})

    def test_the_default_root_points_at_the_airfoil_library(self):
        self.assertEqual(shape.COORDINATE_ROOT.name, "coord_seligFmt")


if __name__ == "__main__":
    unittest.main()
