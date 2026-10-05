"""Mach acceptance follows the code, not a remembered rule of thumb.

A message that suggested 0.3 as an example was read as a ceiling. The
ceiling in the code is Mach 1, and the solver applies a Karman-Tsien
compressibility correction. These tests pin the actual boundary so a
stray example value cannot be mistaken for a limit again.
"""

import math
import unittest

from xfoil_mac.runtime import (
    FlowAssumptions,
    estimate_mach_from_re,
    estimate_re_from_mach,
    validate_xfoil_mach,
)


class MachAcceptanceTest(unittest.TestCase):
    def test_zero_is_accepted(self):
        validate_xfoil_mach(0.0)

    def test_a_low_subsonic_value_is_accepted(self):
        validate_xfoil_mach(0.0588)

    def test_values_well_above_the_remembered_ceiling_are_accepted(self):
        """0.3 is an example in an error message, not a limit."""
        for mach in (0.3, 0.4, 0.5, 0.6):
            with self.subTest(mach=mach):
                validate_xfoil_mach(mach)

    def test_just_below_one_is_still_accepted(self):
        validate_xfoil_mach(0.99)
        validate_xfoil_mach(math.nextafter(1.0, 0.0))

    def test_one_and_above_are_rejected(self):
        for mach in (1.0, 1.2, 2.0):
            with self.subTest(mach=mach):
                with self.assertRaises(ValueError):
                    validate_xfoil_mach(mach)

    def test_negative_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_xfoil_mach(-0.1)

    def test_non_finite_is_rejected(self):
        for mach in (float("nan"), float("inf")):
            with self.subTest(mach=mach):
                with self.assertRaises(ValueError):
                    validate_xfoil_mach(mach)


class FlowAssumptionRoundTripTest(unittest.TestCase):
    """Re and Mach are linked by chord, density, viscosity and sound speed."""

    def setUp(self):
        self.assumptions = FlowAssumptions(chord_m=0.3)

    def test_mach_to_re_and_back(self):
        for mach in (0.0588, 0.10, 0.25, 0.50):
            with self.subTest(mach=mach):
                reynolds = estimate_re_from_mach(mach, self.assumptions)
                self.assertGreater(reynolds, 0)
                # estimate_re_from_mach rounds to an integer, so the
                # round trip is quantised at about 1e-7 in Mach.
                self.assertAlmostEqual(
                    estimate_mach_from_re(reynolds, self.assumptions),
                    mach,
                    delta=1e-6,
                )

    def test_a_longer_chord_raises_the_reynolds_number(self):
        short = FlowAssumptions(chord_m=0.3)
        long = FlowAssumptions(chord_m=0.6)
        self.assertGreater(
            estimate_re_from_mach(0.1, long),
            estimate_re_from_mach(0.1, short),
        )


if __name__ == "__main__":
    unittest.main()
