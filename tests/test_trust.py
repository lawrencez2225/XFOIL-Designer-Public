"""Verdicts come from separation topology, never from one shape factor."""

import unittest

from xfoil_mac.analysis.trust import (
    EXTREME_SHAPE_FACTOR,
    split_surfaces,
    summarise_points,
    verdicts_from_records,
    worst_surface,
    MIN_RUNS_FLAG,
    point_verdict,
    surface_evidence,
)


def station(x, Cf, H=2.0):
    return {
        "s": x,
        "x": x,
        "y": 0.0,
        "Ue_Vinf": 1.0,
        "Dstar": 0.01,
        "Theta": 0.005,
        "Cf": Cf,
        "H": H,
    }


def _upper_x(fraction):
    """Chord fraction on the upper surface, running aft to the LE.

    Stations must fall to a single minimum at the leading edge, which
    is how the judge identifies the two surfaces. A monotonic ramp
    has its minimum at an end and is rejected as a malformed contour.
    """
    return 1.0 - 0.98 * fraction


def _lower_x(fraction):
    return 0.02 + 0.98 * fraction


def half(segments, *, lower=False):
    """Stations for one surface, from (count, Cf, H) triples.

    The upper surface runs from the trailing edge down to the leading
    edge; the lower surface runs from the leading edge back to the
    trailing edge. The judge finds the leading edge as the single
    minimum in x, so both directions matter.
    """
    total = sum(count for count, _, _ in segments)
    stations = []
    position = 0
    for count, Cf, H in segments:
        for _ in range(count):
            fraction = position / max(total - 1, 1)
            x = 0.02 + 0.98 * fraction if lower else 1.0 - 0.98 * fraction
            record = station(x, Cf, H)
            record["s"] = x
            stations.append(record)
            position += 1
    return stations


def surface(upper, lower):
    """A full contour: upper trailing edge through to lower trailing.

    The leading edge is kept once, not once per surface.
    """
    top = half(upper)
    bottom = half(lower, lower=True)[1:]
    return top + bottom


ATTACHED = [(20, 0.002, 2.0)]
"""A lower surface with no separation, for judging the upper one."""


def verdict_of(upper, lower=None):
    """Judge a contour built from a separated upper surface."""
    top, bottom = split_surfaces(surface(upper, lower or ATTACHED))
    evidence = worst_surface(surface_evidence(top), surface_evidence(bottom))
    return point_verdict(evidence=evidence)


class ShapeFactorIsNotAVerdictTest(unittest.TestCase):
    """A single separated run must not be failed for a large shape factor.

    A gate on peak shape factor was removed after it was measured to flag
    4.5..5.0 deg on NACA 0012 while leaving 5.5..8.0 deg unflagged. No
    domain edge is non-monotonic in angle, so the gate was describing
    trailing-edge thickness rather than the solver leaving its domain.
    """

    def test_one_run_with_an_extreme_shape_factor_still_passes(self):
        verdict, reasons = verdict_of(
            [(16, 0.002, 2.0), (4, -0.001, EXTREME_SHAPE_FACTOR * 2)]
        )
        self.assertEqual(verdict, "ok", reasons)

    def test_extreme_shape_factor_is_never_listed_as_a_reason(self):
        _, reasons = verdict_of(
            [(16, 0.002, 2.0), (4, -0.001, EXTREME_SHAPE_FACTOR * 3)]
        )
        self.assertNotIn("extreme_shape_factor", reasons)

    def test_shape_factor_is_still_recorded_as_evidence(self):
        evidence = surface_evidence(surface([(5, 0.002, 12.5)], ATTACHED))
        self.assertEqual(evidence["max_h"], 12.5)

    def test_a_thin_airfoil_trailing_edge_state_passes(self):
        """The measured false positive: one run, H near 9, at x near 1."""
        verdict, reasons = verdict_of([(17, 0.002, 2.0), (3, -0.0002, 9.354)])
        self.assertEqual(verdict, "ok", reasons)


class RecordsApiTest(unittest.TestCase):
    """Judging from records already in memory, not from a directory.

    A caller that has read the dumps should not have to write them back
    out for the judgement to run, and the on-disk layout is not the
    judgement's business.
    """

    def test_a_record_set_yields_one_verdict_per_angle(self):
        first = surface(
            [(16, 0.002, 2.0), (4, -0.001, 3.0)], [(20, 0.002, 2.0)]
        )
        second = surface(
            [(2, -0.001, 3.0), (16, 0.002, 2.0), (2, -0.001, 3.0)],
            [(20, 0.002, 2.0)],
        )
        points = verdicts_from_records({0.0: first, 12.0: second})
        verdicts = {p["alpha"]: p["verdict"] for p in points}
        self.assertEqual(verdicts[0.0], "ok")
        self.assertEqual(verdicts[12.0], "suspect")

    def test_a_failed_angle_without_records_is_kept(self):
        attached = surface([(20, 0.002, 2.0)], [(20, 0.002, 2.0)])
        # Converged neighbours within ISOLATED_NEIGHBOUR_SPAN are what make
        # a lone failure a retry candidate rather than an edge of the
        # trusted range.
        points = verdicts_from_records(
            {3.0: attached, 5.0: attached}, failed_alphas=[4.0]
        )
        verdicts = {p["alpha"]: p["verdict"] for p in points}
        self.assertEqual(len(points), 3)
        self.assertEqual(verdicts[4.0], "retry")

    def test_a_failure_far_from_any_neighbour_stays_invalid(self):
        """Beyond the neighbour span a failure is left as a domain edge."""
        attached = surface([(20, 0.002, 2.0)], [(20, 0.002, 2.0)])
        points = verdicts_from_records(
            {0.0: attached, 8.0: attached}, failed_alphas=[4.0]
        )
        verdicts = {p["alpha"]: p["verdict"] for p in points}
        self.assertEqual(verdicts[4.0], "invalid")

    def test_an_angle_with_no_records_is_unchecked(self):
        points = verdicts_from_records({0.0: []})
        self.assertEqual(points[0]["verdict"], "unchecked")

    def test_summary_counts_each_verdict(self):
        first = surface(
            [(16, 0.002, 2.0), (4, -0.001, 3.0)], [(20, 0.002, 2.0)]
        )
        second = surface(
            [(2, -0.001, 3.0), (16, 0.002, 2.0), (2, -0.001, 3.0)],
            [(20, 0.002, 2.0)],
        )
        summary = summarise_points(
            verdicts_from_records({0.0: first, 12.0: second})
        )
        self.assertEqual(summary["counts"], {"ok": 1, "suspect": 1})
        self.assertEqual(summary["points"], 2)

    def test_summary_reports_the_last_trusted_angle(self):
        first = surface(
            [(16, 0.002, 2.0), (4, -0.001, 3.0)], [(20, 0.002, 2.0)]
        )
        second = surface(
            [(2, -0.001, 3.0), (16, 0.002, 2.0), (2, -0.001, 3.0)],
            [(20, 0.002, 2.0)],
        )
        summary = summarise_points(
            verdicts_from_records({4.0: first, 8.0: second})
        )
        self.assertEqual(summary["last_trusted_alpha"], 4.0)
        self.assertEqual(summary["first_suspect_alpha"], 8.0)

    def test_summary_carries_a_verdict_per_angle(self):
        points = verdicts_from_records(
            {0.0: surface([(20, 0.002, 2.0)], [(20, 0.002, 2.0)])}
        )
        summary = summarise_points(points)
        self.assertEqual(summary["by_alpha"][0.0], "ok")

    def test_an_empty_summary_reports_no_boundary(self):
        summary = summarise_points([])
        self.assertEqual(summary["points"], 0)
        self.assertIsNone(summary["last_trusted_alpha"])


class SeparationTopologyTest(unittest.TestCase):
    def test_fully_attached_surface_passes(self):
        verdict, reasons = verdict_of([(20, 0.002, 2.0)])
        self.assertEqual(verdict, "ok")
        self.assertEqual(reasons, [])

    def test_single_trailing_edge_run_passes(self):
        verdict, reasons = verdict_of([(16, 0.002, 2.0), (4, -0.001, 3.0)])
        self.assertEqual(verdict, "ok", reasons)

    def test_single_leading_edge_bubble_passes(self):
        verdict, reasons = verdict_of([(4, -0.001, 3.0), (16, 0.002, 2.0)])
        self.assertEqual(verdict, "ok", reasons)

    def test_runs_at_both_ends_are_flagged(self):
        verdict, reasons = verdict_of(
            [(2, -0.001, 3.0), (16, 0.002, 2.0), (2, -0.001, 3.0)]
        )
        self.assertEqual(verdict, "suspect")
        self.assertIn("simultaneous_separated_regions", reasons)

    def test_a_run_over_the_length_flag_is_flagged(self):
        # 8 of 20 stations separated is 40% of the chord.
        verdict, reasons = verdict_of([(12, 0.002, 2.0), (8, -0.001, 3.0)])
        self.assertEqual(verdict, "suspect")
        self.assertIn("large_separated_run", reasons)

    def test_a_run_under_the_length_flag_is_not_flagged(self):
        # 3 of 20 stations separated is 15% of the chord.
        verdict, reasons = verdict_of([(17, 0.002, 2.0), (3, -0.001, 3.0)])
        self.assertEqual(verdict, "ok", reasons)

    def test_run_count_matches_the_configured_flag(self):
        evidence = surface_evidence(
            surface(
                [(2, -0.001, 3.0), (16, 0.002, 2.0), (2, -0.001, 3.0)],
                [(20, 0.002, 2.0)],
            )
        )
        self.assertEqual(evidence["run_count"], MIN_RUNS_FLAG)


class UnassessableInputTest(unittest.TestCase):
    def test_no_boundary_layer_is_unchecked_not_passed(self):
        verdict, reasons = point_verdict(evidence={})
        self.assertEqual(verdict, "unchecked")
        self.assertIn("no_boundary_layer_evidence", reasons)

    def test_an_empty_surface_is_unchecked(self):
        verdict, _ = point_verdict(evidence=surface_evidence([]))
        self.assertEqual(verdict, "unchecked")

    def test_a_failed_solve_is_invalid_whatever_the_evidence(self):
        evidence = surface_evidence(surface([(20, 0.002, 2.0)], ATTACHED))
        verdict, reasons = point_verdict(evidence=evidence, converged=False)
        self.assertEqual(verdict, "invalid")
        self.assertEqual(reasons, ["not_converged"])


if __name__ == "__main__":
    unittest.main()
