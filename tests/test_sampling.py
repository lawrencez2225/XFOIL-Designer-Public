"""Price a sweep from measured work instead of a uniform assumption."""

import unittest

from xfoil_mac.analysis.sampling import (
    DEFAULT_DOMAIN_LIMIT,
    conservative_domain_limit,
    domain_limits_from_measurements,
    compare_to_uniform,
    domain_limit,
    estimate_iterations,
    grid_angles,
    plan_sweep,
)


def point(alpha, verdict="ok", converged=True, iterations=None):
    return {
        "alpha": float(alpha),
        "verdict": verdict,
        "converged": converged,
        "iterations": iterations,
    }


def evidence():
    """A sweep whose upper end is flagged, with measured iteration counts."""
    points = []
    for alpha in range(-4, 11):
        verdict = "suspect" if alpha >= 8 else "ok"
        points.append(
            point(alpha, verdict, iterations=(20 + 6 * alpha, 40 + 5 * alpha))
        )
    return points


class GridAnglesTest(unittest.TestCase):
    def test_includes_both_ends(self):
        self.assertEqual(grid_angles(-4, 10, 2), [-4, -2, 0, 2, 4, 6, 8, 10])

    def test_appends_stop_when_step_does_not_land_on_it(self):
        self.assertEqual(grid_angles(0, 10, 4), [0, 4, 8, 10])

    def test_empty_when_stop_precedes_start(self):
        self.assertEqual(grid_angles(4, -4, 1), [])

    def test_single_angle_when_range_collapses(self):
        self.assertEqual(grid_angles(3, 3, 1), [3])

    def test_rejects_non_positive_step(self):
        with self.assertRaises(ValueError):
            grid_angles(0, 1, 0)


class DomainLimitTest(unittest.TestCase):
    def test_last_trusted_stops_before_the_first_flagged_angle(self):
        report = domain_limit(evidence())
        self.assertEqual(report["last_trusted"], 7.0)
        self.assertEqual(report["first_suspect"], 8.0)

    def test_failed_solve_is_reported_separately(self):
        points = [
            point(0),
            point(1),
            point(2, converged=False),
            point(3),
        ]
        report = domain_limit(points)
        self.assertEqual(report["first_failed"], 2.0)

    def test_no_evidence_reports_no_limit(self):
        report = domain_limit([])
        self.assertIsNone(report["last_trusted"])
        self.assertFalse(report["evidence"])


class EstimateIterationsTest(unittest.TestCase):
    def test_measured_range_is_interpolated(self):
        measured = [(0.0, 40.0), (10.0, 100.0)]
        self.assertAlmostEqual(
            estimate_iterations(5.0, measured=measured, low=0, high=10),
            70.0,
        )

    def test_beyond_measured_range_follows_the_measured_slope(self):
        measured = [(0.0, 40.0), (10.0, 100.0)]
        value = estimate_iterations(12.0, measured=measured, low=0, high=10)
        self.assertAlmostEqual(value, 112.0)

    def test_without_measurement_uses_the_documented_ramp(self):
        low = estimate_iterations(0.0, low=0.0, high=10.0)
        high = estimate_iterations(10.0, low=0.0, high=10.0)
        self.assertLess(low, high)


class PlanSweepTest(unittest.TestCase):
    def test_default_plan_stops_at_the_default_limit(self):
        plan = plan_sweep()
        self.assertEqual(plan["stop_angle"], DEFAULT_DOMAIN_LIMIT)
        self.assertTrue(plan["evidence"].get("evidence") is False)

    def test_measured_evidence_sets_the_stopping_angle(self):
        plan = plan_sweep(points=evidence())
        self.assertEqual(plan["stop_angle"], 7.0)
        self.assertTrue(all(a <= 7.0 for a in plan["angles"]))

    def test_grid_tightens_from_the_split_angle(self):
        plan = plan_sweep(points=evidence(), split=4.0)
        names = [region["name"] for region in plan["regions"]]
        self.assertEqual(names, ["coarse", "fine"])
        steps = {region["name"]: region["step"] for region in plan["regions"]}
        self.assertLess(steps["fine"], steps["coarse"])
        dense = [a for a in plan["angles"] if a >= 4.0]
        gaps = [b - a for a, b in zip(dense, dense[1:])]
        self.assertTrue(all(gap <= 0.5 + 1e-9 for gap in gaps))

    def test_angles_are_sorted_and_unique(self):
        plan = plan_sweep(points=evidence())
        self.assertEqual(plan["angles"], sorted(set(plan["angles"])))

    def test_empty_trusted_range_plans_nothing(self):
        points = [point(12, verdict="suspect")]
        plan = plan_sweep(points=points)
        self.assertEqual(plan["angles"], [])
        self.assertEqual(plan["total_iterations"], 0.0)
        self.assertIsNone(plan["stop_angle"])
        self.assertTrue(
            any("no measured angle" in note for note in plan["notes"]),
            plan["notes"],
        )

    def test_evidence_outranks_the_default_limit(self):
        """A rejected sweep must not fall back to the default ceiling."""
        plan = plan_sweep(points=[point(12, verdict="suspect")])
        self.assertEqual(plan["angles"], [])
        self.assertLessEqual(plan["alpha_max"], DEFAULT_DOMAIN_LIMIT * 2)

    def test_total_price_matches_the_sum_of_its_points(self):
        plan = plan_sweep(points=evidence())
        self.assertAlmostEqual(plan["total_iterations"], sum(plan["costs"]))

    def test_short_fine_region_is_called_out(self):
        plan = plan_sweep(points=evidence())
        self.assertTrue(
            any("short" in note for note in plan["notes"]), plan["notes"]
        )

    def test_flagged_and_failed_angles_are_reported(self):
        points = evidence() + [point(11, converged=False)]
        plan = plan_sweep(points=points)
        self.assertTrue(
            any("flagged" in note for note in plan["notes"]), plan["notes"]
        )
        self.assertTrue(
            any("failed" in note for note in plan["notes"]), plan["notes"]
        )


class DomainLimitEstimateTest(unittest.TestCase):
    """A plan for an unsolved airfoil must not reach past an observed limit."""

    def measured(self):
        return [
            {
                "airfoil": "thick",
                "re": 500000.0,
                "mach": 0.10,
                "last_trusted_alpha": 12.5,
            },
            {
                "airfoil": "thin",
                "re": 500000.0,
                "mach": 0.10,
                "last_trusted_alpha": 9.0,
            },
            {
                "airfoil": "other",
                "re": 2000000.0,
                "mach": 0.25,
                "last_trusted_alpha": 11.5,
            },
        ]

    def test_limits_are_grouped_by_condition(self):
        measured = domain_limits_from_measurements(self.measured())
        self.assertEqual(len(measured), 2)
        first = measured[0]
        self.assertEqual(first["airfoils"], 2)
        self.assertEqual(first["min"], 9.0)
        self.assertEqual(first["max"], 12.5)

    def test_rows_without_a_boundary_are_ignored(self):
        measured = domain_limits_from_measurements(
            self.measured()
            + [
                {
                    "airfoil": "x",
                    "re": 500000.0,
                    "mach": 0.10,
                    "last_trusted_alpha": None,
                }
            ]
        )
        self.assertEqual(measured[0]["airfoils"], 2)

    def test_per_angle_evidence_is_also_accepted(self):
        rows = [
            {
                "run_id": "r",
                "re": 5e5,
                "mach": 0.1,
                "alpha": 4.0,
                "verdict": "ok",
            },
            {
                "run_id": "r",
                "re": 5e5,
                "mach": 0.1,
                "alpha": 8.0,
                "verdict": "ok",
            },
            {
                "run_id": "r",
                "re": 5e5,
                "mach": 0.1,
                "alpha": 12.0,
                "verdict": "suspect",
            },
        ]
        measured = domain_limits_from_measurements(rows)
        self.assertEqual(len(measured), 1)
        self.assertEqual(measured[0]["max"], 8.0)

    def test_estimate_uses_the_smallest_nearby_limit(self):
        measured = domain_limits_from_measurements(self.measured())
        estimate = conservative_domain_limit(measured, re=500000.0, mach=0.10)
        self.assertEqual(estimate["matched"], 1)
        self.assertEqual(estimate["limit"], 9.0)

    def test_estimate_reports_the_spread_it_saw(self):
        measured = domain_limits_from_measurements(self.measured())
        estimate = conservative_domain_limit(measured, re=500000.0, mach=0.10)
        self.assertEqual(estimate["spread"], (9.0, 12.5))

    def test_a_distant_condition_falls_back_to_the_default(self):
        measured = domain_limits_from_measurements(self.measured())
        estimate = conservative_domain_limit(
            measured, re=5.0e7, mach=0.6, fallback=11.0
        )
        self.assertEqual(estimate["matched"], 0)
        self.assertEqual(estimate["limit"], 11.0)

    def test_no_measurements_at_all_falls_back(self):
        estimate = conservative_domain_limit([], fallback=11.0)
        self.assertEqual(estimate["limit"], 11.0)
        self.assertEqual(estimate["matched"], 0)


class DomainAwarePlanTest(unittest.TestCase):
    def measured(self):
        return domain_limits_from_measurements(
            [
                {
                    "airfoil": "thin",
                    "re": 500000.0,
                    "mach": 0.10,
                    "last_trusted_alpha": 9.0,
                },
                {
                    "airfoil": "thick",
                    "re": 500000.0,
                    "mach": 0.10,
                    "last_trusted_alpha": 12.5,
                },
            ]
        )

    def test_plan_without_a_domain_is_unchanged(self):
        plan = plan_sweep()
        self.assertFalse(plan["population"]["used"])
        self.assertEqual(plan["stop_angle"], DEFAULT_DOMAIN_LIMIT)

    def test_plan_with_a_domain_stops_at_the_smallest_measured(self):
        plan = plan_sweep(domain=self.measured(), re=500000.0, mach=0.10)
        self.assertTrue(plan["population"]["used"])
        self.assertEqual(plan["stop_angle"], 9.0)
        self.assertTrue(all(a <= 9.0 for a in plan["angles"]))

    def test_sweep_evidence_outranks_the_population(self):
        plan = plan_sweep(
            points=evidence(),
            domain=self.measured(),
            re=500000.0,
            mach=0.10,
        )
        self.assertFalse(plan["population"]["used"])
        self.assertEqual(plan["stop_angle"], 7.0)


class CompareToUniformTest(unittest.TestCase):
    def test_denser_uniform_grid_costs_more(self):
        plan = plan_sweep(points=evidence())
        fine = compare_to_uniform(plan, 0.25)
        coarse = compare_to_uniform(plan, 2.0)
        self.assertGreater(
            fine["total_iterations"], coarse["total_iterations"]
        )

    def test_same_grid_price_is_reported_consistently(self):
        plan = plan_sweep(points=evidence())
        strict = compare_to_uniform(plan, 0.05)
        loose = compare_to_uniform(plan, 5.0)
        self.assertGreaterEqual(
            strict["total_iterations"], loose["total_iterations"]
        )
        self.assertGreater(len(strict["angles"]), len(loose["angles"]))


if __name__ == "__main__":
    unittest.main()
