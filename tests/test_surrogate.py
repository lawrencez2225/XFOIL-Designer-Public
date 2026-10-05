"""A surrogate that states its limits instead of reporting a nice score."""

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from xfoil_mac.analysis import surrogate

SHAPE_ROOT = Path(tempfile.mkdtemp(prefix="shapes_"))


def _write_contour(name, thickness, camber):
    """A closed NACA-like contour, so a shape descriptor can be read."""
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
    (SHAPE_ROOT / f"{name}.dat").write_text(
        "\n".join(f"{x:.7f} {y:.7f}" for x, y in upper + lower) + "\n"
    )


def row(
    thickness,
    camber,
    re,
    mach,
    alpha,
    *,
    run_id="run",
    verdict="ok",
    thickness_x=0.30,
    CL=None,
    CD=None,
    CM=None,
):
    name = f"af{thickness:.3f}_{camber:.3f}".replace(".", "p")
    _write_contour(name, thickness, camber)
    return {
        "run_id": run_id,
        "airfoil": name,
        "verdict": verdict,
        "thickness_ratio": thickness,
        "thickness_x": thickness_x,
        "max_camber": camber,
        "re": re,
        "mach": mach,
        "alpha": alpha,
        "CL": alpha * 0.1 if CL is None else CL,
        "CD": 0.01 if CD is None else CD,
        "CM": -0.02 if CM is None else CM,
    }


CONDITIONS = (
    # thickness, thickness_x, camber, re, mach
    (0.09, 0.26, 0.00, 2.0e5, 0.05),
    (0.10, 0.27, 0.01, 5.0e5, 0.10),
    (0.11, 0.28, 0.02, 1.0e6, 0.15),
    (0.12, 0.29, 0.03, 2.0e6, 0.20),
    (0.13, 0.30, 0.04, 3.5e5, 0.25),
    (0.14, 0.31, 0.05, 8.0e5, 0.30),
    (0.15, 0.32, 0.06, 1.5e6, 0.35),
    (0.16, 0.33, 0.07, 3.0e6, 0.40),
)
"""Every conditioning feature takes eight distinct values, so a fit is
not refused and a held-out split can be formed along each of them."""


def sufficient_rows():
    """Rows with enough variation in every conditioning feature."""
    rows = []
    for index, condition in enumerate(CONDITIONS):
        thickness, thickness_x, camber, re, mach = condition
        for alpha in (-2.0, 0.0, 2.0, 4.0):
            rows.append(
                row(
                    thickness,
                    camber,
                    re,
                    mach,
                    alpha,
                    run_id=f"run_{index:02d}",
                    thickness_x=thickness_x,
                )
            )
    return rows


def one_condition_rows():
    """Rows that vary only in angle of attack."""
    return [
        row(0.12, 0.02, 367500.0, 0.0588, alpha, run_id="run_00")
        for alpha in (-4.0, -2.0, 0.0, 2.0, 4.0, 6.0)
    ]


class LazyImportTest(unittest.TestCase):
    def test_importing_the_module_does_not_load_sklearn(self):
        """Checked in a fresh interpreter.

        This interpreter loads sklearn in other tests, so the
        question can only be answered in a clean process.
        """
        script = (
            "import sys; "
            "import xfoil_mac.analysis.surrogate; "
            "print('sklearn' in sys.modules, 'numpy' in sys.modules)"
        )
        outcome = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        self.assertEqual(outcome.returncode, 0, outcome.stderr)
        self.assertEqual(outcome.stdout.strip(), "False False")

    def test_version_helper_is_optional(self):
        version = surrogate.sklearn_version()
        self.assertTrue(version is None or isinstance(version, str))


class SufficiencyTest(unittest.TestCase):
    def test_single_condition_is_refused_with_named_gaps(self):
        report = surrogate.sufficiency(one_condition_rows())
        self.assertFalse(report["fit_allowed"])
        for name in ("re", "mach"):
            self.assertIn(name, report["insufficient"])

    def test_refusal_says_what_to_add(self):
        reason = surrogate.fit(one_condition_rows())["reason"]
        self.assertIn("not attempted", reason)
        self.assertIn("distinct value", reason)

    def test_varied_conditions_can_be_learned(self):
        report = surrogate.sufficiency(sufficient_rows())
        self.assertTrue(report["fit_allowed"], report)
        self.assertEqual(report["insufficient"], {})

    def test_unchecked_rows_are_not_learned_from(self):
        rows = sufficient_rows()
        before = surrogate.sufficiency(rows)["usable_rows"]
        rows += [dict(item, verdict="unchecked") for item in sufficient_rows()]
        after = surrogate.sufficiency(rows)["usable_rows"]
        self.assertEqual(before, after)

    def test_rows_missing_a_target_are_dropped(self):
        rows = sufficient_rows() + [
            dict(item, CL=None) for item in sufficient_rows()[:5]
        ]
        usable = surrogate.rows_with_targets(rows)
        self.assertTrue(all(item["CL"] is not None for item in usable))

    def test_too_few_rows_is_refused(self):
        rows = sufficient_rows()[:5]
        self.assertFalse(surrogate.sufficiency(rows)["fit_allowed"])

    def test_one_run_cannot_support_a_held_out_split(self):
        rows = [dict(item, run_id="only") for item in sufficient_rows()]
        report = surrogate.sufficiency(rows)
        self.assertFalse(report["fit_allowed"])
        self.assertEqual(report["runs"], 1)


class FitTest(unittest.TestCase):
    def test_fit_refuses_rather_than_reporting_a_score(self):
        outcome = surrogate.fit(one_condition_rows(), shape_root=SHAPE_ROOT)
        self.assertFalse(outcome["fitted"])
        self.assertNotIn("scores", outcome)

    def test_fit_succeeds_when_conditions_vary(self):
        outcome = surrogate.fit(sufficient_rows(), shape_root=SHAPE_ROOT)
        self.assertTrue(outcome["fitted"], outcome.get("reason"))
        self.assertEqual(set(outcome["targets"]), set(surrogate.TARGETS))
        self.assertEqual(
            outcome["sklearn_version"], surrogate.sklearn_version()
        )

    def test_ridge_fit_succeeds_when_conditions_vary(self):
        outcome = surrogate.fit(
            sufficient_rows(), kind="ridge", shape_root=SHAPE_ROOT
        )
        self.assertTrue(outcome["fitted"], outcome.get("reason"))

    def test_saving_a_refused_fit_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                surrogate.save(
                    surrogate.fit(one_condition_rows(), shape_root=SHAPE_ROOT),
                    Path(folder) / "model.joblib",
                )

    def test_saved_model_records_its_library_version(self):
        outcome = surrogate.fit(sufficient_rows(), shape_root=SHAPE_ROOT)
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "model.joblib"
            written = surrogate.save(outcome, destination)
            metadata = json.loads(Path(written["metadata"]).read_text())
            self.assertEqual(
                metadata["sklearn_version"],
                surrogate.sklearn_version(),
            )
            loaded = surrogate.load(destination)
            self.assertTrue(loaded["version_matches"])
            self.assertEqual(set(loaded["models"]), set(surrogate.TARGETS))


class EvaluationTest(unittest.TestCase):
    def test_evaluate_refuses_on_insufficient_data(self):
        outcome = surrogate.evaluate(one_condition_rows(), lambda: None)
        self.assertFalse(outcome["fitted"])
        self.assertIn("reason", outcome)

    def test_scores_are_reported_per_tier_never_only_pooled(self):
        outcome = surrogate.evaluate(
            sufficient_rows(),
            surrogate.ridge_factory,
            folds=2,
            shape_root=SHAPE_ROOT,
        )
        self.assertTrue(outcome["fitted"], outcome.get("reason"))
        for target, scores in outcome["scores"].items():
            self.assertIn("pooled", scores)
            self.assertIn("tiers", scores)

    def test_flagged_rows_are_scored_as_their_own_tier(self):
        rows = sufficient_rows()
        for item in rows[:8]:
            item["verdict"] = "suspect"
        outcome = surrogate.evaluate(
            rows, surrogate.ridge_factory, folds=2, shape_root=SHAPE_ROOT
        )
        self.assertTrue(outcome["fitted"], outcome.get("reason"))
        tiers = outcome["scores"]["CL"]["tiers"]
        self.assertIn("trusted", tiers)
        self.assertIn("flagged", tiers)


class SplitTest(unittest.TestCase):
    """The test set must be reserved before anything is selected on it."""

    def test_split_is_grouped_by_run(self):
        part = surrogate.split_runs(sufficient_rows(), seed=1)
        train_runs = {r["run_id"] for r in part["train"]}
        test_runs = {r["run_id"] for r in part["test"]}
        self.assertEqual(train_runs & test_runs, set())

    def test_no_row_appears_in_both_sides(self):
        rows = sufficient_rows()
        part = surrogate.split_runs(rows, seed=1)
        train_ids = {id(r) for r in part["train"]}
        test_ids = {id(r) for r in part["test"]}
        self.assertEqual(train_ids & test_ids, set())
        self.assertEqual(len(part["train"]) + len(part["test"]), len(rows))

    def test_same_seed_gives_the_same_partition(self):
        rows = sufficient_rows()
        first = surrogate.split_runs(rows, seed=7)
        second = surrogate.split_runs(rows, seed=7)
        self.assertEqual(first["test_runs"], second["test_runs"])

    def test_different_seeds_generally_differ(self):
        rows = sufficient_rows()
        first = surrogate.split_runs(rows, seed=1)
        second = surrogate.split_runs(rows, seed=2)
        self.assertNotEqual(first["test_runs"], second["test_runs"])

    def test_too_few_runs_reserves_nothing(self):
        rows = [dict(item, run_id="only") for item in sufficient_rows()]
        part = surrogate.split_runs(rows)
        self.assertEqual(part["test"], [])
        self.assertIn("only 1 runs", part["reason"])

    def test_test_share_is_close_to_the_configured_fraction(self):
        part = surrogate.split_runs(sufficient_rows(), seed=1)
        total = len(part["train_runs"]) + len(part["test_runs"])
        share = len(part["test_runs"]) / total
        self.assertAlmostEqual(share, surrogate.TEST_FRACTION, delta=0.15)


class HoldoutEvaluateTest(unittest.TestCase):
    def test_report_comes_from_the_reserved_runs_only(self):
        outcome = surrogate.holdout_evaluate(
            sufficient_rows(),
            surrogate.ridge_factory,
            folds=2,
            shape_root=SHAPE_ROOT,
        )
        self.assertTrue(outcome["reported"], outcome.get("reason"))
        self.assertEqual(
            set(outcome["partition"]["test_runs"]),
            set(outcome["document"]["test_run_ids"]),
        )

    def test_selection_and_test_scores_are_reported_separately(self):
        outcome = surrogate.holdout_evaluate(
            sufficient_rows(),
            surrogate.ridge_factory,
            folds=2,
            shape_root=SHAPE_ROOT,
        )
        document = outcome["document"]
        self.assertIn("selection_scores", document)
        self.assertIn("test_scores", document)

    def test_evaluation_is_appended_to_an_audit_trail(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "experiments.jsonl"
            surrogate.holdout_evaluate(
                sufficient_rows(),
                surrogate.ridge_factory,
                log_path=log,
                folds=2,
                shape_root=SHAPE_ROOT,
            )
            first = log.read_text().count("\n")
            surrogate.holdout_evaluate(
                sufficient_rows(),
                surrogate.ridge_factory,
                log_path=log,
                folds=2,
                shape_root=SHAPE_ROOT,
            )
            second = log.read_text().count("\n")
            self.assertEqual(first, 1)
            self.assertEqual(second, 2)

    def test_insufficient_data_reports_nothing(self):
        outcome = surrogate.holdout_evaluate(
            one_condition_rows(),
            surrogate.ridge_factory,
            shape_root=SHAPE_ROOT,
        )
        self.assertFalse(outcome["reported"])
        self.assertIn("reason", outcome)


class PredictionTest(unittest.TestCase):
    """A prediction must describe the airfoil it was asked about.

    _matrix looks the shape up by the row's airfoil name. A query without
    that name gets zero-filled shape columns, which describes a flat
    plate and produces a plausible-looking but wrong answer, so this is
    pinned rather than left to inspection.
    """

    def test_query_carries_the_airfoil_name(self):
        name = "af0p120_0p020"
        row = dict(sufficient_rows()[0], airfoil=name)
        scalars = surrogate.shape_scalars(None, name, SHAPE_ROOT)
        query = {
            "airfoil": name,
            "re": 5e5,
            "mach": 0.1,
            "alpha": 0.0,
            **scalars,
        }
        shapes = surrogate.shape.attach([row], shape_root=SHAPE_ROOT)
        matrix = surrogate._matrix([query], surrogate.FEATURES, shapes)
        shape_block = matrix[0][
            len(surrogate.FEATURES) + len(surrogate.DERIVED_FEATURES) :
        ]
        self.assertTrue(
            any(abs(v) > 0 for v in shape_block),
            "shape columns were zero-filled",
        )

    def test_a_query_without_the_name_gets_zero_shape_columns(self):
        """The failure mode itself, so the guard above has a contrast."""
        name = "af0p120_0p020"
        row = dict(sufficient_rows()[0], airfoil=name)
        shapes = surrogate.shape.attach([row], shape_root=SHAPE_ROOT)
        query = {
            "re": 5e5,
            "mach": 0.1,
            "alpha": 0.0,
            "thickness_ratio": 0.12,
            "thickness_x": 0.30,
            "max_camber": 0.02,
        }
        matrix = surrogate._matrix([query], surrogate.FEATURES, shapes)
        shape_block = matrix[0][
            len(surrogate.FEATURES) + len(surrogate.DERIVED_FEATURES) :
        ]
        self.assertEqual(shape_block, [0.0] * len(surrogate.SHAPE_FEATURES))

    def test_missing_scalars_are_refused_not_guessed(self):
        with self.assertRaises(ValueError):
            surrogate.shape_scalars(None, "no-such-airfoil", SHAPE_ROOT)

    def test_predict_refuses_an_unknown_airfoil(self):
        rows = sufficient_rows()
        outcome = surrogate.fit(rows, shape_root=SHAPE_ROOT)
        with self.assertRaises(ValueError):
            surrogate.predict(
                outcome,
                airfoil="no-such-airfoil",
                re=5e5,
                mach=0.1,
                alphas=[0.0],
                shape_root=SHAPE_ROOT,
            )

    def test_predict_marks_values_outside_the_training_range(self):
        rows = sufficient_rows()
        outcome = surrogate.fit(rows, shape_root=SHAPE_ROOT)
        loaded = {
            "models": outcome["models"],
            "metadata": {
                "training_limits": {
                    "alpha": [0.0, 4.0],
                    "re": [5e5, 5e5],
                    "mach": [0.1, 0.1],
                },
            },
        }
        inside = surrogate.predict(
            loaded,
            airfoil=rows[0]["airfoil"],
            re=5e5,
            mach=0.1,
            alphas=[2.0],
            shape_root=SHAPE_ROOT,
        )
        outside = surrogate.predict(
            loaded,
            airfoil=rows[0]["airfoil"],
            re=5e5,
            mach=0.1,
            alphas=[40.0],
            shape_root=SHAPE_ROOT,
        )
        self.assertFalse(inside[0]["extrapolates"])
        self.assertTrue(outside[0]["extrapolates"])


class DerivedFeatureTest(unittest.TestCase):
    """Shape combinations the raw columns cannot express.

    A tree splits on the columns it is given, so camber alone does not
    tell it about camber relative to thickness. These were measured to
    cut error on every target, most on CM.
    """

    def test_model_consumes_more_columns_than_the_raw_features(self):
        names = surrogate.feature_names()
        self.assertEqual(
            names[: len(surrogate.FEATURES)], list(surrogate.FEATURES)
        )
        self.assertEqual(
            len(names),
            len(surrogate.FEATURES)
            + len(surrogate.DERIVED_FEATURES)
            + len(surrogate.SHAPE_FEATURES),
        )

    def test_shape_columns_can_be_left_out(self):
        raw = surrogate.feature_names(include_shape=False)
        self.assertNotIn(surrogate.SHAPE_FEATURES[0], raw)
        self.assertEqual(
            len(raw),
            len(surrogate.FEATURES) + len(surrogate.DERIVED_FEATURES),
        )

    def test_a_row_without_coordinates_is_refused_not_zero_filled(self):
        rows = sufficient_rows()
        shapes = surrogate.shape.attach(rows, shape_root=SHAPE_ROOT)
        kept, dropped = surrogate.require_shapes(
            [dict(rows[0], airfoil="not-a-real-airfoil")], shapes
        )
        self.assertEqual(kept, [])
        self.assertEqual(len(dropped), 1)

    def test_every_fixture_row_gets_a_shape(self):
        rows = sufficient_rows()
        shapes = surrogate.shape.attach(rows, shape_root=SHAPE_ROOT)
        kept, dropped = surrogate.require_shapes(rows, shapes)
        self.assertEqual(dropped, [])
        self.assertEqual(len(kept), len(rows))

    def test_the_matrix_has_one_column_per_name(self):
        rows = sufficient_rows()
        matrix = surrogate._matrix(rows, surrogate.FEATURES)
        self.assertEqual(len(matrix[0]), len(surrogate.feature_names()))

    def test_the_matrix_keeps_raw_values_first(self):
        rows = sufficient_rows()
        matrix = surrogate._matrix(rows, surrogate.FEATURES)
        first = rows[0]
        self.assertEqual(
            matrix[0][: len(surrogate.FEATURES)],
            [float(first[name]) for name in surrogate.FEATURES],
        )

    def test_an_explicit_name_list_bypasses_the_derived_columns(self):
        rows = sufficient_rows()
        matrix = surrogate._matrix(rows, ["alpha"])
        self.assertEqual(matrix[0], [float(rows[0]["alpha"])])

    def test_camber_ratio_is_guarded_against_zero_thickness(self):
        row = dict(sufficient_rows()[0], thickness_ratio=0.0)
        derived = surrogate._derived(row)
        self.assertEqual(derived[0], 0.0)

    def test_derived_values_follow_the_definition(self):
        row = dict(
            sufficient_rows()[0],
            thickness_ratio=0.10,
            max_camber=0.02,
            alpha=4.0,
        )
        derived = surrogate._derived(row)
        self.assertAlmostEqual(derived[0], 0.2)
        self.assertAlmostEqual(derived[1], 0.4)
        self.assertAlmostEqual(derived[2], 0.002)

    def test_fit_and_evaluate_agree_on_the_columns(self):
        rows = sufficient_rows()
        outcome = surrogate.evaluate(
            rows, surrogate.ridge_factory, folds=2, shape_root=SHAPE_ROOT
        )
        self.assertTrue(outcome["fitted"], outcome.get("reason"))
        self.assertEqual(outcome["features"], list(surrogate.FEATURES))


class IntervalTest(unittest.TestCase):
    """Intervals keep error estimates from implying excess precision."""

    def test_interval_brackets_the_point_estimate(self):
        result = surrogate.interval_for(
            [0.1, 0.2, 0.15, 0.3, 0.25, 0.4],
            ["a", "a", "b", "b", "c", "c"],
        )
        self.assertLess(result["low"], result["mae"])
        self.assertGreater(result["high"], result["mae"])

    def test_groups_are_counted_not_rows(self):
        result = surrogate.interval_for([0.1, 0.2, 0.3], ["a", "a", "a"])
        self.assertEqual(result["groups"], 1)
        self.assertEqual(result["n"], 3)

    def test_a_single_group_cannot_produce_an_interval(self):
        result = surrogate.interval_for([0.1, 0.2], ["only", "only"])
        self.assertIsNone(result["low"])
        self.assertIsNone(result["high"])

    def test_more_groups_narrow_the_interval(self):
        few = surrogate.interval_for(
            [0.1, 0.2, 0.3, 0.4], ["a", "a", "b", "b"]
        )
        many = surrogate.interval_for(
            [0.1, 0.2, 0.3, 0.4] * 5,
            [f"g{i}" for i in range(20)],
        )
        self.assertLess(many["high"] - many["low"], few["high"] - few["low"])

    def test_empty_input_reports_nothing(self):
        result = surrogate.interval_for([], [])
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["mae"])


class EvaluationReportsIntervalsTest(unittest.TestCase):
    def test_scores_carry_an_interval(self):
        outcome = surrogate.evaluate(
            sufficient_rows(),
            surrogate.ridge_factory,
            folds=2,
            shape_root=SHAPE_ROOT,
        )
        self.assertTrue(outcome["fitted"], outcome.get("reason"))
        for target, scores in outcome["scores"].items():
            self.assertIn("interval", scores)
            self.assertIn("mae", scores["interval"])


class StratifiedScoresTest(unittest.TestCase):
    def test_tiers_do_not_merge(self):
        scores = surrogate.stratified_scores(
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.5, 4.5],
            ["ok", "ok", "suspect", "suspect"],
        )
        self.assertEqual(scores["tiers"]["trusted"]["n"], 2)
        self.assertEqual(scores["tiers"]["flagged"]["n"], 2)
        self.assertAlmostEqual(scores["tiers"]["trusted"]["mae"], 0.0)
        self.assertAlmostEqual(scores["tiers"]["flagged"]["mae"], 0.5)

    def test_empty_input_reports_nothing_rather_than_zero(self):
        scores = surrogate.stratified_scores([], [], [])
        self.assertEqual(scores["pooled"]["n"], 0)
        self.assertIsNone(scores["pooled"]["mae"])


if __name__ == "__main__":
    unittest.main()
