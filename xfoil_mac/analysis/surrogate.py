"""Learned approximation of solved results, with its own limits stated.

A surrogate is only worth having if it generalizes to a configuration
that was not solved. That cannot be checked from rows that vary only in
angle of attack: a model fitted across one airfoil at one Reynolds and
one Mach number will report a flattering score while having learned
nothing transferable, and the score will look exactly like success.

So fitting is refused, by name, whenever a conditioning feature shows too
little variation to learn from. The refusal lists what is missing, which
is the useful output in that situation.

Three further rules:

* Dependencies are imported inside the functions that need them, so
  importing this module never loads numpy's tree or a compiled backend
  into the solver's path.
* The scikit-learn version is recorded in the artefact, because a model
  pickled by one version is not guaranteed to load in another.
* Scores are reported separately for trusted and flagged rows. A single
  pooled score would be dominated by the trusted rows and would hide
  behaviour in the region that matters most.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..data import timestamp_label
from . import shape

FEATURES = (
    "thickness_ratio",
    "thickness_x",
    "max_camber",
    "re",
    "mach",
    "alpha",
)
"""Inputs. Geometry comes from measured descriptors, so the design space
is self-controlled and smooth without an external airfoil library."""

GEOMETRY_FEATURES = ("thickness_ratio", "thickness_x", "max_camber")
CONDITION_FEATURES = ("re", "mach")
TARGETS = ("CL", "CD", "CM")

TRAINABLE_VERDICTS = ("ok", "suspect")
"""Verdicts whose numbers may be learned from.

``unchecked`` is excluded rather than trusted: it means the run saved no
boundary layer, so nothing establishes that the solver was inside its
domain. Learning from it would treat unverified output as truth, which
is the failure this project exists to avoid. ``suspect`` stays in, and
is scored separately, because dropping it would silently narrow the
range a surrogate is asked about.
"""

CONDITIONING_FEATURES = GEOMETRY_FEATURES + CONDITION_FEATURES

MIN_DISTINCT_CONDITIONS = 4
"""Distinct values a conditioning feature needs before it can be learned.

Two or three values cannot support a held-out split along that feature,
and a model fitted on them would be reporting interpolation.
"""

MIN_ROWS = 20

GROUPS = ("run_id",)
"""Held-out splits follow whole runs, so rows sharing a run never straddle
the split. Splitting rows of one run would let a model memorise a curve
and score well on it."""

SPLIT_SEED = 20260101
"""Fixed seed for the train/test partition.

The partition must be reproducible: a split drawn afresh each run would
let a model be selected against a test set without anyone recording it,
which is indistinguishable from tuning on the test set.
"""

TEST_FRACTION = 0.2
"""Share of whole runs reserved for the final estimate.

Carved out before any fitting. Selecting a model, a feature set, or a
cross-validation scheme against the same rows that produce the reported
error makes that error a training score wearing a test label.
"""

EXPERIMENT_LOG = "experiments.jsonl"
"""Append-only record, one line per reported evaluation.

Appending rather than overwriting makes repeated inspection of the test
set visible. A test set read often enough stops being a test set.
"""


def split_runs(rows: list[dict], *, seed: int = SPLIT_SEED) -> dict:
    """Partition whole runs into a training pool and a reserved test set.

    The unit is the run, not the row: rows inside one run share an
    airfoil and a flow condition, so a row-level split would place near
    duplicates of the test rows in training.
    """
    import random

    usable = rows_with_targets(rows)
    names = sorted({str(row["run_id"]) for row in usable})
    if len(names) < 5:
        return {
            "train": usable,
            "test": [],
            "train_runs": names,
            "test_runs": [],
            "reason": (
                f"only {len(names)} runs; a reserved test set would "
                f"leave too little to fit on"
            ),
        }
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    held = max(1, int(round(len(shuffled) * TEST_FRACTION)))
    test_runs = sorted(shuffled[:held])
    train_runs = sorted(shuffled[held:])
    reserved = set(test_runs)
    return {
        "train": [r for r in usable if str(r["run_id"]) not in reserved],
        "test": [r for r in usable if str(r["run_id"]) in reserved],
        "train_runs": train_runs,
        "test_runs": test_runs,
        "reason": "",
    }


def log_experiment(destination, document: dict) -> str:
    """Append one evaluation to the audit trail."""
    import json
    from pathlib import Path as _Path

    path = _Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(document)
    record["logged"] = timestamp_label()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    return str(path)


RIDGE_ALPHA = 1.0
MAX_BOOST_ROUNDS = 200


def sklearn_version() -> str | None:
    """Installed scikit-learn version, or None when it is absent."""
    import importlib.metadata as metadata

    try:
        return metadata.version("scikit-learn")
    except metadata.PackageNotFoundError:
        return None


def require_sklearn():
    """Import scikit-learn on demand with an actionable message."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold
    except ImportError as error:
        raise ImportError(
            "Fitting needs scikit-learn in the active environment. "
            "Install it with: python -m pip install --no-user "
            "scikit-learn"
        ) from error
    return {
        "boosting": HistGradientBoostingRegressor,
        "ridge": Ridge,
        "group_kfold": GroupKFold,
    }


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def rows_with_targets(rows: list[dict]) -> list[dict]:
    """Rows that carry a value for every feature and target.

    Rows whose verdict is ``unchecked`` are left out. They have numbers,
    but nothing establishes those numbers are inside the solver's domain.
    """
    usable = []
    for row in rows:
        if row.get("verdict") not in TRAINABLE_VERDICTS:
            continue
        values = [_finite(row.get(name)) for name in FEATURES]
        targets = [_finite(row.get(name)) for name in TARGETS]
        if any(value is None for value in values + targets):
            continue
        usable.append(row)
    return usable


def sufficiency(rows: list[dict]) -> dict:
    """Whether the rows can support a fit that means anything."""
    usable = rows_with_targets(rows)
    distinct = {
        name: len({row[name] for row in usable})
        for name in CONDITIONING_FEATURES
    }
    runs = len({row.get("run_id") for row in usable})
    missing = {
        name: count
        for name, count in distinct.items()
        if count < MIN_DISTINCT_CONDITIONS
    }
    return {
        "rows": len(rows),
        "usable_rows": len(usable),
        "runs": runs,
        "distinct_conditions": distinct,
        "insufficient": missing,
        "fit_allowed": (len(usable) >= MIN_ROWS and runs >= 2 and not missing),
    }


def refusal_reason(report: dict) -> str:
    """A refusal phrased as what to add, not as what is wrong."""
    if report["fit_allowed"]:
        return ""
    parts = []
    if report["usable_rows"] < MIN_ROWS:
        parts.append(
            f"only {report['usable_rows']} usable rows, " f"need {MIN_ROWS}"
        )
    if report["runs"] < 2:
        parts.append(
            f"only {report['runs']} run, need at least 2 to hold one out"
        )
    for name, count in sorted(report["insufficient"].items()):
        parts.append(
            f"{name} takes {count} distinct value(s); "
            f"need {MIN_DISTINCT_CONDITIONS} to learn it"
        )
    return (
        "A fit was not attempted. " + "; ".join(parts) + ". "
        "Solve more configurations before expecting transferable "
        "predictions."
    )


DERIVED_FEATURES = (
    "camber_over_thickness",
    "thickness_times_alpha",
    "camber_times_thickness",
)
"""Shape combinations the raw six columns do not express.

Pitching moment depends on the camber distribution, not only on its peak
value, and a tree can only split on the columns it is given. Adding these
cut measured error on every target, and most on CM.
"""


def _derived(row: dict) -> list[float]:
    thickness = float(row["thickness_ratio"])
    camber = float(row["max_camber"])
    alpha = float(row["alpha"])
    ratio = camber / thickness if thickness else 0.0
    return [ratio, thickness * alpha, camber * thickness]


SHAPE_FEATURES = shape.FEATURE_NAMES
"""Thickness and camber sampled along the chord.

Read from coordinates, not from the dataset row, so they are appended
when a matrix is built rather than stored per row. A row whose
coordinates cannot be read contributes no shape columns and is refused
by :func:`require_shapes` unless the caller opts out.
"""


def feature_names(include_shape: bool = True) -> list[str]:
    """Columns the model actually consumes, raw first."""
    names = list(FEATURES) + list(DERIVED_FEATURES)
    return names + list(SHAPE_FEATURES) if include_shape else names


def _matrix(rows: list[dict], names, shapes: dict | None = None):
    """Rows to a numeric matrix, appending shape columns when available.

    ``shapes`` maps airfoil name to descriptor. A row whose airfoil is
    absent keeps its raw and derived columns and is padded with zeros in
    the shape block, so matrix width never varies between fit and
    predict; :func:`require_shapes` is what refuses such a row.
    """
    if tuple(names) != tuple(FEATURES):
        return [[float(row[name]) for name in names] for row in rows]
    width = len(SHAPE_FEATURES)
    matrix = []
    for row in rows:
        values = [float(row[name]) for name in FEATURES] + _derived(row)
        found = (shapes or {}).get(str(row.get("airfoil")))
        values += list(found) if found else [0.0] * width
        matrix.append(values)
    return matrix


def require_shapes(rows: list, shapes: dict) -> tuple:
    """Split rows into those with a readable shape and those without."""
    kept, dropped = [], []
    for row in rows:
        (kept if str(row.get("airfoil")) in shapes else dropped).append(row)
    return kept, dropped


def stratified_scores(
    truth: list[float], predicted: list[float], verdicts: list[str]
) -> dict:
    """Error reported per trust tier, never only pooled.

    A pooled score would be dominated by whichever tier has more rows.
    """
    tiers: dict[str, list[float]] = {}
    for actual, guess, verdict in zip(truth, predicted, verdicts):
        tier = "trusted" if verdict == "ok" else "flagged"
        tiers.setdefault(tier, []).append(abs(actual - guess))
    pooled = [abs(a - p) for a, p in zip(truth, predicted)]
    return {
        "pooled": _errors(pooled),
        "tiers": {name: _errors(v) for name, v in sorted(tiers.items())},
    }


def _errors(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mae": None, "max": None}
    return {
        "n": len(values),
        "mae": sum(values) / len(values),
        "max": max(values),
    }


BOOTSTRAP_SAMPLES = 2000
"""Resamples used to put an interval on an error estimate.

Thirty-odd runs do not pin an error down to five decimals. Without an
interval, a mean absolute error reads as far more precise than the
evidence supports, which is the failure this guards against.
"""


def _bootstrap_interval(per_group: dict, *, samples: int, seed: int) -> tuple:
    """Percentile interval, resampling whole groups rather than rows.

    Rows inside one run share an airfoil and a flow condition and are
    strongly correlated, so resampling rows would understate the spread.
    """
    import random

    keys = sorted(per_group)
    if len(keys) < 2:
        return None, None
    means = [sum(per_group[k]) / len(per_group[k]) for k in keys]
    sizes = [len(per_group[k]) for k in keys]
    generator = random.Random(seed)
    draws = []
    for _ in range(samples):
        picks = [generator.randrange(len(keys)) for _ in keys]
        total = sum(means[i] * sizes[i] for i in picks)
        weight = sum(sizes[i] for i in picks)
        draws.append(total / weight if weight else 0.0)
    draws.sort()
    low = draws[int(0.025 * len(draws))]
    high = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return low, high


def interval_for(error_values: list, groups: list) -> dict:
    """Bootstrap interval for a mean absolute error over grouped rows."""
    if not error_values:
        return {"n": 0, "mae": None, "low": None, "high": None}
    per_group = {}
    for value, group in zip(error_values, groups):
        per_group.setdefault(group, []).append(value)
    means = [sum(v) / len(v) for v in per_group.values()]
    sizes = [len(v) for v in per_group.values()]
    mae = sum(m * s for m, s in zip(means, sizes)) / sum(sizes)
    low, high = _bootstrap_interval(
        per_group, samples=BOOTSTRAP_SAMPLES, seed=0
    )
    return {
        "n": len(error_values),
        "groups": len(sizes),
        "mae": mae,
        "low": low,
        "high": high,
    }


def evaluate(
    rows: list[dict],
    model_factory,
    *,
    folds: int = 3,
    shape_root: Path | None = None,
) -> dict:
    """Grouped cross-validation with per-tier scoring.

    Whole runs are held out together, so a score cannot come from
    memorising the curve it is scored on.
    """
    usable = rows_with_targets(rows)
    report = sufficiency(rows)
    if not report["fit_allowed"]:
        return {
            "fitted": False,
            "reason": refusal_reason(report),
            "sufficiency": report,
        }
    imports = require_sklearn()
    usable, dropped = require_shapes(
        usable, shape.attach(usable, shape_root=shape_root)
    )
    if dropped:
        report = dict(report)
        report["rows_without_shape"] = len(dropped)
    if not usable:
        return {
            "fitted": False,
            "reason": "no row has a readable shape descriptor",
            "sufficiency": report,
        }
    groups = [row["run_id"] for row in usable]
    unique_groups = sorted(set(groups))
    splits = min(folds, len(unique_groups))
    kfold = imports["group_kfold"](n_splits=splits)
    shapes = shape.attach(usable, shape_root=shape_root)
    matrix = _matrix(usable, FEATURES, shapes)
    results = {}
    for target in TARGETS:
        truth: list[float] = []
        predicted: list[float] = []
        verdicts: list[str] = []
        held_out: list[str] = []
        for train_index, test_index in kfold.split(matrix, groups=groups):
            train = [usable[i] for i in train_index]
            test = [usable[i] for i in test_index]
            model = model_factory()
            model.fit(
                _matrix(train, FEATURES, shapes),
                [float(row[target]) for row in train],
            )
            guesses = model.predict(_matrix(test, FEATURES, shapes))
            truth.extend(float(row[target]) for row in test)
            predicted.extend(float(value) for value in guesses)
            verdicts.extend(str(row.get("verdict")) for row in test)
            held_out.extend(str(row.get("run_id")) for row in test)
        scores = stratified_scores(truth, predicted, verdicts)
        scores["interval"] = interval_for(
            [abs(a - b) for a, b in zip(truth, predicted)], held_out
        )
        results[target] = scores
    return {
        "fitted": True,
        "folds": splits,
        "features": list(FEATURES),
        "sufficiency": report,
        "scores": results,
    }


def holdout_evaluate(
    rows: list[dict],
    model_factory,
    *,
    log_path=None,
    folds: int = 4,
    seed: int = SPLIT_SEED,
    shape_root: Path | None = None,
) -> dict:
    """Select on the training pool, then report once on the test set.

    The training pool is scored by grouped cross-validation, which is a
    model-selection number. The reserved runs are touched only at the
    end, and the result is appended to an audit trail so that repeated
    looks at the test set are visible rather than silent.
    """
    partition = split_runs(rows, seed=seed)
    if not partition["test"]:
        return {
            "reported": False,
            "reason": partition["reason"],
            "train_runs": len(partition["train_runs"]),
            "test_runs": 0,
        }
    selection = evaluate(
        partition["train"],
        model_factory,
        folds=folds,
        shape_root=shape_root,
    )
    if not selection["fitted"]:
        return {
            "reported": False,
            "reason": selection["reason"],
            "train_runs": len(partition["train_runs"]),
            "test_runs": len(partition["test_runs"]),
        }
    require_sklearn()
    every = partition["train"] + partition["test"]
    shapes = shape.attach(every, shape_root=shape_root)
    train, _ = require_shapes(partition["train"], shapes)
    test, _ = require_shapes(partition["test"], shapes)
    if not train or not test:
        return {
            "reported": False,
            "reason": "a readable shape descriptor is needed on both sides",
            "train_runs": len(partition["train_runs"]),
            "test_runs": len(partition["test_runs"]),
        }
    matrix = _matrix(train, FEATURES, shapes)
    scores = {}
    for target in TARGETS:
        model = model_factory()
        model.fit(matrix, [float(row[target]) for row in train])
        guesses = model.predict(_matrix(test, FEATURES, shapes))
        truth = [float(row[target]) for row in test]
        verdicts = [str(row.get("verdict")) for row in test]
        per_target = stratified_scores(truth, list(guesses), verdicts)
        per_target["interval"] = interval_for(
            [abs(a - b) for a, b in zip(truth, guesses)],
            [str(row["run_id"]) for row in test],
        )
        scores[target] = per_target
    document = {
        "seed": seed,
        "train_runs": len(partition["train_runs"]),
        "test_runs": len(partition["test_runs"]),
        "test_run_ids": partition["test_runs"],
        "selection_scores": {
            t: selection["scores"][t]["interval"] for t in TARGETS
        },
        "test_scores": {t: scores[t]["interval"] for t in TARGETS},
    }
    if log_path is not None:
        document["log"] = log_experiment(log_path, document)
    return {
        "reported": True,
        "partition": {
            "train_runs": partition["train_runs"],
            "test_runs": partition["test_runs"],
        },
        "selection": selection,
        "test": scores,
        "document": document,
    }


def ridge_factory():
    imports = require_sklearn()
    return imports["ridge"](alpha=RIDGE_ALPHA)


def boosting_factory():
    imports = require_sklearn()
    return imports["boosting"](max_iter=MAX_BOOST_ROUNDS, random_state=0)


def fit(
    rows: list[dict], *, kind: str = "boosting", shape_root: Path | None = None
) -> dict:
    """Fit one model per target, or refuse with a stated reason."""
    report = sufficiency(rows)
    if not report["fit_allowed"]:
        return {
            "fitted": False,
            "reason": refusal_reason(report),
            "sufficiency": report,
        }
    require_sklearn()
    factory = boosting_factory if kind == "boosting" else ridge_factory
    usable = rows_with_targets(rows)
    usable, _ = require_shapes(
        usable, shape.attach(usable, shape_root=shape_root)
    )
    if not usable:
        return {
            "fitted": False,
            "reason": "no row has a readable shape descriptor",
            "sufficiency": report,
        }
    matrix = _matrix(
        usable, FEATURES, shape.attach(usable, shape_root=shape_root)
    )
    models = {}
    for target in TARGETS:
        model = factory()
        model.fit(matrix, [float(row[target]) for row in usable])
        models[target] = model
    return {
        "fitted": True,
        "kind": kind,
        "features": list(FEATURES),
        "targets": list(TARGETS),
        "models": models,
        "sufficiency": report,
        "sklearn_version": sklearn_version(),
        "training_rows": usable,
    }


EXTRAPOLATION_TOLERANCE = 0.15
"""Relative slack allowed beyond a training bound before a query counts
as extrapolating.

Measured: the tree splits only at values it saw, so a query just outside
a bound usually lands in the same leaf as the boundary and returns the
same number. On this data, moving Mach from 0.059 to 0.01 and then to
0.7 left the error unchanged at 0.0376, because Mach was never a useful
split. A strict bound refusal therefore blocks usable queries without
blocking any measured risk. Beyond this fraction the query is no longer
near the data and the flag stays.
"""


def within_training_limits(
    query: dict, metadata: dict, *, tolerance: float | None = None
) -> tuple:
    """Which query features fall meaningfully outside the training values.

    A tree returns a number everywhere; it has no way to decline. This is
    what lets a caller tell a prediction that interpolates from one that
    extrapolates. A small overshoot is tolerated because the model has
    been measured not to depend on it; a large one is reported.
    """
    slack = EXTRAPOLATION_TOLERANCE if tolerance is None else tolerance
    limits = metadata.get("training_limits") or {}
    outside = []
    for name, bounds in limits.items():
        value = _finite(query.get(name))
        if value is None or len(bounds) != 2:
            continue
        low, high = float(bounds[0]), float(bounds[1])
        width = high - low
        if width <= 0:
            margin = abs(high) * slack
        else:
            margin = width * slack
        if value < low - margin or value > high + margin:
            outside.append(f"{name}={value:g} outside [{low:g}, {high:g}]")
    return (not outside), outside


def predict(
    loaded: dict,
    *,
    airfoil: str | None = None,
    coordinates: Path | None = None,
    re: float,
    mach: float,
    alphas,
    shape_root: Path | None = None,
) -> list:
    """Predict the coefficients at each angle, with an extrapolation flag.

    Either a library airfoil name or a coordinate file must be given: the
    model reads the shape from coordinates, so it cannot answer for a
    shape it has never seen the outline of.
    """
    if coordinates is not None:
        descriptor = shape.descriptor(Path(coordinates))
        label = Path(coordinates).stem
    elif airfoil:
        descriptor = shape.descriptor_for(
            airfoil, shape_root or shape.COORDINATE_ROOT
        )
        label = airfoil
    else:
        raise ValueError("Give an airfoil name or a coordinate file")
    if descriptor is None:
        raise ValueError(f"Could not read a shape for {label}")
    models = loaded["models"]
    metadata = loaded.get("metadata") or {}
    scalars = shape_scalars(coordinates, airfoil, shape_root)
    rows = []
    for alpha in alphas:
        query = {
            # The airfoil name is what _matrix looks the shape up by.
            # Without it every shape column is zero-filled and the
            # answer describes an airfoil nobody asked about.
            "airfoil": label,
            "re": float(re),
            "mach": float(mach),
            "alpha": float(alpha),
            **scalars,
        }
        outside_alpha = _alpha_outside(float(alpha), metadata)
        inside, outside = within_training_limits(query, metadata)
        matrix = _matrix([query], FEATURES, {label: descriptor})
        values = {
            target: float(models[target].predict(matrix)[0])
            for target in models
        }
        rows.append(
            {
                "alpha": float(alpha),
                **values,
                "extrapolates": (not inside) or outside_alpha,
                "outside": outside
                + ([outside_alpha] if outside_alpha else []),
            }
        )
    return rows


def shape_scalars(
    coordinates: Path | None,
    airfoil: str | None,
    shape_root: Path | None,
) -> dict:
    """Peak thickness, its location and peak camber, from coordinates.

    These are three of the model's inputs, so they must describe the
    airfoil being asked about rather than being left at a placeholder.
    """
    from ..geometry import inspect_points, load_coordinates

    path = None
    if coordinates is not None:
        path = Path(coordinates)
    elif airfoil:
        root = shape_root or shape.COORDINATE_ROOT
        candidate = Path(root) / f"{airfoil}.dat"
        if candidate.is_file():
            path = candidate
    if path is None or not path.is_file():
        raise ValueError(
            "Cannot read the airfoil, so its thickness and camber are "
            "unknown and the prediction would be meaningless"
        )
    report, _ = inspect_points(load_coordinates(path))
    for name in ("thickness_ratio", "thickness_x", "max_camber"):
        if report.get(name) is None:
            raise ValueError(f"Could not measure {name} from {path}")
    return {
        "thickness_ratio": float(report["thickness_ratio"]),
        "thickness_x": float(report["thickness_x"]),
        "max_camber": float(report["max_camber"]),
    }


def _alpha_outside(
    alpha: float, metadata: dict, tolerance: float | None = None
) -> str:
    slack = EXTRAPOLATION_TOLERANCE if tolerance is None else tolerance
    bounds = (metadata.get("training_limits") or {}).get("alpha")
    if not bounds or len(bounds) != 2:
        return ""
    low, high = float(bounds[0]), float(bounds[1])
    margin = (high - low) * slack
    if alpha < low - margin or alpha > high + margin:
        return f"alpha={alpha:g} outside [{low:g}, {high:g}]"
    return ""


def save(fitted: dict, destination: Path) -> dict:
    """Write the models and a sidecar recording what made them.

    A model pickled by one scikit-learn version is not guaranteed to load
    in another, so the version is stored beside the artefact and checked
    on load.
    """
    destination = Path(destination).expanduser().resolve()
    if not fitted.get("fitted"):
        raise ValueError("Refusing to save: nothing was fitted")
    import joblib

    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted["models"], destination)
    limits = {}
    for name in FEATURES:
        values = [float(row[name]) for row in fitted.get("training_rows", [])]
        if values:
            limits[name] = [min(values), max(values)]
    document = {
        "saved": timestamp_label(),
        "kind": fitted["kind"],
        "features": fitted["features"],
        "feature_count": len(feature_names()),
        "shape_count": len(SHAPE_FEATURES),
        "targets": fitted["targets"],
        "sklearn_version": fitted.get("sklearn_version"),
        "sufficiency": fitted.get("sufficiency"),
        "training_limits": limits,
        "training_rows": len(fitted.get("training_rows", [])),
        "notes": [
            "scores are only meaningful per trust tier",
            "the domain never exceeds that of the solved rows",
            "a query outside training_limits is extrapolation and is "
            "reported as such",
        ],
    }
    sidecar = destination.with_suffix(".metadata.json")
    sidecar.write_text(json.dumps(document, indent=2, default=str))
    return {"model": str(destination), "metadata": str(sidecar)}


def load(destination: Path) -> dict:
    """Load models, reporting a version mismatch instead of failing late."""
    import joblib

    destination = Path(destination).expanduser().resolve()
    sidecar = destination.with_suffix(".metadata.json")
    document = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    recorded = document.get("sklearn_version")
    current = sklearn_version()
    return {
        "models": joblib.load(destination),
        "metadata": document,
        "sklearn_version": current,
        "version_matches": recorded == current,
    }
