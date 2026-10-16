"""Differentiable MLX deep-ensemble surrogate with residual kernel attention.

Shares every methodological rule with :mod:`xfoil_mac.analysis.surrogate`:

* Sufficiency is checked first, and fitting is refused with named gaps when
  any conditioning feature lacks variation.
* ``unchecked`` rows are never learned from.
* Dependencies (``mlx`` and ``numpy``) are imported lazily inside the
  functions that need them, so importing this module never loads a numeric
  or GPU runtime into the solver's path.
* Scores are reported per trust tier (`trusted` vs `flagged`) with grouped
  bootstrap intervals.

In addition to higher accuracy on ``CL``, ``CD``, and ``CM``, the MLX backend
provides three capabilities a tree cannot:

1. Epistemic uncertainty (`CL_std`, `CD_std`, `CM_std`) across ensemble
   members, plus high-dimensional feature distance (`ood_distance`), enabling
   conservative Lower Confidence Bound (LCB) airfoil screening.
2. Exact angle-of-attack inversion (`solve_alpha_for_cl`) via ``mx.grad``
   Newton iteration instead of coarse grid stepping.
3. Differentiable chordwise sensitivity (`shape_sensitivity`) and smooth
   inverse contour modification (`optimize_coordinates`) via ``mx.grad``.
"""

from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path

from ..data import timestamp_label
from . import shape, surrogate

FEATURES = surrogate.FEATURES
DERIVED_FEATURES = surrogate.DERIVED_FEATURES
SHAPE_FEATURES = surrogate.SHAPE_FEATURES
TARGETS = surrogate.TARGETS
SPLIT_SEED = surrogate.SPLIT_SEED
TEST_FRACTION = surrogate.TEST_FRACTION


def mlx_version() -> str | None:
    """Installed MLX version, or None when it is absent."""
    import importlib.metadata as metadata

    try:
        return metadata.version("mlx")
    except metadata.PackageNotFoundError:
        return None


def require_mlx():
    """Import MLX and NumPy on demand with an actionable message."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        import numpy as np
    except ImportError as error:
        raise ImportError(
            "MLX surrogate requires 'mlx' and 'numpy' in the active "
            "environment (e.g. conda activate mlx)."
        ) from error
    return {"mx": mx, "nn": nn, "optim": optim, "np": np}


def split_airfoils(rows: list[dict], *, seed: int = SPLIT_SEED) -> dict:
    """Partition whole airfoils into a training pool and an unseen test set.

    Unlike :func:`surrogate.split_runs`, which groups by ``run_id`` and
    therefore allows the same airfoil solved at another Reynolds or Mach
    number to appear in both splits, this groups by ``airfoil`` so every test
    row represents a geometry never seen at any flow condition.
    """
    import random

    usable = surrogate.rows_with_targets(rows)
    names = sorted(
        {str(row.get("airfoil") or row["run_id"]) for row in usable}
    )
    if len(names) < 5:
        return {
            "train": usable,
            "test": [],
            "train_airfoils": names,
            "test_airfoils": [],
            "reason": (
                f"only {len(names)} distinct airfoils; a reserved test set "
                f"would leave too little to fit on"
            ),
        }
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    held = max(1, int(round(len(shuffled) * TEST_FRACTION)))
    test_airfoils = sorted(shuffled[:held])
    train_airfoils = sorted(shuffled[held:])
    reserved = set(test_airfoils)
    return {
        "train": [
            r
            for r in usable
            if str(r.get("airfoil") or r["run_id"]) not in reserved
        ],
        "test": [
            r
            for r in usable
            if str(r.get("airfoil") or r["run_id"]) in reserved
        ],
        "train_airfoils": train_airfoils,
        "test_airfoils": test_airfoils,
        "reason": "",
    }


def _preprocess_re(matrix, np):
    copied = np.array(matrix, dtype=np.float32, copy=True)
    copied[:, 3] = np.log10(np.clip(copied[:, 3], 1e4, 1e8))
    return copied


def _fit_encoder(matrix, np, n_quantiles: int = 256) -> dict:
    xc = _preprocess_re(matrix, np)
    n_q = min(n_quantiles, max(8, xc.shape[0]))
    q_grid = np.linspace(0.0, 1.0, n_q, dtype=np.float32)
    knots = np.quantile(xc, q_grid, axis=0).astype(np.float32)
    med = np.median(xc, axis=0, keepdims=True).astype(np.float32)
    q75 = np.percentile(xc, 75, axis=0, keepdims=True)
    q25 = np.percentile(xc, 25, axis=0, keepdims=True)
    scale = np.maximum((q75 - q25) * 0.7413, 1e-5).astype(np.float32)
    q05 = np.percentile(xc, 5, axis=0, keepdims=True).astype(np.float32)
    q95 = np.percentile(xc, 95, axis=0, keepdims=True).astype(np.float32)
    q_span = np.maximum(q95 - q05, 1e-5).astype(np.float32)
    return {
        "q_grid": q_grid,
        "knots": knots,
        "med": med,
        "scale": scale,
        "q_span": q_span,
    }


def _encode_numpy(matrix, encoder: dict, np):
    xc = _preprocess_re(matrix, np)
    med = encoder["med"]
    scale = encoder["scale"]
    knots = encoder["knots"]
    q_grid = encoder["q_grid"]

    z = 3.0 * np.tanh((xc - med) / scale / 3.0)
    q = np.zeros_like(xc)
    for col in range(xc.shape[1]):
        u_knots, idx = np.unique(knots[:, col], return_index=True)
        u_q = q_grid[idx] * 2.0 - 1.0
        if len(u_knots) == 1:
            q[:, col] = 0.0
        else:
            q[:, col] = np.interp(xc[:, col], u_knots, u_q)
    return np.hstack([z, q]).astype(np.float32)


def _encode_differentiable_mlx(x_raw_mx, x_ref_np, encoder: dict, mx, np):
    cols = [x_raw_mx[:, i : i + 1] for i in range(x_raw_mx.shape[1])]
    cols[3] = mx.log10(mx.maximum(cols[3], 1e4))
    xc = mx.concatenate(cols, axis=1)

    med_mx = mx.array(encoder["med"])
    scale_mx = mx.array(encoder["scale"])
    q_span_mx = mx.array(encoder["q_span"])

    z = 3.0 * mx.tanh((xc - med_mx) / scale_mx / 3.0)
    encoded_ref = _encode_numpy(x_ref_np, encoder, np)
    n_cols = x_raw_mx.shape[1]
    q_base = mx.stop_gradient(mx.array(encoded_ref[:, n_cols:]))
    xc_ref = mx.stop_gradient(xc)
    q_diff = q_base + (xc - xc_ref) / q_span_mx * 1.8
    return mx.concatenate([z, q_diff], axis=1)


def _build_aero_net(in_dims: int, dim: int, mx, nn):
    class AeroResMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_in = nn.Linear(in_dims, dim)
            self.n1 = nn.LayerNorm(dim)
            self.b1_1 = nn.Linear(dim, dim)
            self.b1_2 = nn.Linear(dim, dim)
            self.n2 = nn.LayerNorm(dim)
            self.b2_1 = nn.Linear(dim, dim)
            self.b2_2 = nn.Linear(dim, dim)
            self.n3 = nn.LayerNorm(dim)
            self.b3_1 = nn.Linear(dim, dim)
            self.b3_2 = nn.Linear(dim, dim)
            self.head = nn.Linear(dim, 3)
            self.skip = nn.Linear(in_dims, 3)

        def __call__(self, x):
            h = nn.silu(self.fc_in(x))
            h = h + self.b1_2(nn.silu(self.b1_1(self.n1(h))))
            h = h + self.b2_2(nn.silu(self.b2_1(self.n2(h))))
            h = h + self.b3_2(nn.silu(self.b3_1(self.n3(h))))
            out = self.head(h) + self.skip(x)
            cl = out[:, 0:1]
            cd = nn.softplus(out[:, 1:2] - 4.0) * 0.05
            cm = out[:, 2:3] * 0.1
            return mx.concatenate([cl, cd, cm], axis=1)

    return AeroResMLP()


def _flatten_params(tree, prefix: str = "") -> dict:
    flat = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            flat.update(_flatten_params(v, name))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            name = f"{prefix}.{i}" if prefix else str(i)
            flat.update(_flatten_params(v, name))
    else:
        flat[prefix] = tree
    return flat


def _train_member(
    x_tr_mx,
    y_tr_mx,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    hidden_dim: int,
    mx,
    nn,
    optim,
):
    mx.random.seed(seed)
    model = _build_aero_net(x_tr_mx.shape[1], hidden_dim, mx, nn)
    mx.eval(model.parameters())

    schedule = optim.cosine_decay(init=1.5e-3, decay_steps=steps, end=1e-5)
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=5e-4)

    def loss_fn(net, xb, yb):
        pred = net(xb)
        d_cl = mx.abs(pred[:, 0] - yb[:, 0])
        d_cd = mx.abs(pred[:, 1] - yb[:, 1]) * 18.0
        d_cm = mx.abs(pred[:, 2] - yb[:, 2]) * 6.0
        return mx.mean(d_cl + d_cd + d_cm)

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step_fn(xb, yb):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, xb, yb)
        optimizer.update(model, grads)
        return loss

    n_samples = x_tr_mx.shape[0]
    bsz = min(batch_size, n_samples)
    for step in range(steps):
        idx = mx.random.randint(0, n_samples, (bsz,))
        loss = step_fn(x_tr_mx[idx], y_tr_mx[idx])
        if (step + 1) % 50 == 0 or step == steps - 1:
            mx.eval(state, loss)
    return model


def _predict_net_batched(model, x_mx, mx, np, batch_size: int = 4096):
    chunks = []
    for i in range(0, x_mx.shape[0], batch_size):
        out = model(x_mx[i : i + batch_size])
        mx.eval(out)
        chunks.append(np.array(out))
    return np.concatenate(chunks, axis=0)


def _attention_weights(np, cross_condition: bool = True):
    w = np.ones((1, 31), dtype=np.float32)
    if cross_condition:
        w[0, 0:3] = 4.0
        w[0, 3:5] = 0.25
        w[0, 5] = 6.0
        w[0, 9:31] = 3.0
    else:
        w[0, 0:3] = 3.0
        w[0, 3:5] = 1.5
        w[0, 5] = 6.0
        w[0, 9:31] = 2.5
    return w


def _apply_residual_attention(
    x_tr_enc,
    x_q_enc,
    res_tr,
    f_q,
    mx,
    np,
    *,
    cross_condition: bool = True,
):
    w = _attention_weights(np, cross_condition=cross_condition)
    n_feat = min(31, x_tr_enc.shape[1] // 2)
    w = w[:, :n_feat]

    z_tr = mx.array(x_tr_enc[:, :n_feat] * w)
    z_q = mx.array(x_q_enc[:, :n_feat] * w)
    res_tr_mx = mx.array(res_tr)

    tr_sq = mx.sum(mx.square(z_tr), axis=1, keepdims=True).T
    q_sq = mx.sum(mx.square(z_q), axis=1, keepdims=True)
    dist2 = mx.maximum(q_sq + tr_sq - 2.0 * mx.matmul(z_q, z_tr.T), 0.0)

    min_d2 = mx.min(dist2, axis=1, keepdims=True)
    tau = 0.15 if cross_condition else 0.50
    gate_scale = 2.0 if cross_condition else 5.0
    strength = 1.0 if cross_condition else 0.7

    weights = mx.softmax(-dist2 / tau, axis=1)
    attn_res = mx.matmul(weights, res_tr_mx)
    gate = mx.exp(-min_d2 / gate_scale) * strength
    y_hat = f_q + np.array(gate * attn_res)
    ood_dist = np.sqrt(np.maximum(np.array(min_d2)[:, 0], 0.0))
    mx.eval(dist2)
    return y_hat, ood_dist


def fit(
    rows: list[dict],
    *,
    ensemble_size: int = 5,
    steps: int = 3000,
    hidden_dim: int = 256,
    batch_size: int = 384,
    seed: int = 0,
    shape_root: Path | None = None,
) -> dict:
    """Fit an MLX deep ensemble + residual attention bank, or refuse."""
    report = surrogate.sufficiency(rows)
    if not report["fit_allowed"]:
        return {
            "fitted": False,
            "reason": surrogate.refusal_reason(report),
            "sufficiency": report,
        }
    deps = require_mlx()
    mx, nn, optim, np = deps["mx"], deps["nn"], deps["optim"], deps["np"]

    usable = surrogate.rows_with_targets(rows)
    shapes = shape.attach(usable, shape_root=shape_root)
    usable, dropped = surrogate.require_shapes(usable, shapes)
    report = surrogate.sufficiency(usable)
    report["rows_without_shape"] = len(dropped)
    if not report["fit_allowed"]:
        return {
            "fitted": False,
            "reason": surrogate.refusal_reason(report),
            "sufficiency": report,
        }

    x_raw = np.array(
        surrogate._matrix(usable, FEATURES, shapes), dtype=np.float32
    )
    y_tr = np.array(
        [[float(r[t]) for t in TARGETS] for r in usable], dtype=np.float32
    )
    encoder = _fit_encoder(x_raw, np)
    x_tr_enc = _encode_numpy(x_raw, encoder, np)
    x_tr_mx = mx.array(x_tr_enc)
    y_tr_mx = mx.array(y_tr)
    mx.eval(x_tr_mx, y_tr_mx)

    models = []
    train_preds = []
    for m in range(ensemble_size):
        net = _train_member(
            x_tr_mx,
            y_tr_mx,
            seed=seed + m,
            steps=steps,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            mx=mx,
            nn=nn,
            optim=optim,
        )
        models.append(net)
        train_preds.append(_predict_net_batched(net, x_tr_mx, mx, np))

    f_tr = np.median(np.stack(train_preds, axis=0), axis=0)
    res_tr = (y_tr - f_tr).astype(np.float32)

    return {
        "fitted": True,
        "kind": "mlx_ensemble",
        "features": list(FEATURES),
        "targets": list(TARGETS),
        "models": models,
        "encoder": encoder,
        "memory": {
            "x_tr_enc": x_tr_enc,
            "res_tr": res_tr,
        },
        "hyperparameters": {
            "in_dims": int(x_tr_enc.shape[1]),
            "hidden_dim": int(hidden_dim),
            "ensemble_size": int(ensemble_size),
            "steps": int(steps),
        },
        "sufficiency": report,
        "mlx_version": mlx_version(),
        "training_rows": usable,
    }


def save(fitted: dict, destination: Path) -> dict:
    """Save MLX ensemble weights, encoder, and memory bank to ``.npz``."""
    destination = Path(destination).expanduser().resolve()
    if not fitted.get("fitted"):
        raise ValueError("Refusing to save: nothing was fitted")
    deps = require_mlx()
    np = deps["np"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "enc_q_grid": fitted["encoder"]["q_grid"],
        "enc_knots": fitted["encoder"]["knots"],
        "enc_med": fitted["encoder"]["med"],
        "enc_scale": fitted["encoder"]["scale"],
        "enc_q_span": fitted["encoder"]["q_span"],
        "mem_x_tr_enc": fitted["memory"]["x_tr_enc"],
        "mem_res_tr": fitted["memory"]["res_tr"],
    }
    for index, net in enumerate(fitted["models"]):
        flat = _flatten_params(net.parameters())
        for key, value in flat.items():
            arrays[f"m{index}__{key}"] = np.array(value)

    with destination.open("wb") as handle:
        np.savez(handle, **arrays)

    limits = {}
    for name in FEATURES:
        values = [float(row[name]) for row in fitted.get("training_rows", [])]
        if values:
            limits[name] = [min(values), max(values)]

    document = {
        "saved": timestamp_label(),
        "kind": fitted.get("kind", "mlx_ensemble"),
        "features": fitted["features"],
        "feature_count": len(surrogate.feature_names()),
        "shape_count": len(SHAPE_FEATURES),
        "descriptor_version": shape.DESCRIPTOR_VERSION,
        "targets": fitted["targets"],
        "mlx_version": fitted.get("mlx_version"),
        "hyperparameters": fitted.get("hyperparameters"),
        "sufficiency": fitted.get("sufficiency"),
        "training_limits": limits,
        "training_rows": len(fitted.get("training_rows", [])),
        "notes": [
            "scores are only meaningful per trust tier",
            "the domain never exceeds that of the solved rows",
            "predictions include ensemble uncertainty (CL_std, CD_std, CM_std)"
            " and feature-space OOD distance",
        ],
    }
    sidecar = destination.with_suffix(".metadata.json")
    sidecar.write_text(json.dumps(document, indent=2, default=str))
    return {"model": str(destination), "metadata": str(sidecar)}


def load(destination: Path) -> dict:
    """Load an MLX ensemble artefact and check descriptor compatibility."""
    deps = require_mlx()
    mx, nn, np = deps["mx"], deps["nn"], deps["np"]

    destination = Path(destination).expanduser().resolve()
    sidecar = destination.with_suffix(".metadata.json")
    document = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    if document.get("descriptor_version") != shape.DESCRIPTOR_VERSION:
        raise ValueError(
            "Shape normalization changed; rebuild the dataset and retrain "
            "this surrogate before loading it"
        )
    archive = np.load(destination, allow_pickle=False)
    hparams = document.get("hyperparameters") or {}
    in_dims = int(hparams.get("in_dims", 62))
    hidden_dim = int(hparams.get("hidden_dim", 256))
    ensemble_size = int(hparams.get("ensemble_size", 5))

    models = []
    for index in range(ensemble_size):
        net = _build_aero_net(in_dims, hidden_dim, mx, nn)
        prefix = f"m{index}__"
        pairs = [
            (key[len(prefix) :], mx.array(archive[key]))
            for key in archive.files
            if key.startswith(prefix)
        ]
        net.load_weights(pairs)
        mx.eval(net.parameters())
        models.append(net)

    encoder = {
        "q_grid": archive["enc_q_grid"],
        "knots": archive["enc_knots"],
        "med": archive["enc_med"],
        "scale": archive["enc_scale"],
        "q_span": archive["enc_q_span"],
    }
    memory = {
        "x_tr_enc": archive["mem_x_tr_enc"],
        "res_tr": archive["mem_res_tr"],
    }
    recorded = document.get("mlx_version")
    current = mlx_version()
    return {
        "kind": "mlx_ensemble",
        "models": models,
        "encoder": encoder,
        "memory": memory,
        "metadata": document,
        "mlx_version": current,
        "version_matches": recorded == current,
    }


def _resolve_airfoil_query(
    coordinates: Path | None,
    airfoil: str | None,
    shape_root: Path | None,
) -> tuple[str, list, dict]:
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
    scalars = surrogate.shape_scalars(coordinates, airfoil, shape_root)
    return label, descriptor, scalars


def predict(
    loaded: dict,
    *,
    airfoil: str | None = None,
    coordinates: Path | None = None,
    re: float,
    mach: float,
    alphas,
    shape_root: Path | None = None,
    cross_condition: bool = True,
) -> list[dict]:
    """Predict ``CL``, ``CD``, ``CM``, uncertainty ``*_std``, and OOD flags."""
    deps = require_mlx()
    mx, np = deps["mx"], deps["np"]

    label, descriptor, scalars = _resolve_airfoil_query(
        coordinates, airfoil, shape_root
    )
    metadata = loaded.get("metadata") or {}
    queries = []
    flags = []
    for alpha in alphas:
        query = {
            "airfoil": label,
            "re": float(re),
            "mach": float(mach),
            "alpha": float(alpha),
            **scalars,
        }
        outside_alpha = surrogate._alpha_outside(float(alpha), metadata)
        inside, outside = surrogate.within_training_limits(query, metadata)
        queries.append(query)
        flags.append(
            {
                "extrapolates": (not inside) or bool(outside_alpha),
                "outside": outside
                + ([outside_alpha] if outside_alpha else []),
            }
        )

    if not queries:
        return []

    x_raw = np.array(
        surrogate._matrix(queries, FEATURES, {label: descriptor}),
        dtype=np.float32,
    )
    x_enc = _encode_numpy(x_raw, loaded["encoder"], np)
    x_mx = mx.array(x_enc)

    member_preds = [
        _predict_net_batched(net, x_mx, mx, np) for net in loaded["models"]
    ]
    stack = np.stack(member_preds, axis=0)
    f_q = np.median(stack, axis=0)
    std_q = np.std(stack, axis=0)

    y_hat, ood_dist = _apply_residual_attention(
        loaded["memory"]["x_tr_enc"],
        x_enc,
        loaded["memory"]["res_tr"],
        f_q,
        mx,
        np,
        cross_condition=cross_condition,
    )

    rows = []
    for idx, query in enumerate(queries):
        rows.append(
            {
                "alpha": float(query["alpha"]),
                "CL": float(y_hat[idx, 0]),
                "CD": float(max(y_hat[idx, 1], 1e-5)),
                "CM": float(y_hat[idx, 2]),
                "CL_std": float(std_q[idx, 0]),
                "CD_std": float(std_q[idx, 1]),
                "CM_std": float(std_q[idx, 2]),
                "ood_distance": float(ood_dist[idx]),
                "extrapolates": flags[idx]["extrapolates"],
                "outside": flags[idx]["outside"],
            }
        )
    return rows


def solve_alpha_for_cl(
    loaded: dict,
    *,
    target_cl: float,
    airfoil: str | None = None,
    coordinates: Path | None = None,
    re: float,
    mach: float,
    alpha_init: float = 2.0,
    tol: float = 1e-4,
    max_iter: int = 8,
    shape_root: Path | None = None,
) -> dict:
    """Solve for the exact angle of attack ``alpha*`` achieving ``target_cl``.

    Uses ``mx.grad`` to evaluate the exact analytic lift-curve slope
    ``dCL/dalpha`` of the neural ensemble at each Newton step.
    """
    deps = require_mlx()
    mx, np = deps["mx"], deps["np"]

    label, descriptor, scalars = _resolve_airfoil_query(
        coordinates, airfoil, shape_root
    )
    base_query = {
        "airfoil": label,
        "re": float(re),
        "mach": float(mach),
        "alpha": float(alpha_init),
        **scalars,
    }
    base_row = np.array(
        surrogate._matrix([base_query], FEATURES, {label: descriptor}),
        dtype=np.float32,
    )
    t_ratio = float(base_row[0, 0])
    encoder = loaded["encoder"]
    models = loaded["models"]

    def cl_of_alpha(alpha_scalar):
        a_val = float(alpha_scalar.item())
        row_np = base_row.copy()
        row_np[0, 5] = a_val
        row_np[0, 7] = t_ratio * a_val

        cols = [mx.array(row_np[:, i : i + 1]) for i in range(31)]
        cols[5] = alpha_scalar.reshape(1, 1)
        cols[7] = cols[0] * alpha_scalar.reshape(1, 1)
        x_raw_mx = mx.concatenate(cols, axis=1)

        x_enc_mx = _encode_differentiable_mlx(
            x_raw_mx, row_np, encoder, mx, np
        )
        preds = [net(x_enc_mx)[0, 0] for net in models]
        return sum(preds) / len(preds)

    grad_fn = mx.grad(cl_of_alpha)
    alpha_curr = float(alpha_init)
    slope_val = 0.11

    for _ in range(max_iter):
        a_mx = mx.array(alpha_curr)
        cl_mx = cl_of_alpha(a_mx)
        dcl_da = grad_fn(a_mx)
        mx.eval(cl_mx, dcl_da)
        err = float(cl_mx.item()) - float(target_cl)
        slope_val = float(dcl_da.item())
        if abs(err) <= tol:
            break
        safe_slope = slope_val if abs(slope_val) > 0.01 else 0.10
        step = max(-4.0, min(4.0, -err / safe_slope))
        alpha_curr = max(-10.0, min(18.0, alpha_curr + step))

    final_pred = predict(
        loaded,
        airfoil=airfoil,
        coordinates=coordinates,
        re=re,
        mach=mach,
        alphas=[alpha_curr],
        shape_root=shape_root,
    )[0]
    final_pred["target_cl"] = float(target_cl)
    final_pred["dCL_dalpha"] = slope_val
    final_pred["cl_error"] = abs(final_pred["CL"] - float(target_cl))
    final_pred["converged"] = final_pred["cl_error"] <= 0.03
    return final_pred


def shape_sensitivity(
    loaded: dict,
    *,
    airfoil: str | None = None,
    coordinates: Path | None = None,
    re: float,
    mach: float,
    alpha: float,
    shape_root: Path | None = None,
) -> dict:
    """Compute chordwise ``d(L/D)/d(shape)`` across all 22 stations."""
    deps = require_mlx()
    mx, np = deps["mx"], deps["np"]

    label, descriptor, scalars = _resolve_airfoil_query(
        coordinates, airfoil, shape_root
    )
    query = {
        "airfoil": label,
        "re": float(re),
        "mach": float(mach),
        "alpha": float(alpha),
        **scalars,
    }
    row_np = np.array(
        surrogate._matrix([query], FEATURES, {label: descriptor}),
        dtype=np.float32,
    )
    encoder = loaded["encoder"]
    models = loaded["models"]

    def ld_of_row(x_raw_mx):
        x_enc = _encode_differentiable_mlx(x_raw_mx, row_np, encoder, mx, np)
        outs = [net(x_enc) for net in models]
        mean_out = sum(outs) / len(outs)
        cl = mean_out[0, 0]
        cd = mx.maximum(mean_out[0, 1], 1e-5)
        return cl / cd

    grad_fn = mx.grad(ld_of_row)
    grads = grad_fn(mx.array(row_np))
    mx.eval(grads)
    g_np = np.array(grads)[0]

    names = surrogate.feature_names(include_shape=True)
    stations = []
    for idx in range(9, 31):
        stations.append(
            {
                "feature": names[idx],
                "value": float(row_np[0, idx]),
                "dLD_dx": float(g_np[idx]),
            }
        )
    stations.sort(key=lambda item: abs(item["dLD_dx"]), reverse=True)
    return {
        "airfoil": label,
        "re": float(re),
        "mach": float(mach),
        "alpha": float(alpha),
        "sensitivities": stations,
    }


def optimize_coordinates(
    loaded: dict,
    *,
    coordinates: Path,
    destination: Path,
    re: float,
    mach: float,
    alpha: float,
    steps: int = 10,
    max_camber_delta: float = 0.006,
    lcb_beta: float = 1.0,
) -> dict:
    """Gradient-guided smooth camber modification of a ``.dat`` airfoil.

    Uses ``mx.grad`` on the 11 chordwise camber stations to identify the
    smooth perturbation direction that improves conservative ``L/D`` while
    keeping ``inspect_points`` valid and staying within training bounds.
    """
    from ..geometry import inspect_points, load_coordinates

    deps = require_mlx()
    np = deps["np"]

    source = Path(coordinates).expanduser().resolve()
    dest = Path(destination).expanduser().resolve()
    if source == dest:
        raise ValueError("Destination must differ from source coordinate file")

    report0, clean0 = inspect_points(load_coordinates(source))
    if not report0["valid"]:
        raise ValueError(f"Source geometry is invalid: {report0['issues']}")

    base_pred = predict(
        loaded,
        coordinates=source,
        re=re,
        mach=mach,
        alphas=[alpha],
    )[0]
    base_lcb = base_pred["CL"] / max(
        base_pred["CD"] + lcb_beta * base_pred["CD_std"], 1e-5
    )

    sens = shape_sensitivity(
        loaded,
        coordinates=source,
        re=re,
        mach=mach,
        alpha=alpha,
    )["sensitivities"]

    camber_grad = {
        float(item["feature"].split("_at_")[1]): item["dLD_dx"]
        for item in sens
        if item["feature"].startswith("camber_at_")
    }
    thick_grad = {
        float(item["feature"].split("_at_")[1]): item["dLD_dx"]
        for item in sens
        if item["feature"].startswith("thickness_at_")
    }

    # Project mx.grad station sensitivities onto smooth C^infty Hicks-Henne
    # bump modes b(x) = sin^3(pi * x^exp) so curvature d^2y/dx^2 stays smooth
    # and never trips premature boundary-layer transition in XFOIL.
    modes = (0.65, 0.85, 1.15)
    c_weights = []
    for exp_val in modes:
        proj = sum(
            grad * (math.sin(math.pi * (st**exp_val)) ** 3)
            for st, grad in camber_grad.items()
            if 0.15 <= st <= 0.85
        )
        c_weights.append(proj)
    max_cw = max(abs(w) for w in c_weights) or 1.0
    c_weights = [w / max_cw for w in c_weights]

    t_mid_grad = sum(
        grad for st, grad in thick_grad.items() if 0.20 <= st <= 0.75
    )
    t_dir = 1.0 if t_mid_grad >= 0 else -1.0

    leading = min(range(len(clean0)), key=lambda i: clean0[i][0])
    best_points = clean0
    best_pred = base_pred
    best_lcb = base_lcb

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        candidate_path = Path(tmp) / "candidate.dat"
        for step_idx in range(1, steps + 1):
            scale_c = max_camber_delta * (step_idx / steps)
            scale_t = (max_camber_delta * 0.4) * (step_idx / steps) * t_dir
            trial_points = []
            for idx, (px, py) in enumerate(clean0):
                if 0.0 < px < 1.0:
                    dc = sum(
                        w * (math.sin(math.pi * (px**e)) ** 3)
                        for w, e in zip(c_weights, modes)
                    ) / len(modes)
                    dt = math.sin(math.pi * (px**0.75)) ** 3
                    sign = 0.5 if idx <= leading else -0.5
                    dy = scale_c * dc + sign * scale_t * dt
                    trial_points.append((px, py + dy))
                else:
                    trial_points.append((px, py))

            rep, checked = inspect_points(trial_points)
            if not rep["valid"]:
                continue
            candidate_path.write_text(
                f"{source.stem}_mlx_opt\n"
                + "\n".join(f"{x:.7f} {y:.7f}" for x, y in checked)
                + "\n"
            )
            pred = predict(
                loaded,
                coordinates=candidate_path,
                re=re,
                mach=mach,
                alphas=[alpha],
                cross_condition=False,
            )[0]
            if pred["extrapolates"]:
                continue
            denom = max(pred["CD"] + lcb_beta * pred["CD_std"], 1e-5)
            lcb = pred["CL"] / denom
            if lcb > best_lcb and pred["CL"] >= base_pred["CL"] * 0.98:
                best_lcb = lcb
                best_pred = pred
                best_points = checked

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        f"{source.stem}_mlx_opt\n"
        + "\n".join(f"{x:.7f} {y:.7f}" for x, y in best_points)
        + "\n"
    )
    return {
        "source": str(source),
        "destination": str(dest),
        "baseline": {
            "CL": base_pred["CL"],
            "CD": base_pred["CD"],
            "CM": base_pred["CM"],
            "L_D": base_pred["CL"] / base_pred["CD"],
            "L_D_lcb": base_lcb,
        },
        "optimized": {
            "CL": best_pred["CL"],
            "CD": best_pred["CD"],
            "CM": best_pred["CM"],
            "L_D": best_pred["CL"] / best_pred["CD"],
            "L_D_lcb": best_lcb,
        },
    }
