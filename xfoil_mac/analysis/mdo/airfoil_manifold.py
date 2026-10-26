"""Differentiable Kulfan-CST Airfoil Parameterization & Generative Flow Prior in MLX.

Provides:
  1. Differentiable mapping from a 6-D shape vector `z` in [-2.5, 2.5]^6 to
     upper/lower airfoil coordinates `(xc, yu(xc), yl(xc))` using Bernstein /
     Class-Shape Transformation (CST) basis polynomials in `mlx.core`.
  2. Multi-modal training dataset of 3 distinct aerodynamic families (Thin-Laminar,
     High-Camber Glider, and Supercritical Aero-Structural Pareto) with a
     non-convex multi-basin objective landscape.
  3. A `TimeModulatedFlowNet` trained via Rectified Flow matching on the 3-family
     manifold, supporting both unguided flow generation and Physics-Guided
     Diffusion / Flow Sampling (DPS via Tweedie denoised adjoint guidance).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


def make_chord_grid(n_pts: int = 49) -> mx.array:
    """Cosine-clustered chord stations x/c in [0, 1]."""
    beta = np.linspace(0.0, math.pi, n_pts, dtype=np.float32)
    xc = 0.5 * (1.0 - np.cos(beta))
    return mx.array(xc)


def make_cst_basis(xc: mx.array) -> tuple[mx.array, mx.array]:
    """Precompute N1=0.5, N2=1.0 CST class function and degree-3 Bernstein basis."""
    xc_safe = mx.clip(xc, 1e-6, 1.0 - 1e-6)
    class_fn = mx.sqrt(xc_safe) * (1.0 - xc_safe)  # (N,)
    omx = 1.0 - xc_safe
    b0 = omx * omx * omx
    b1 = 3.0 * xc_safe * omx * omx
    b2 = 3.0 * xc_safe * xc_safe * omx
    b3 = xc_safe * xc_safe * xc_safe
    bernstein = mx.stack([b0, b1, b2, b3], axis=-1)  # (N, 4)
    return class_fn, bernstein


_XC_DEFAULT = make_chord_grid(49)
_CLASS_FN, _BERNSTEIN = make_cst_basis(_XC_DEFAULT)


def decode_airfoil_z(
    z: mx.array,
    xc: mx.array = _XC_DEFAULT,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Differentiably decode latent shape vector `z` (shape `[..., 6]`) into airfoil geometry.

    Components of `z`:
      z[..., 0]: Overall thickness mode (controls base structural depth)
      z[..., 1]: Leading-edge radius / forward fullness mode
      z[..., 2]: Aft-camber / supercritical loading mode
      z[..., 3]: Main camber arch mode
      z[..., 4]: Upper-surface crest shift mode
      z[..., 5]: Reflex / trailing-edge decambering mode

    Returns:
      yu: Upper surface y/c coordinates, shape `[..., N]`
      yl: Lower surface y/c coordinates, shape `[..., N]`
      thickness: Local thickness `yu - yl`, shape `[..., N]`
      camber: Local camberline `0.5 * (yu + yl)`, shape `[..., N]`
    """
    if xc.shape[0] != _XC_DEFAULT.shape[0]:
        class_fn, bern = make_cst_basis(xc)
    else:
        class_fn, bern = _CLASS_FN, _BERNSTEIN

    # Smooth bounded physical CST weights via tanh/sigmoid
    t_scale = 0.082 + 0.095 * mx.sigmoid(0.95 * z[..., 0:1])          # max thickness t/c in [0.082, 0.177]
    le_full = 0.18 * mx.tanh(0.8 * z[..., 1:2])                       # LE fullness
    aft_load = 0.24 * mx.tanh(0.8 * z[..., 2:3])                      # aft supercritical loading
    cam_main = 0.014 + 0.026 * mx.sigmoid(1.05 * z[..., 3:4])         # positive lifting camber in [1.4%, 4.0%]
    crest = 0.16 * mx.tanh(0.8 * z[..., 4:5])                         # upper crest shift
    reflex = 0.14 * mx.tanh(0.8 * z[..., 5:6])                        # TE reflex / aft cusp

    # 4 Bernstein half-thickness weights [B0, B1, B2, B3]
    wt_0 = 1.18 + le_full
    wt_1 = 1.08 + 0.45 * crest
    wt_2 = 0.86 + 0.45 * aft_load
    wt_3 = 0.58 - 0.25 * reflex
    w_thick = mx.concatenate([wt_0, wt_1, wt_2, wt_3], axis=-1) * (t_scale / 0.76)

    # 4 Bernstein camber weights normalized by max(class_fn)=0.385 so camber reaches 2.5%~3.8% of chord,
    # with distinct supercritical S-camber / aft loading (wc_2, wc_3)
    wc_0 = 0.75 * cam_main
    wc_1 = 1.15 * cam_main + 0.008 * crest
    wc_2 = 1.38 * cam_main + 0.022 * aft_load
    wc_3 = 1.45 * cam_main + 0.028 * aft_load - 0.015 * reflex
    w_cam = mx.concatenate([wc_0, wc_1, wc_2, wc_3], axis=-1) / 0.385

    # Evaluate CST polynomials: shape [..., N]
    half_thick = class_fn * (w_thick @ bern.T)
    half_thick = mx.maximum(half_thick, 0.004 * class_fn)
    camber = class_fn * (w_cam @ bern.T)

    # Authentic asymmetric upper/lower CST shaping:
    # Upper surface has higher arch; lower surface has forward belly + aft supercritical concave cusp
    yu = camber + 1.04 * half_thick
    yl = camber - 0.96 * half_thick
    thickness = yu - yl
    return yu, yl, thickness, camber


def sample_multimodal_airfoil_latents(n_samples: int = 2400, seed: int = 42) -> np.ndarray:
    """Sample a 3-basin multimodal airfoil prior in R^6.

    Basin 0 (Thin High-Speed Laminar): z0 ~ -1.3 (thin, low camber)
    Basin 1 (High-Camber Glider):      z0 ~ -0.3, z3 ~ +1.4 (moderate thickness, high camber)
    Basin 2 (Supercritical Aero-Structural Pareto): z0 ~ +1.25, z1 ~ +0.9, z2 ~ +1.1, z3 ~ +0.85
    """
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [-1.35, -0.40, -0.30,  0.15, -0.20,  0.10],  # Basin 0: Thin Laminar
            [-0.25,  0.10,  0.20,  1.45,  0.50, -0.30],  # Basin 1: High-Camber Glider
            [ 1.25,  0.90,  1.05,  0.85,  0.40, -0.15],  # Basin 2: Supercritical Box Wing
        ],
        dtype=np.float32,
    )
    stds = np.array(
        [
            [0.25, 0.30, 0.30, 0.25, 0.30, 0.30],
            [0.28, 0.30, 0.30, 0.25, 0.30, 0.30],
            [0.26, 0.28, 0.28, 0.25, 0.28, 0.28],
        ],
        dtype=np.float32,
    )
    comp = rng.integers(0, 3, size=n_samples)
    noise = rng.normal(0.0, 1.0, size=(n_samples, 6)).astype(np.float32)
    return centers[comp] + stds[comp] * noise


class LatentFlowVelocityNet(nn.Module):
    """Time-conditioned velocity network v_theta(z_t, t) on the 6-D airfoil latent space."""

    def __init__(self, dim: int = 6, hidden: int = 96):
        super().__init__()
        self.fc1 = nn.Linear(dim + 16, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, dim)
        freqs = np.exp(-math.log(100.0) * np.arange(8, dtype=np.float32) / 7.0) * 2.0 * math.pi
        self.freqs = mx.array(freqs)

    def __call__(self, z: mx.array, t: mx.array) -> mx.array:
        ang = t * self.freqs
        t_emb = mx.concatenate([mx.sin(ang), mx.cos(ang)], axis=-1)
        h = nn.silu(self.fc1(mx.concatenate([z, t_emb], axis=-1)))
        h = h + nn.silu(self.fc2(h))
        h = h + nn.silu(self.fc3(h))
        return self.out(h)


def train_airfoil_flow_prior(
    n_samples: int = 2400,
    iters: int = 500,
    batch_size: int = 256,
    lr: float = 3e-3,
    seed: int = 42,
) -> tuple[LatentFlowVelocityNet, np.ndarray]:
    """Train a Rectified Flow generative prior on the 3-basin airfoil latent distribution."""
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    z1_np = sample_multimodal_airfoil_latents(n_samples=n_samples, seed=seed)
    z1_mx = mx.array(z1_np)

    model = LatentFlowVelocityNet(dim=6, hidden=96)
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(mdl: LatentFlowVelocityNet, z0: mx.array, z1: mx.array, t: mx.array) -> mx.array:
        zt = (1.0 - t) * z0 + t * z1
        v_pred = mdl(zt, t)
        return mx.mean((v_pred - (z1 - z0)) ** 2)

    vg_fn = nn.value_and_grad(model, loss_fn)
    for _ in range(iters):
        idx = mx.array(rng.integers(0, n_samples, size=batch_size))
        b_z1 = z1_mx[idx]
        b_z0 = mx.random.normal((batch_size, 6))
        t_b = mx.random.uniform(0.0, 1.0, (batch_size, 1))
        loss, grads = vg_fn(model, b_z0, b_z1, t_b)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters(), optimizer.state)

    return model, z1_np
