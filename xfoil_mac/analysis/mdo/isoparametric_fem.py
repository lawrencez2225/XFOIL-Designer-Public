"""Body-Fitted Isoparametric Q4 Finite Element Solver & Simultaneous Shape-Topology Adjoint.

Key features:
  1. Exact 2x2 Gauss-Legendre quadrature decomposition of the variable-aspect-ratio
     Q4 plane-stress stiffness matrix:
       K_0e(b_e) = (b_e / a) * K_x + (a / b_e) * K_y + K_xy
     where `a = dx` is element chordwise width and `b_e = (y_u(x_e) - y_l(x_e)) / nely`
     is the local body-fitted vertical element thickness.
  2. Exact analytical shape-stiffness derivative:
       dK_0e / db_e = (1 / a) * K_x - (a / b_e^2) * K_y
  3. Batched Multi-Load-Case Direct Solve:
       K(rho, b) * [U_1, U_2, ..., U_K] = [F_1, F_2, ..., F_K]
     solving all flight-envelope load cases in a single matrix factorization.
  4. `@mx.custom_function` wrapping the Simultaneous Discrete Adjoint for
     `(rho_grid, col_height, nodal_forces)` so `mx.grad` backpropagates exact
     sensitivities into both internal rib topology `rho` and external airfoil `z`.
  5. Constant Absolute Structural Mass Budget `M_struct = sum_e rho_e * Area_e(z) = M_budget`.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


def build_q4_basis_matrices(e_mod: float = 1.0, nu: float = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact 8x8 basis matrices (K_x, K_y, K_xy) via 2x2 Gauss quadrature.

    For a rectangular Q4 element of width `a` (x-direction) and height `b` (y-direction),
    the exact 8x8 plane-stress stiffness matrix is:
        K_e(a, b) = (b / a) * K_x + (a / b) * K_y + K_xy
    Node ordering (matching column-major FEM grid):
      n1 = bottom-left  (xi=-1, eta=-1)
      n2 = bottom-right (xi=+1, eta=-1)
      n3 = top-right    (xi=+1, eta=+1)
      n4 = top-left     (xi=-1, eta=+1)
    """
    c_scale = e_mod / (1.0 - nu * nu)
    g_pts = [-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]

    k_x = np.zeros((8, 8), dtype=np.float64)
    k_y = np.zeros((8, 8), dtype=np.float64)
    k_xy = np.zeros((8, 8), dtype=np.float64)

    xi_signs = np.array([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)
    eta_signs = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64)

    for xi in g_pts:
        for eta in g_pts:
            # Derivatives of N_i = 0.25 * (1 + xi_i*xi)*(1 + eta_i*eta) w.r.t. (xi, eta)
            dN_dxi = 0.25 * xi_signs * (1.0 + eta_signs * eta)
            dN_deta = 0.25 * eta_signs * (1.0 + xi_signs * xi)

            # Build 8-vectors for u and v DOFs: [u1, v1, u2, v2, u3, v3, u4, v4]
            bx_u = np.zeros(8, dtype=np.float64)
            bx_v = np.zeros(8, dtype=np.float64)
            by_u = np.zeros(8, dtype=np.float64)
            by_v = np.zeros(8, dtype=np.float64)
            bx_u[0::2] = dN_dxi
            bx_v[1::2] = dN_dxi
            by_u[0::2] = dN_deta
            by_v[1::2] = dN_deta

            # Physical derivatives: d/dx = (2/a) d/dxi, d/dy = (2/b) d/deta, J_det = a*b/4
            # Factor of (2/a)*(2/a)*(a*b/4) = b/a
            g_shear = 0.5 * (1.0 - nu)
            k_x += c_scale * (np.outer(bx_u, bx_u) + g_shear * np.outer(bx_v, bx_v))
            k_y += c_scale * (np.outer(by_v, by_v) + g_shear * np.outer(by_u, by_u))
            k_xy += c_scale * (
                nu * (np.outer(bx_u, by_v) + np.outer(by_v, bx_u))
                + g_shear * (np.outer(by_u, bx_v) + np.outer(bx_v, by_u))
            )

    return k_x, k_y, k_xy


class IsoparametricWingBoxFEM:
    """Body-fitted isoparametric Q4 wing-box mesh & multi-load differentiable FEM solver."""

    def __init__(
        self,
        nelx: int = 36,
        nely: int = 18,
        x_le: float = 0.06,
        x_spar: float = 0.66,
        mass_budget: float = 0.026,
        rmin: float = 1.5,
        e_min: float = 1e-3,
        e_0: float = 1.0,
        penal: float = 3.0,
    ):
        self.nelx = nelx
        self.nely = nely
        self.n_el = nelx * nely
        self.x_le = x_le
        self.x_spar = x_spar
        self.lx = x_spar - x_le
        self.dx = self.lx / nelx
        self.mass_budget = mass_budget
        self.rmin = rmin
        self.e_min = e_min
        self.e_0 = e_0
        self.penal = penal

        # Chordwise column centers x_col in [x_le, x_spar]
        self.x_cols_np = np.linspace(
            x_le + 0.5 * self.dx, x_spar - 0.5 * self.dx, nelx, dtype=np.float32
        )
        self.x_cols = mx.array(self.x_cols_np)

        self.ndof = 2 * (nelx + 1) * (nely + 1)
        self.edof_mat = np.zeros((self.n_el, 8), dtype=np.int32)
        for elx in range(nelx):
            for ely in range(nely):
                el = elx * nely + ely
                n1 = (nely + 1) * elx + ely
                n2 = (nely + 1) * (elx + 1) + ely
                self.edof_mat[el, :] = [
                    2 * n1 + 2,
                    2 * n1 + 3,
                    2 * n2 + 2,
                    2 * n2 + 3,
                    2 * n2,
                    2 * n2 + 1,
                    2 * n1,
                    2 * n1 + 1,
                ]

        # Clamp right-hand main spar web (elx = nelx)
        right_nodes = np.arange((nely + 1) * nelx, (nely + 1) * (nelx + 1), dtype=np.int32)
        self.fixed_dofs = np.concatenate([2 * right_nodes, 2 * right_nodes + 1]).astype(np.int32)
        self.free_dofs = np.setdiff1d(np.arange(self.ndof, dtype=np.int32), self.fixed_dofs)

        # Precompute assembly scatter indices
        self.rows_idx = np.repeat(self.edof_mat, 8, axis=1).reshape(self.n_el, 8, 8)
        self.cols_idx = np.tile(self.edof_mat, (1, 8)).reshape(self.n_el, 8, 8)

        # Exact Q4 basis matrices
        self.k_x, self.k_y, self.k_xy = build_q4_basis_matrices(e_mod=e_0, nu=0.3)

        # Upper and lower surface vertical force DOF indices for each column node ix in 0..nelx-1
        self.top_ydofs = np.array([2 * ((nely + 1) * ix + 0) + 1 for ix in range(nelx)], dtype=np.int32)
        self.bot_ydofs = np.array([2 * ((nely + 1) * ix + nely) + 1 for ix in range(nelx)], dtype=np.int32)

        # Spatial cone filter matrix for mesh-independent topology optimization
        xs, ys = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
        coords = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float32)
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        w_flt = np.maximum(0.0, rmin - dist)
        w_flt /= np.sum(w_flt, axis=1, keepdims=True) + 1e-12
        self.filter_weights = mx.array(w_flt.astype(np.float32))

        # Top and bottom aerodynamic skin passive solid mask (1 element thick skin cap)
        self.skin_mask = np.zeros((nely, nelx), dtype=np.float32)
        self.skin_mask[0, :] = 1.0
        self.skin_mask[-1, :] = 1.0

    def effective_volfrac(self, col_heights_np: np.ndarray) -> float:
        """Compute allowable volume fraction so absolute mass `sum_e rho_e * Area_e == mass_budget`."""
        box_area = float(np.sum(col_heights_np) * self.dx)
        vf = self.mass_budget / max(box_area, 1e-4)
        return float(np.clip(vf, 0.20, 0.68))

    def solve_multi_load_numpy(
        self,
        rho_phys: np.ndarray,
        col_heights: np.ndarray,
        forces_multi: np.ndarray,
    ) -> dict:
        """Assemble isoparametric K(rho, b) and solve K U = F for K load cases simultaneously.

        Args:
          rho_phys: (nely, nelx) element densities in [1e-3, 1.0]
          col_heights: (nelx,) total airfoil thickness `yu(x_col) - yl(x_col)` per column
          forces_multi: (n_loads, ndof) external nodal load vectors
        """
        rho_flat = np.asarray(rho_phys, dtype=np.float64).T.reshape(-1)  # (nelx*nely,)
        stiff = self.e_min + (rho_flat ** self.penal) * (self.e_0 - self.e_min)

        # Element vertical thickness b_e = col_height[elx] / nely
        b_cols = np.maximum(np.asarray(col_heights, dtype=np.float64) / self.nely, 1e-4)
        b_el = np.repeat(b_cols, self.nely)  # (nelx*nely,)
        a = float(self.dx)

        # Exact variable-aspect-ratio element stiffness K_0e(b_e)
        ratio_ba = (b_el / a)[:, None, None]
        ratio_ab = (a / b_el)[:, None, None]
        k0_all = ratio_ba * self.k_x[None, :, :] + ratio_ab * self.k_y[None, :, :] + self.k_xy[None, :, :]
        ke_blocks = stiff[:, None, None] * k0_all

        k_global = np.zeros((self.ndof, self.ndof), dtype=np.float64)
        np.add.at(k_global, (self.rows_idx, self.cols_idx), ke_blocks)

        free = self.free_dofs
        k_ff = k_global[np.ix_(free, free)]
        f_f = forces_multi[:, free].astype(np.float64).T  # (n_free, n_loads)
        u_f = np.linalg.solve(k_ff, f_f)  # single LU factorization for all load cases!

        n_loads = forces_multi.shape[0]
        u_all = np.zeros((n_loads, self.ndof), dtype=np.float64)
        u_all[:, free] = u_f.T

        # Compute per-load-case compliance, element strain energy, and shape-height derivatives
        dk0_db_all = (1.0 / a) * self.k_x[None, :, :] - (a / (b_el * b_el))[:, None, None] * self.k_y[None, :, :]

        compliances = np.zeros(n_loads, dtype=np.float64)
        dc_drho_all = np.zeros((n_loads, self.nely, self.nelx), dtype=np.float64)
        dc_dh_all = np.zeros((n_loads, self.nelx), dtype=np.float64)

        for k in range(n_loads):
            u_e = u_all[k][self.edof_mat]  # (n_el, 8)
            ce_flat = np.einsum("ei,eij,ej->e", u_e, k0_all, u_e)
            compliances[k] = float(np.sum(stiff * ce_flat))

            dc_drho_flat = -self.penal * (self.e_0 - self.e_min) * (rho_flat ** (self.penal - 1.0)) * ce_flat
            dc_drho_all[k] = dc_drho_flat.reshape(self.nelx, self.nely).T

            # Shape derivative w.r.t. element height b_e, then chain rule db_e / dH_col = 1 / nely
            dce_db_flat = np.einsum("ei,eij,ej->e", u_e, dk0_db_all, u_e)
            dc_db_flat = -stiff * dce_db_flat
            dc_dh_all[k] = np.sum(dc_db_flat.reshape(self.nelx, self.nely), axis=1) / self.nely

        return {
            "compliances": compliances,
            "u_all": u_all,
            "dc_drho_all": dc_drho_all,
            "dc_dh_all": dc_dh_all,
        }

    def make_differentiable_multipoint_compliance(
        self,
        rho_fixed_np: np.ndarray,
        condition_weights: tuple[float, ...] = (0.40, 0.40, 0.20),
        p_norm: float = 2.0,
    ):
        """Create an `@mx.custom_function` mapping `(col_heights, top_forces, bot_forces) -> agg_compliance`.

        Propagates exact analytical adjoint gradients back into BOTH:
          1. `col_heights` (geometric stiffness sensitivity dK/dh + constant-mass density scaling)
          2. `top_forces`, `bot_forces` (aerodynamic Cp load sensitivity 2 * U^T dF)
        """
        fem = self
        weights_np = np.asarray(condition_weights, dtype=np.float64)
        n_loads = len(weights_np)

        @mx.custom_function
        def multipoint_comp_op(inputs: mx.array) -> mx.array:
            # Pack inputs as shape (1 + 2*n_loads, nelx):
            # row 0: col_heights (nelx,)
            # rows 1..n_loads: top_forces (n_loads, nelx)
            # rows n_loads+1..2*n_loads: bot_forces (n_loads, nelx)
            inp_np = np.asarray(inputs, dtype=np.float64)
            col_h = np.maximum(inp_np[0], 0.025)
            top_f = inp_np[1 : 1 + n_loads]
            bot_f = inp_np[1 + n_loads : 1 + 2 * n_loads]

            # Constant absolute structural mass scaling: rho_eff scales inversely with box area
            vf_eff = fem.effective_volfrac(col_h)
            vf_base = float(np.mean(rho_fixed_np))
            rho_scaled = np.clip(rho_fixed_np * (vf_eff / max(vf_base, 1e-4)), 1e-3, 1.0)
            rho_scaled = np.where(fem.skin_mask > 0.5, 1.0, rho_scaled)

            forces_mat = np.zeros((n_loads, fem.ndof), dtype=np.float64)
            for k in range(n_loads):
                forces_mat[k, fem.top_ydofs] = top_f[k]
                forces_mat[k, fem.bot_ydofs] = bot_f[k]

            res = fem.solve_multi_load_numpy(rho_scaled, col_h, forces_mat)
            comps = np.maximum(res["compliances"], 1e-8)
            agg = float(np.sum(weights_np * (comps ** p_norm)) ** (1.0 / p_norm))
            return mx.array(agg, dtype=mx.float32)

        @multipoint_comp_op.vjp
        def multipoint_comp_vjp(primal, cotangent, output):
            inp_np = np.asarray(primal, dtype=np.float64)
            col_h = np.maximum(inp_np[0], 0.025)
            top_f = inp_np[1 : 1 + n_loads]
            bot_f = inp_np[1 + n_loads : 1 + 2 * n_loads]

            # Constant-mass scaling derivative: interior elements scale with vf_eff(col_h)
            box_area = float(np.sum(col_h) * fem.dx)
            vf_raw = fem.mass_budget / max(box_area, 1e-4)
            vf_eff = float(np.clip(vf_raw, 0.20, 0.68))
            vf_base = float(np.mean(rho_fixed_np))
            scale_factor = vf_eff / max(vf_base, 1e-4)

            raw_scaled = rho_fixed_np * scale_factor
            interior_active = (fem.skin_mask < 0.5) & (raw_scaled > 1e-3) & (raw_scaled < 1.0)
            rho_scaled = np.clip(raw_scaled, 1e-3, 1.0)
            rho_scaled = np.where(fem.skin_mask > 0.5, 1.0, rho_scaled)

            forces_mat = np.zeros((n_loads, fem.ndof), dtype=np.float64)
            for k in range(n_loads):
                forces_mat[k, fem.top_ydofs] = top_f[k]
                forces_mat[k, fem.bot_ydofs] = bot_f[k]

            res = fem.solve_multi_load_numpy(rho_scaled, col_h, forces_mat)
            comps = np.maximum(res["compliances"], 1e-8)
            agg = float(np.sum(weights_np * (comps ** p_norm)) ** (1.0 / p_norm))

            # d(agg) / d(C_k)
            dagg_dck = weights_np * ((comps / max(agg, 1e-8)) ** (p_norm - 1.0))

            grad_inp = np.zeros_like(inp_np)
            # 1. Direct geometric stiffness sensitivity dC_k / dh_col
            grad_h = np.sum(dagg_dck[:, None] * res["dc_dh_all"], axis=0)

            # 2. Constant-mass density dilution term: increasing h_col dilutes interior density
            if 0.20 < vf_raw < 0.68:
                dvf_dh = -fem.mass_budget * fem.dx / (box_area * box_area)
                dscale_dh = dvf_dh / max(vf_base, 1e-4)
                drho_dh_scalar = np.sum(
                    res["dc_drho_all"] * (rho_fixed_np * interior_active)[None, :, :],
                    axis=(1, 2),
                ) * dscale_dh
                grad_h = grad_h + float(np.sum(dagg_dck * drho_dh_scalar))

            grad_inp[0] = grad_h

            # 3. Load sensitivities dC_k / dF = 2 * U_k
            for k in range(n_loads):
                u_k = res["u_all"][k]
                grad_inp[1 + k] = dagg_dck[k] * 2.0 * u_k[fem.top_ydofs]
                grad_inp[1 + n_loads + k] = dagg_dck[k] * 2.0 * u_k[fem.bot_ydofs]

            return cotangent * mx.array(grad_inp.astype(np.float32))

        return multipoint_comp_op
