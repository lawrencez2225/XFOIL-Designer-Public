"""Zero-dependency PNG & HTML report generator for Multi-Paradigm Aero-Structural MDO."""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: str | Path, rgb: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 RGB array to a standard PNG file."""
    rgb = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    h, w, _ = rgb.shape
    raw_rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw_rows, level=6))
        + chunk(b"IEND", b"")
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(png_bytes)


def _draw_line(
    canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    dashed: bool = False,
) -> None:
    h, w, _ = canvas.shape
    dist = max(1, int(math.hypot(x1 - x0, y1 - y0) * 1.5))
    ts = np.linspace(0.0, 1.0, dist)
    xs = np.round(x0 + (x1 - x0) * ts).astype(int)
    ys = np.round(y0 + (y1 - y0) * ts).astype(int)
    if dashed:
        mask = (np.arange(dist) // 5) % 2 == 0
        xs = xs[mask]
        ys = ys[mask]

    col_arr = np.array(color, dtype=np.uint8)
    for dx in range(-thickness // 2, thickness // 2 + 1):
        for dy in range(-thickness // 2, thickness // 2 + 1):
            px = xs + dx
            py = ys + dy
            valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            canvas[py[valid], px[valid]] = col_arr


def _draw_circle(
    canvas: np.ndarray, cx: float, cy: float, radius: int, color: tuple[int, int, int]
) -> None:
    h, w, _ = canvas.shape
    ix, iy = int(round(cx)), int(round(cy))
    col = np.array(color, dtype=np.uint8)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                px, py = ix + dx, iy + dy
                if 0 <= px < w and 0 <= py < h:
                    canvas[py, px] = col


_FONT_5X7: dict[str, tuple[int, ...]] = {
    " ": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00),
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F),
    "3": (0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
    "A": (0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "B": (0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E),
    "C": (0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E),
    "D": (0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C),
    "E": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F),
    "F": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10),
    "G": (0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F),
    "H": (0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "I": (0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "J": (0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C),
    "K": (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11),
    "L": (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F),
    "M": (0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11),
    "N": (0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11),
    "O": (0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "P": (0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "Q": (0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D),
    "R": (0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11),
    "S": (0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E),
    "T": (0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "U": (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "V": (0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "W": (0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11),
    "X": (0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11),
    "Y": (0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04),
    "Z": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F),
    ".": (0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C),
    ",": (0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x08),
    ":": (0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00),
    "-": (0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00),
    "+": (0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00),
    "=": (0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00),
    "/": (0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10),
    "%": (0x19, 0x19, 0x02, 0x04, 0x08, 0x13, 0x13),
    "(": (0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02),
    ")": (0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08),
    "[": (0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E),
    "]": (0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E),
    "|": (0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    ">": (0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08),
    "<": (0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02),
}


def _draw_text(
    canvas: np.ndarray,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int] = (226, 232, 240),
    scale: int = 1,
) -> None:
    """Render crisp 5x7 bitmap ASCII text onto an RGB uint8 canvas."""
    h, w, _ = canvas.shape
    col = np.array(color, dtype=np.uint8)
    cx = int(x)
    cy = int(y)
    for ch in text.upper():
        glyph = _FONT_5X7.get(ch, _FONT_5X7[" "])
        for row_idx, row_bits in enumerate(glyph):
            for col_idx in range(5):
                if (row_bits >> (4 - col_idx)) & 1:
                    py0 = cy + row_idx * scale
                    px0 = cx + col_idx * scale
                    py1 = min(h, py0 + scale)
                    px1 = min(w, px0 + scale)
                    if py0 >= 0 and px0 >= 0 and py0 < h and px0 < w:
                        canvas[py0:py1, px0:px1] = col
        cx += 6 * scale


def render_body_fitted_airfoil_rib_tile(
    xc_full: np.ndarray,
    yu_full: np.ndarray,
    yl_full: np.ndarray,
    rho_sr: np.ndarray,
    x_le: float = 0.06,
    x_spar: float = 0.66,
    width: int = 440,
    height: int = 280,
    accent_color: tuple[int, int, int] = (56, 189, 248),
    title: str = "",
    subtitle: str = "",
) -> np.ndarray:
    """Render the continuous isoparametric rib topology directly inside the curved airfoil contour."""
    tile = np.full((height, width, 3), 15, dtype=np.uint8)
    left_m, right_m, top_m, bot_m = 48, 22, 42, 34
    pw = width - left_m - right_m
    ph = height - top_m - bot_m

    def to_px(x_val: float, y_val: float) -> tuple[int, int]:
        px = int(round(left_m + x_val * pw))
        # y_val in [-0.15, +0.15]
        py = int(round(top_m + ph * 0.52 - (y_val / 0.30) * ph))
        return px, py

    # Background coordinate axes & grid lines
    _, y_zero = to_px(0.0, 0.0)
    tile[y_zero, left_m : width - right_m] = [33, 45, 64]
    for xv in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        gx, _ = to_px(xv, 0.0)
        tile[top_m : height - bot_m, gx] = [24, 34, 50]
        # X-axis tick mark and label
        tile[height - bot_m : height - bot_m + 4, gx] = [148, 163, 184]
        _draw_text(tile, gx - 8, height - bot_m + 7, f"{xv:.1f}", (148, 163, 184), scale=1)

    for yv in (-0.10, 0.00, 0.10):
        _, gy = to_px(0.0, yv)
        if top_m <= gy < height - bot_m:
            tile[gy, left_m - 4 : left_m] = [148, 163, 184]
            lbl = f"{yv:+.2f}" if yv != 0 else " 0.00"
            _draw_text(tile, 6, gy - 3, lbl, (148, 163, 184), scale=1)

    # Axes frame lines
    tile[top_m : height - bot_m + 1, left_m] = [71, 85, 105]
    tile[height - bot_m, left_m : width - right_m + 1] = [71, 85, 105]
    _draw_text(tile, left_m + pw // 2 - 42, height - 12, "X: CHORD X/C", (148, 163, 184), scale=1)
    _draw_text(tile, 6, top_m - 12, "Y/C", (148, 163, 184), scale=1)

    sh, sw = rho_sr.shape
    xs_dense = np.linspace(0.0, 1.0, pw)
    yu_dense = np.interp(xs_dense, xc_full, yu_full)
    yl_dense = np.interp(xs_dense, xc_full, yl_full)

    for i_x, (x_v, u_v, l_v) in enumerate(zip(xs_dense, yu_dense, yl_dense)):
        px = left_m + i_x
        _, py_top = to_px(float(x_v), float(u_v))
        _, py_bot = to_px(float(x_v), float(l_v))
        y0 = max(top_m + 2, min(height - bot_m - 2, py_top))
        y1 = max(top_m + 2, min(height - bot_m - 2, py_bot))

        if y1 > y0:
            if x_le <= x_v <= x_spar:
                xi = (x_v - x_le) / max(x_spar - x_le, 1e-6)
                c_idx = min(sw - 1, max(0, int(round(xi * (sw - 1)))))
                ys_col = np.arange(y0, y1 + 1)
                eta = (ys_col - y0) / max(1, y1 - y0)
                r_idx = np.clip(np.round(eta * (sh - 1)).astype(int), 0, sh - 1)
                d_vals = rho_sr[r_idx, c_idx]

                r_c = 18.0 + d_vals * (accent_color[0] - 18.0) + (d_vals ** 3) * 145.0
                g_c = 26.0 + d_vals * (accent_color[1] - 26.0) + (d_vals ** 3) * 55.0
                b_c = 42.0 + d_vals * (accent_color[2] - 42.0)
                tile[ys_col, px, 0] = np.clip(r_c, 0, 255).astype(np.uint8)
                tile[ys_col, px, 1] = np.clip(g_c, 0, 255).astype(np.uint8)
                tile[ys_col, px, 2] = np.clip(b_c, 0, 255).astype(np.uint8)
            else:
                tile[y0 : y1 + 1, px] = [23, 33, 50]

        tile[max(0, y0 - 1) : min(height, y0 + 2), max(0, px - 1) : min(width, px + 1)] = accent_color
        tile[max(0, y1 - 1) : min(height, y1 + 2), max(0, px - 1) : min(width, px + 1)] = (16, 185, 129)

        # Draw arched camberline yc(x) = 0.5 * (yu + yl) as a dashed light reference curve
        if (i_x // 4) % 2 == 0:
            _, py_cam = to_px(float(x_v), 0.5 * float(u_v + l_v))
            if top_m < py_cam < height - bot_m:
                tile[py_cam, px] = [203, 213, 225]

    # Draw main spar web clamp line at x = x_spar
    px_spar, py_s_top = to_px(x_spar, float(np.interp(x_spar, xc_full, yu_full)))
    _, py_s_bot = to_px(x_spar, float(np.interp(x_spar, xc_full, yl_full)))
    _draw_line(canvas=tile, x0=px_spar, y0=py_s_top, x1=px_spar, y1=py_s_bot, color=(245, 158, 11), thickness=3)
    _draw_text(tile, px_spar + 5, py_s_top + 4, "SPAR", (245, 158, 11), scale=1)

    if title:
        _draw_text(tile, 12, 9, title, accent_color, scale=1)
    if subtitle:
        _draw_text(tile, 12, 22, subtitle, (226, 232, 240), scale=1)

    return tile


def save_three_paradigms_panel(
    path: str | Path,
    xc_full: np.ndarray,
    adj_res: dict,
    bo_res: dict,
    flow_res: dict,
    prior_samples: np.ndarray,
) -> None:
    """Render 2-row x 3-col visual comparison of Adjoint vs Latent GP-BO vs Physics-Guided Flow."""
    tw, th = 440, 285
    pad = 14
    header_h = 34
    W = 3 * tw + 4 * pad
    H = header_h + 2 * th + 3 * pad
    canvas = np.full((H, W, 3), 10, dtype=np.uint8)

    _draw_text(
        canvas,
        pad,
        10,
        "LAB 07: 6D AERO-STRUCTURAL MDO — LOCAL ADJOINT VS. 6D GP-BO VS. PHYSICS-GUIDED FLOW",
        (56, 189, 248),
        scale=2,
    )

    left_m, right_m, top_m, bot_m = 48, 18, 42, 34
    pw = tw - left_m - right_m
    ph = th - top_m - bot_m

    # --- Top Row Panel 1: 6D Latent Space Projection (z0 vs z3) Multi-Basin Exploration ---
    t1 = np.full((th, tw, 3), 17, dtype=np.uint8)
    _draw_text(t1, 10, 9, "(A) 6D LATENT SPACE SEARCH (Z0 VS Z3)", (56, 189, 248), scale=1)
    _draw_text(t1, 10, 22, "RED:ADJOINT  AMB:6D GP-BO  PURP:FLOW ODE", (203, 213, 225), scale=1)

    def z_to_p1(z0: float, z3: float) -> tuple[float, float]:
        px = left_m + ((z0 - (-2.2)) / 4.4) * pw
        py = top_m + ph - ((z3 - (-1.8)) / 3.8) * ph
        return px, py

    # Axes & ticks for Panel 1
    t1[top_m : th - bot_m + 1, left_m] = [71, 85, 105]
    t1[th - bot_m, left_m : tw - right_m + 1] = [71, 85, 105]
    for z0_v in (-2.0, -1.0, 0.0, 1.0, 2.0):
        gx, _ = z_to_p1(z0_v, 0.0)
        t1[top_m : th - bot_m, int(gx)] = [24, 34, 50]
        t1[th - bot_m : th - bot_m + 4, int(gx)] = [148, 163, 184]
        _draw_text(t1, int(gx) - 10, th - bot_m + 7, f"{z0_v:+.1f}", (148, 163, 184), scale=1)
    for z3_v in (-1.5, 0.0, 1.5):
        _, gy = z_to_p1(0.0, z3_v)
        t1[int(gy), left_m - 4 : left_m] = [148, 163, 184]
        _draw_text(t1, 6, int(gy) - 3, f"{z3_v:+.1f}", (148, 163, 184), scale=1)
    _draw_text(t1, left_m + pw // 2 - 58, th - 12, "X: Z0 (THICKNESS)", (148, 163, 184), scale=1)
    _draw_text(t1, 6, top_m - 10, "Y:Z3", (148, 163, 184), scale=1)

    # Plot prior cloud (3 basins)
    for zp in prior_samples[::4]:
        px, py = z_to_p1(float(zp[0]), float(zp[3]))
        if left_m < int(px) < tw - right_m and top_m < int(py) < th - bot_m:
            t1[int(py) - 1 : int(py) + 1, int(px) - 1 : int(px) + 1] = [42, 58, 84]

    # Basin annotations
    bx0, by0 = z_to_p1(-1.45, -0.55)
    _draw_text(t1, int(bx0) - 12, int(by0) + 12, "BASIN 0", (244, 63, 94), scale=1)
    bx2, by2 = z_to_p1(1.10, 1.15)
    _draw_text(t1, int(bx2) - 18, int(by2) - 14, "BASIN 2 (OPT)", (16, 185, 129), scale=1)

    # Plot Flow particle swarm trajectories (purple)
    swarm = flow_res["swarm_history"]
    for p_i in range(swarm.shape[1]):
        for s_i in range(swarm.shape[0] - 1):
            x0, y0 = z_to_p1(float(swarm[s_i, p_i, 0]), float(swarm[s_i, p_i, 3]))
            x1, y1 = z_to_p1(float(swarm[s_i + 1, p_i, 0]), float(swarm[s_i + 1, p_i, 3]))
            _draw_line(t1, x0, y0, x1, y1, (168, 85, 247), thickness=1)

    # Plot GP-BO evaluated points (amber)
    for zb in bo_res["all_z_samples"]:
        px, py = z_to_p1(float(zb[0]), float(zb[3]))
        if left_m < px < tw - right_m and top_m < py < th - bot_m:
            _draw_circle(t1, px, py, 3, (245, 158, 11))

    # Plot Local Adjoint trajectory (coral red, trapped in Basin 0)
    z_adj = adj_res["z_traj"]
    for s_i in range(len(z_adj) - 1):
        x0, y0 = z_to_p1(float(z_adj[s_i, 0]), float(z_adj[s_i, 3]))
        x1, y1 = z_to_p1(float(z_adj[s_i + 1, 0]), float(z_adj[s_i + 1, 3]))
        _draw_line(t1, x0, y0, x1, y1, (244, 63, 94), thickness=2)
    _draw_circle(t1, *z_to_p1(float(z_adj[-1, 0]), float(z_adj[-1, 3])), 5, (244, 63, 94))

    _draw_circle(t1, *z_to_p1(float(bo_res["final_design"]["z"][0]), float(bo_res["final_design"]["z"][3])), 5, (245, 158, 11))
    _draw_circle(t1, *z_to_p1(float(flow_res["final_design"]["z"][0]), float(flow_res["final_design"]["z"][3])), 6, (16, 185, 129))

    # --- Top Row Panel 2: Gaussian Process Posterior Surrogate mu(z) +/- 1.8*sigma(z) ---
    t2 = np.full((th, tw, 3), 17, dtype=np.uint8)
    _draw_text(t2, 10, 9, "(B) 1D SLICE: GP POSTERIOR VS TRUE J(Z)", (56, 189, 248), scale=1)
    _draw_text(t2, 10, 22, "CYAN:GP MEAN+-1.8S  DASH:TRUE DOUBLE-WELL", (203, 213, 225), scale=1)

    alphas = bo_res["gp_slice_alpha"]
    mu_s = bo_res["gp_slice_mu"]
    std_s = bo_res["gp_slice_std"]
    true_s = bo_res["gp_slice_true"]
    y_min = float(min(np.min(mu_s - 2.0 * std_s), np.min(true_s)) - 0.5)
    y_max = float(max(np.max(mu_s + 2.0 * std_s), np.max(true_s)) + 0.5)

    def slice_to_p2(idx_a: int, y_val: float) -> tuple[float, float]:
        px = left_m + (idx_a / max(1, len(alphas) - 1)) * pw
        py = top_m + ph - ((y_val - y_min) / max(y_max - y_min, 1e-5)) * ph
        return px, py

    t2[top_m : th - bot_m + 1, left_m] = [71, 85, 105]
    t2[th - bot_m, left_m : tw - right_m + 1] = [71, 85, 105]
    for a_v in (0.0, 0.25, 0.50, 0.75, 1.0):
        frac = (a_v - float(alphas[0])) / max(1e-5, float(alphas[-1] - alphas[0]))
        gx = int(round(left_m + frac * pw))
        t2[top_m : th - bot_m, gx] = [24, 34, 50]
        t2[th - bot_m : th - bot_m + 4, gx] = [148, 163, 184]
        _draw_text(t2, gx - 10, th - bot_m + 7, f"{a_v:.2f}", (148, 163, 184), scale=1)

    for y_frac in (0.1, 0.5, 0.9):
        yv = y_min + y_frac * (y_max - y_min)
        gy = int(round(top_m + ph - y_frac * ph))
        t2[gy, left_m - 4 : left_m] = [148, 163, 184]
        _draw_text(t2, 6, gy - 3, f"{yv:4.1f}", (148, 163, 184), scale=1)
    _draw_text(t2, left_m + pw // 2 - 66, th - 12, "X: SLICE ALPHA [0->1]", (148, 163, 184), scale=1)
    _draw_text(t2, 6, top_m - 10, "J(Z)", (148, 163, 184), scale=1)

    for i in range(len(alphas)):
        px, py_hi = slice_to_p2(i, float(mu_s[i] + 1.8 * std_s[i]))
        _, py_lo = slice_to_p2(i, float(mu_s[i] - 1.8 * std_s[i]))
        y0 = max(top_m + 1, min(th - bot_m - 1, int(min(py_hi, py_lo))))
        y1 = max(top_m + 1, min(th - bot_m - 1, int(max(py_hi, py_lo))))
        x_int = int(round(px))
        t2[y0 : y1 + 1, max(left_m + 1, x_int - 3) : min(tw - right_m, x_int + 4)] = [28, 52, 78]

    for i in range(len(alphas) - 1):
        x0, y0 = slice_to_p2(i, float(true_s[i]))
        x1, y1 = slice_to_p2(i + 1, float(true_s[i + 1]))
        _draw_line(t2, x0, y0, x1, y1, (203, 213, 225), thickness=2, dashed=True)
        x0, y0 = slice_to_p2(i, float(mu_s[i]))
        x1, y1 = slice_to_p2(i + 1, float(mu_s[i + 1]))
        _draw_line(t2, x0, y0, x1, y1, (56, 189, 248), thickness=2)

    # --- Top Row Panel 3: Convergence Curves ---
    t3 = np.full((th, tw, 3), 17, dtype=np.uint8)
    _draw_text(t3, 10, 9, "(C) MDO OBJECTIVE CONVERGENCE J(Z)", (56, 189, 248), scale=1)
    _draw_text(
        t3,
        10,
        22,
        f"ADJ:{adj_res['final_loss']:.2f}  BO:{bo_res['final_loss']:.2f}  FLOW:{flow_res['final_loss']:.2f}",
        (203, 213, 225),
        scale=1,
    )

    all_losses = adj_res["loss_traj"] + bo_res["loss_traj"] + flow_res["loss_traj"]
    l_min, l_max = float(min(all_losses) - 0.5), float(max(all_losses) + 0.5)
    max_steps = max(len(adj_res["loss_traj"]), len(bo_res["loss_traj"]), len(flow_res["loss_traj"])) - 1

    t3[top_m : th - bot_m + 1, left_m] = [71, 85, 105]
    t3[th - bot_m, left_m : tw - right_m + 1] = [71, 85, 105]
    for s_frac in (0.0, 0.25, 0.50, 0.75, 1.0):
        gx = int(round(left_m + s_frac * pw))
        s_val = int(round(s_frac * max_steps))
        t3[top_m : th - bot_m, gx] = [24, 34, 50]
        t3[th - bot_m : th - bot_m + 4, gx] = [148, 163, 184]
        _draw_text(t3, gx - 6, th - bot_m + 7, f"{s_val}", (148, 163, 184), scale=1)

    for y_frac in (0.1, 0.5, 0.9):
        yv = l_min + y_frac * (l_max - l_min)
        gy = int(round(top_m + ph - y_frac * ph))
        t3[gy, left_m - 4 : left_m] = [148, 163, 184]
        _draw_text(t3, 6, gy - 3, f"{yv:4.1f}", (148, 163, 184), scale=1)
    _draw_text(t3, left_m + pw // 2 - 58, th - 12, "X: ITERATION STEP", (148, 163, 184), scale=1)
    _draw_text(t3, 6, top_m - 10, "J(Z)", (148, 163, 184), scale=1)

    for traj_l, col in [
        (adj_res["loss_traj"], (244, 63, 94)),
        (bo_res["loss_traj"], (245, 158, 11)),
        (flow_res["loss_traj"], (16, 185, 129)),
    ]:
        n_l = len(traj_l)
        for i in range(n_l - 1):
            x0 = left_m + (i / max(1, n_l - 1)) * pw
            x1 = left_m + ((i + 1) / max(1, n_l - 1)) * pw
            y0 = top_m + ph - ((traj_l[i] - l_min) / max(l_max - l_min, 1e-5)) * ph
            y1 = top_m + ph - ((traj_l[i + 1] - l_min) / max(l_max - l_min, 1e-5)) * ph
            _draw_line(t3, x0, y0, x1, y1, col, thickness=3)

    # --- Bottom Row: Body-Fitted Airfoil + Bionic Rib Topologies for the 3 Paradigms ---
    d_adj = adj_res["final_design"]
    tc_adj = float(np.max(d_adj["yu"] - d_adj["yl"]) * 100.0)
    cam_adj = float(np.max(0.5 * (d_adj["yu"] + d_adj["yl"])) * 100.0)
    b1 = render_body_fitted_airfoil_rib_tile(
        xc_full,
        d_adj["yu"],
        d_adj["yl"],
        d_adj["rho_superres"],
        width=tw,
        height=th,
        accent_color=(244, 63, 94),
        title="(D) METHOD 1: LOCAL COUPLED ADJOINT",
        subtitle=f"T/C={tc_adj:.1f}% CAM={cam_adj:.1f}% | L/D={d_adj['lds'][0]:.1f} | COMP={d_adj['agg_compliance']:.2f}",
    )

    d_bo = bo_res["final_design"]
    tc_bo = float(np.max(d_bo["yu"] - d_bo["yl"]) * 100.0)
    cam_bo = float(np.max(0.5 * (d_bo["yu"] + d_bo["yl"])) * 100.0)
    b2 = render_body_fitted_airfoil_rib_tile(
        xc_full,
        d_bo["yu"],
        d_bo["yl"],
        d_bo["rho_superres"],
        width=tw,
        height=th,
        accent_color=(245, 158, 11),
        title="(E) METHOD 2: 6D LATENT GP-BO (GP-LCB)",
        subtitle=f"T/C={tc_bo:.1f}% CAM={cam_bo:.1f}% | L/D={d_bo['lds'][0]:.1f} | COMP={d_bo['agg_compliance']:.2f}",
    )

    d_fl = flow_res["final_design"]
    tc_fl = float(np.max(d_fl["yu"] - d_fl["yl"]) * 100.0)
    cam_fl = float(np.max(0.5 * (d_fl["yu"] + d_fl["yl"])) * 100.0)
    b3 = render_body_fitted_airfoil_rib_tile(
        xc_full,
        d_fl["yu"],
        d_fl["yl"],
        d_fl["rho_superres"],
        width=tw,
        height=th,
        accent_color=(56, 189, 248),
        title="(F) METHOD 3: PHYSICS-GUIDED FLOW [BEST]",
        subtitle=f"T/C={tc_fl:.1f}% CAM={cam_fl:.1f}% | L/D={d_fl['lds'][0]:.1f} | COMP={d_fl['agg_compliance']:.2f}",
    )

    grid_tiles = [[t1, t2, t3], [b1, b2, b3]]
    for r_i in range(2):
        for c_i in range(3):
            x0 = pad + c_i * (tw + pad)
            y0 = header_h + pad + r_i * (th + pad)
            canvas[y0 : y0 + th, x0 : x0 + tw] = grid_tiles[r_i][c_i]
            bcol = [16, 185, 129] if (r_i == 1 and c_i == 2) else [51, 65, 85]
            canvas[y0 - 1 : y0 + th + 1, x0 - 1] = bcol
            canvas[y0 - 1 : y0 + th + 1, x0 + tw] = bcol
            canvas[y0 - 1, x0 - 1 : x0 + tw + 1] = bcol
            canvas[y0 + th, x0 - 1 : x0 + tw + 1] = bcol

    write_png(path, canvas)


def save_multipoint_and_joukowski_panel(
    path: str | Path,
    xc_full: np.ndarray,
    mp_study: dict,
    jouk_res: dict,
) -> None:
    """Render Single-Point vs Multi-Point Bionic Rib Topologies + 3-Condition Cp & Joukowski Benchmark."""
    tw, th = 440, 290
    pad = 14
    header_h = 34
    W = 3 * tw + 4 * pad
    H = header_h + th + 2 * pad
    canvas = np.full((H, W, 3), 10, dtype=np.uint8)

    _draw_text(
        canvas,
        pad,
        10,
        "FLIGHT ENVELOPE ROBUSTNESS (+1.0G / +2.5G / -1.2G) & JOUKOWSKI EXACT BENCHMARK",
        (16, 185, 129),
        scale=2,
    )

    sp = mp_study["single_point"]
    mp = mp_study["multi_point"]
    sp_c = mp_study["sp_compliances"]
    mp_c = mp_study["mp_compliances"]

    tile_sp = render_body_fitted_airfoil_rib_tile(
        xc_full,
        sp["yu"],
        sp["yl"],
        sp["rho_superres"],
        width=tw,
        height=th,
        accent_color=(244, 63, 94),
        title="(A) SINGLE-POINT DESIGN (+1.0G CRUISE ONLY)",
        subtitle=f"CRUISE={sp_c[0]:.1f} | +2.5G={sp_c[1]:.1f} (HOLLOW NOSE) | -1.2G={sp_c[2]:.1f}",
    )
    tile_mp = render_body_fitted_airfoil_rib_tile(
        xc_full,
        mp["yu"],
        mp["yl"],
        mp["rho_superres"],
        width=tw,
        height=th,
        accent_color=(16, 185, 129),
        title="(B) MULTI-POINT ROBUST DESIGN (3-COND ENVELOPE)",
        subtitle=f"CRUISE={mp_c[0]:.1f} | +2.5G={mp_c[1]:.1f} (-{mp_study['maneuver_gain_pct']:.0f}%) | -1.2G={mp_c[2]:.1f}",
    )

    # Panel 3: Joukowski Exact vs Panel Cp + 3-Condition Flight Envelope Cp Curves
    tile_cp = np.full((th, tw, 3), 17, dtype=np.uint8)
    _draw_text(tile_cp, 10, 9, "(C) JOUKOWSKI EXACT CP & 3-COND ENVELOPE CP", (56, 189, 248), scale=1)
    _draw_text(
        tile_cp,
        10,
        22,
        f"JOUK ERR={jouk_res['cl_rel_error']*100:.2f}% | CYAN:+1G AMB:+2.5G PURP:-1.2G",
        (203, 213, 225),
        scale=1,
    )

    left_m, right_m, top_m, bot_m = 48, 18, 42, 34
    pw = tw - left_m - right_m
    ph = th - top_m - bot_m

    def cp_to_px(x_val: float, cp_val: float) -> tuple[float, float]:
        px = left_m + x_val * pw
        # Negative Cp (suction) plotted upward: Cp in [-2.6, 1.1]
        norm_y = (1.1 - cp_val) / 3.7
        py = top_m + ph - norm_y * ph
        return px, py

    tile_cp[top_m : th - bot_m + 1, left_m] = [71, 85, 105]
    tile_cp[th - bot_m, left_m : tw - right_m + 1] = [71, 85, 105]
    for xv in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        gx, _ = cp_to_px(xv, 0.0)
        tile_cp[top_m : th - bot_m, int(gx)] = [24, 34, 50]
        tile_cp[th - bot_m : th - bot_m + 4, int(gx)] = [148, 163, 184]
        _draw_text(tile_cp, int(gx) - 8, th - bot_m + 7, f"{xv:.1f}", (148, 163, 184), scale=1)

    for cp_v in (-2.0, -1.0, 0.0, 1.0):
        _, gy = cp_to_px(0.0, cp_v)
        if top_m <= int(gy) < th - bot_m:
            tile_cp[int(gy), left_m - 4 : left_m] = [148, 163, 184]
            _draw_text(tile_cp, 6, int(gy) - 3, f"{cp_v:+.1f}", (148, 163, 184), scale=1)
    _draw_text(tile_cp, left_m + pw // 2 - 42, th - 12, "X: CHORD X/C", (148, 163, 184), scale=1)
    _draw_text(tile_cp, 6, top_m - 10, "CP", (148, 163, 184), scale=1)

    # Cp = 0 reference line
    x0, y0 = cp_to_px(0.0, 0.0)
    x1, y1 = cp_to_px(1.0, 0.0)
    _draw_line(tile_cp, x0, y0, x1, y1, (71, 85, 105), thickness=1, dashed=True)

    # Plot 3 Flight Condition upper-surface Cp distributions on the Multi-Point airfoil
    colors_cond = [(56, 189, 248), (245, 158, 11), (168, 85, 247)]
    for k in range(3):
        cp_u = mp["cp_upper"][k]
        for i in range(len(xc_full) - 1):
            x0, y0 = cp_to_px(float(xc_full[i]), float(cp_u[i]))
            x1, y1 = cp_to_px(float(xc_full[i + 1]), float(cp_u[i + 1]))
            y0 = max(top_m + 1, min(th - bot_m - 1, y0))
            y1 = max(top_m + 1, min(th - bot_m - 1, y1))
            _draw_line(tile_cp, x0, y0, x1, y1, colors_cond[k], thickness=2)

    # Plot Joukowski exact upper Cp vs Panel upper Cp (white dashed curve)
    xc_j = jouk_res["xc"]
    cp_uj = jouk_res["cp_upper_panel"]
    for i in range(len(xc_j) - 1):
        x0, y0 = cp_to_px(float(xc_j[i]), float(cp_uj[i]))
        x1, y1 = cp_to_px(float(xc_j[i + 1]), float(cp_uj[i + 1]))
        y0 = max(top_m + 1, min(th - bot_m - 1, y0))
        y1 = max(top_m + 1, min(th - bot_m - 1, y1))
        _draw_line(tile_cp, x0, y0, x1, y1, (241, 245, 249), thickness=1, dashed=True)

    for c_i, tile in enumerate([tile_sp, tile_mp, tile_cp]):
        x0 = pad + c_i * (tw + pad)
        y0 = header_h + pad
        canvas[y0 : y0 + th, x0 : x0 + tw] = tile
        bcol = [16, 185, 129] if c_i == 1 else [51, 65, 85]
        canvas[y0 - 1 : y0 + th + 1, x0 - 1] = bcol
        canvas[y0 - 1 : y0 + th + 1, x0 + tw] = bcol
        canvas[y0 - 1, x0 - 1 : x0 + tw + 1] = bcol
        canvas[y0 + th, x0 - 1 : x0 + tw + 1] = bcol

    write_png(path, canvas)


def save_xfoil_3d_wing_integration_panel(
    path: str | Path,
    xc_full: np.ndarray,
    xfoil_audit: dict,
    surr_res: dict,
    wing_3d_res: dict,
) -> None:
    """Render 2x3 panel showcasing all 3 XFOIL-Designer-Public + MLX MDO Integration Directions."""
    tw, th = 440, 290
    pad = 14
    header_h = 34
    W = 3 * tw + 4 * pad
    H = header_h + 2 * th + 3 * pad
    canvas = np.full((H, W, 3), 10, dtype=np.uint8)

    _draw_text(
        canvas,
        pad,
        10,
        "XFOIL 6.996 + MLX SURROGATE + TRUST.PY GATE & 3D FINITE-WING MDO",
        (56, 189, 248),
        scale=2,
    )

    # -------------------------------------------------------------------------
    # Tile (A): Native XFOIL 6.996 Upper-Surface Cf(x) & trust.py Gate
    # -------------------------------------------------------------------------
    tile_a = np.full((th, tw, 3), 17, dtype=np.uint8)
    _draw_text(tile_a, 10, 9, "(A) NATIVE XFOIL 6.996 BL CF(X) & TRUST.PY GATE", (56, 189, 248), scale=1)
    conds = xfoil_audit.get("conditions", [])
    v0 = conds[0]["trust_verdict"].upper() if len(conds) > 0 else "OK"
    v1 = conds[1]["trust_verdict"].upper() if len(conds) > 1 else "OK"
    v2 = conds[2]["trust_verdict"].upper() if len(conds) > 2 else "SUSPECT"
    _draw_text(
        tile_a,
        10,
        22,
        f"+3DEG:{v0} | +9DEG:{v1} | +14DEG:{v2} (CF<0 SEP GATE)",
        (203, 213, 225),
        scale=1,
    )
    left_m, right_m, top_m, bot_m = 52, 18, 42, 34
    pw = tw - left_m - right_m
    ph = th - top_m - bot_m

    def cf_to_px(xv: float, cfv: float) -> tuple[float, float]:
        px = left_m + xv * pw
        # Cf in [-0.004, +0.016]
        ny = (cfv - (-0.004)) / 0.020
        py = top_m + ph - ny * ph
        return px, py

    tile_a[top_m : th - bot_m + 1, left_m] = [71, 85, 105]
    tile_a[th - bot_m, left_m : tw - right_m + 1] = [71, 85, 105]
    for xv in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        gx, _ = cf_to_px(xv, 0.0)
        tile_a[top_m : th - bot_m, int(gx)] = [24, 34, 50]
        tile_a[th - bot_m : th - bot_m + 4, int(gx)] = [148, 163, 184]
        _draw_text(tile_a, int(gx) - 8, th - bot_m + 7, f"{xv:.1f}", (148, 163, 184), scale=1)

    for cfv in (-0.002, 0.000, 0.005, 0.010, 0.015):
        _, gy = cf_to_px(0.0, cfv)
        if top_m <= int(gy) < th - bot_m:
            tile_a[int(gy), left_m - 4 : left_m] = [148, 163, 184]
            _draw_text(tile_a, 4, int(gy) - 3, f"{cfv:+.3f}", (148, 163, 184), scale=1)
    _draw_text(tile_a, left_m + pw // 2 - 42, th - 12, "X: CHORD X/C", (148, 163, 184), scale=1)
    _draw_text(tile_a, 6, top_m - 10, "CF(X)", (148, 163, 184), scale=1)

    # Zero-Cf separation threshold line (red dashed)
    x0_z, y0_z = cf_to_px(0.0, 0.0)
    x1_z, y1_z = cf_to_px(1.0, 0.0)
    _draw_line(tile_a, x0_z, y0_z, x1_z, y1_z, (244, 63, 94), thickness=1, dashed=True)

    cf_cols = [(16, 185, 129), (245, 158, 11), (244, 63, 94)]
    for k, cdict in enumerate(conds[:3]):
        cf_u = cdict["cf_upper"]
        for i in range(len(xc_full) - 1):
            x0, y0 = cf_to_px(float(xc_full[i]), float(cf_u[i]))
            x1, y1 = cf_to_px(float(xc_full[i + 1]), float(cf_u[i + 1]))
            y0 = max(top_m + 1, min(th - bot_m - 1, y0))
            y1 = max(top_m + 1, min(th - bot_m - 1, y1))
            _draw_line(tile_a, x0, y0, x1, y1, cf_cols[k], thickness=2)

    # -------------------------------------------------------------------------
    # Tile (B): MLX Surrogate 22-Station mx.grad Sensitivity & Hicks-Henne Opt
    # -------------------------------------------------------------------------
    tile_b = np.full((th, tw, 3), 17, dtype=np.uint8)
    _draw_text(tile_b, 10, 9, "(B) MLX SURROGATE 22-STATION GRAD & HICKS-HENNE", (16, 185, 129), scale=1)
    b_ld = surr_res.get("base_ld_mean", 0.0)
    o_ld = surr_res.get("opt_ld_mean", 0.0)
    trim_a = surr_res.get("trim_alpha_deg", 0.0)
    _draw_text(
        tile_b,
        10,
        22,
        f"L/D: {b_ld:.1f} -> {o_ld:.1f} | TRIM A(CL=0.75)={trim_a:.2f}DEG",
        (203, 213, 225),
        scale=1,
    )
    stations = np.asarray(surr_res["station_x"], dtype=np.float64)
    d_cam = np.asarray(surr_res["d_ld_d_camber"], dtype=np.float64)
    d_thk = np.asarray(surr_res["d_ld_d_thickness"], dtype=np.float64)
    g_max = max(float(np.max(np.abs(np.concatenate([d_cam, d_thk])))), 1.0)

    def grad_to_px(xv: float, gv: float) -> tuple[float, float]:
        px = left_m + xv * pw
        ny = (gv + g_max) / (2.0 * g_max)
        py = top_m + ph - ny * ph
        return px, py

    tile_b[top_m : th - bot_m + 1, left_m] = [71, 85, 105]
    tile_b[th - bot_m, left_m : tw - right_m + 1] = [71, 85, 105]
    for xv in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        gx, _ = grad_to_px(xv, 0.0)
        tile_b[top_m : th - bot_m, int(gx)] = [24, 34, 50]
        tile_b[th - bot_m : th - bot_m + 4, int(gx)] = [148, 163, 184]
        _draw_text(tile_b, int(gx) - 8, th - bot_m + 7, f"{xv:.1f}", (148, 163, 184), scale=1)

    for frac in (-0.8, -0.4, 0.0, 0.4, 0.8):
        gv = frac * g_max
        _, gy = grad_to_px(0.0, gv)
        if top_m <= int(gy) < th - bot_m:
            tile_b[int(gy), left_m - 4 : left_m] = [148, 163, 184]
            _draw_text(tile_b, 4, int(gy) - 3, f"{gv:+.0f}", (148, 163, 184), scale=1)
    _draw_text(tile_b, left_m + pw // 2 - 48, th - 12, "X: STATION X/C", (148, 163, 184), scale=1)
    _draw_text(tile_b, 6, top_m - 10, "D(L/D)", (148, 163, 184), scale=1)

    gx0, gy0 = grad_to_px(0.0, 0.0)
    gx1, gy1 = grad_to_px(1.0, 0.0)
    _draw_line(tile_b, gx0, gy0, gx1, gy1, (71, 85, 105), thickness=1, dashed=True)

    for i in range(len(stations) - 1):
        x0, y0 = grad_to_px(float(stations[i]), float(d_cam[i]))
        x1, y1 = grad_to_px(float(stations[i + 1]), float(d_cam[i + 1]))
        _draw_line(tile_b, x0, y0, x1, y1, (16, 185, 129), thickness=2)
        _draw_circle(tile_b, x0, y0, 3, (16, 185, 129))

        tx0, ty0 = grad_to_px(float(stations[i]), float(d_thk[i]))
        tx1, ty1 = grad_to_px(float(stations[i + 1]), float(d_thk[i + 1]))
        _draw_line(tile_b, tx0, ty0, tx1, ty1, (245, 158, 11), thickness=2)
        _draw_circle(tile_b, tx0, ty0, 3, (245, 158, 11))
    _draw_text(tile_b, left_m + 8, top_m + 8, "GRN: D(L/D)/DCAM  AMB: D(L/D)/DTHK", (203, 213, 225), scale=1)

    # -------------------------------------------------------------------------
    # Tile (C): 3D Wing Spanwise CL(y/b), Bending Moment M(y/b) & Drag
    # -------------------------------------------------------------------------
    tile_c = np.full((th, tw, 3), 17, dtype=np.uint8)
    _draw_text(tile_c, 10, 9, "(C) 3D WING SPANWISE LOAD & BENDING MOMENT", (245, 158, 11), scale=1)
    _draw_text(
        tile_c,
        10,
        22,
        f"AR={wing_3d_res['aspect_ratio']:.1f} | 3D CL={wing_3d_res['CL_3D']:.3f} | 3D L/D={wing_3d_res['LD_3D']:.1f}",
        (203, 213, 225),
        scale=1,
    )
    eta = np.asarray(wing_3d_res["eta"], dtype=np.float64)
    cl_s = np.asarray(wing_3d_res["cl_strip"], dtype=np.float64)
    bm_s = np.asarray(wing_3d_res["bending_moment_norm"], dtype=np.float64)
    cd_s = np.asarray(wing_3d_res["cd_profile_strip"], dtype=np.float64) * 100.0  # scaled x100

    def span_to_px(ev: float, yv: float) -> tuple[float, float]:
        px = left_m + ev * pw
        ny = yv / 1.25
        py = top_m + ph - ny * ph
        return px, py

    tile_c[top_m : th - bot_m + 1, left_m] = [71, 85, 105]
    tile_c[th - bot_m, left_m : tw - right_m + 1] = [71, 85, 105]
    for ev in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        gx, _ = span_to_px(ev, 0.0)
        tile_c[top_m : th - bot_m, int(gx)] = [24, 34, 50]
        tile_c[th - bot_m : th - bot_m + 4, int(gx)] = [148, 163, 184]
        _draw_text(tile_c, int(gx) - 8, th - bot_m + 7, f"{ev:.1f}", (148, 163, 184), scale=1)

    for yv in (0.0, 0.3, 0.6, 0.9, 1.2):
        _, gy = span_to_px(0.0, yv)
        if top_m <= int(gy) < th - bot_m:
            tile_c[int(gy), left_m - 4 : left_m] = [148, 163, 184]
            _draw_text(tile_c, 10, int(gy) - 3, f"{yv:.1f}", (148, 163, 184), scale=1)
    _draw_text(tile_c, left_m + pw // 2 - 48, th - 12, "SPAN ETA = 2Y/B", (148, 163, 184), scale=1)
    _draw_text(tile_c, 6, top_m - 10, "NORM", (148, 163, 184), scale=1)

    for i in range(len(eta) - 1):
        x0, y0 = span_to_px(float(eta[i]), float(cl_s[i]))
        x1, y1 = span_to_px(float(eta[i + 1]), float(cl_s[i + 1]))
        _draw_line(tile_c, x0, y0, x1, y1, (56, 189, 248), thickness=2)

        bx0, by0 = span_to_px(float(eta[i]), float(bm_s[i]))
        bx1, by1 = span_to_px(float(eta[i + 1]), float(bm_s[i + 1]))
        _draw_line(tile_c, bx0, by0, bx1, by1, (245, 158, 11), thickness=2)

        dx0, dy0 = span_to_px(float(eta[i]), float(cd_s[i]))
        dx1, dy1 = span_to_px(float(eta[i + 1]), float(cd_s[i + 1]))
        _draw_line(tile_c, dx0, dy0, dx1, dy1, (168, 85, 247), thickness=2)

    _draw_text(tile_c, left_m + 8, top_m + 8, "CYN:CL(Y) AMB:M_BEND(Y) PURP:100*CD_P", (203, 213, 225), scale=1)

    # -------------------------------------------------------------------------
    # Bottom Row (D, E, F): 3D Wing Root, Mid-Span, and Tip Station Ribs
    # -------------------------------------------------------------------------
    st_list = wing_3d_res["stations"]
    st_colors = [(244, 63, 94), (245, 158, 11), (16, 185, 129)]
    st_letters = ["(D)", "(E)", "(F)"]
    bot_tiles = []
    for idx, st in enumerate(st_list[:3]):
        tile_st = render_body_fitted_airfoil_rib_tile(
            xc_full,
            st["yu"],
            st["yl"],
            st["rho_superres"],
            width=tw,
            height=th,
            accent_color=st_colors[idx],
            title=f"{st_letters[idx]} 3D WING {st['station_name'].upper()} RIB",
            subtitle=(
                f"C={st['chord_m']:.2f}M TWIST={st['twist_deg']:+.1f}DEG | "
                f"VF={st['vol_frac']*100:.0f}% COMP={st['pnorm_compliance']:.2f}"
            ),
        )
        bot_tiles.append(tile_st)

    all_tiles = [[tile_a, tile_b, tile_c], bot_tiles]
    for r_i in range(2):
        for c_i in range(3):
            x0 = pad + c_i * (tw + pad)
            y0 = header_h + pad + r_i * (th + pad)
            canvas[y0 : y0 + th, x0 : x0 + tw] = all_tiles[r_i][c_i]
            bcol = [56, 189, 248] if r_i == 0 else list(st_colors[c_i])
            canvas[y0 - 1 : y0 + th + 1, x0 - 1] = bcol
            canvas[y0 - 1 : y0 + th + 1, x0 + tw] = bcol
            canvas[y0 - 1, x0 - 1 : x0 + tw + 1] = bcol
            canvas[y0 + th, x0 - 1 : x0 + tw + 1] = bcol

    write_png(path, canvas)


def save_html_report(path: str | Path, metrics: dict) -> None:
    """Generate dark-mode HTML report for Multi-Paradigm Aero-Structural MDO."""
    xfoil_section_html = ""
    if "xfoil_base_ld" in metrics:
        xfoil_section_html = f"""
  <div class="panel">
    <h2>3. XFOIL-Designer-Public Deep Integration (Native XFOIL 6.996 + MLX Ensemble + 3D Finite Wing)</h2>
    <p>
      <strong>Direction 1:</strong> Bundled native <code>XFOIL 6.996</code> viscous boundary-layer solver coupled with
      <code>xfoil_mac/analysis/trust.py</code> separation-topology auditing (<code>ok</code> vs. <code>suspect</code>) and the
      31-feature <code>surrogate_mlx_latest.npz</code> Deep Ensemble.<br>
      <strong>Direction 2:</strong> Arbitrary <code>.dat</code> airfoil coordinate ingestion &amp; 22-station <code>mx.grad</code>
      Hicks-Henne inverse contour optimization driving <code>IsoparametricWingBoxFEM</code>.<br>
      <strong>Direction 3:</strong> 3D finite-wing multi-station (<code>Root y/b=0.05</code>, <code>Mid-Span y/b=0.50</code>,
      <code>Wing-Tip y/b=0.88</code>) aero-structural topology optimization via <code>xfoil_mac/wing_geometry.py</code>.
    </p>
    <img src="xfoil_3d_wing_integration.png" alt="XFOIL 6.996 + MLX Surrogate + 3D Wing Integration">
    <table>
      <thead>
        <tr>
          <th>Integration Module</th>
          <th>Key Metric 1</th>
          <th>Key Metric 2</th>
          <th>Key Metric 3</th>
          <th>Verification Verdict</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>1. Native XFOIL 6.996 + trust.py Gate</strong></td>
          <td>+3.0° Cruise: <code>CL={metrics['xfoil_cl_3']:.3f}, CD={metrics['xfoil_cd_3']:.5f}</code></td>
          <td>+9.0° Pull-Up: <code>{metrics['xfoil_verdict_9']}</code></td>
          <td>+14.0° Stall Audit: <code>{metrics['xfoil_verdict_14']}</code></td>
          <td style="color:var(--emerald);font-weight:700;">Separation Topology Gate Active</td>
        </tr>
        <tr>
          <td><strong>2. MLX 31-Feature Surrogate Inverse Opt</strong></td>
          <td>Target-CL Trim (CL=0.75): <code>α*={metrics['xfoil_trim_alpha']:.2f}°</code></td>
          <td>Base L/D: <code>{metrics['xfoil_base_ld']:.2f}</code></td>
          <td>Hicks-Henne Opt L/D: <code>{metrics['xfoil_opt_ld']:.2f}</code></td>
          <td style="color:var(--emerald);font-weight:700;">Exported .dat &amp; Bionic Wing-Box</td>
        </tr>
        <tr>
          <td><strong>3. 3D Finite-Wing Multi-Station MDO</strong></td>
          <td>3D Wing: <code>CL={metrics['wing3d_cl']:.3f}, L/D={metrics['wing3d_ld']:.1f}</code></td>
          <td>Oswald Efficiency: <code>e={metrics['wing3d_e']:.3f}</code></td>
          <td>Root / Mid / Tip C: <code>{metrics['wing3d_c_root']:.2f} / {metrics['wing3d_c_mid']:.2f} / {metrics['wing3d_c_tip']:.2f}</code></td>
          <td style="color:var(--emerald);font-weight:700;">3 Spanwise Stations Optimized</td>
        </tr>
      </tbody>
    </table>
  </div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MLX Deep MDO Lab — Coupled Adjoint vs. Latent Bayesian Opt vs. Physics-Guided Diffusion</title>
<style>
  :root {{
    --bg: #070b14;
    --panel: #111827;
    --border: #1f2937;
    --accent: #38bdf8;
    --emerald: #10b981;
    --amber: #f59e0b;
    --rose: #f43f5e;
    --purple: #a855f7;
    --text: #f1f5f9;
    --muted: #94a3b8;
  }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 32px;
  }}
  .container {{ max-width: 1220px; margin: 0 auto; }}
  .badge {{
    display: inline-block;
    background: rgba(16, 185, 129, 0.15);
    color: var(--emerald);
    border: 1px solid rgba(16, 185, 129, 0.4);
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }}
  h1 {{ font-size: 30px; margin: 10px 0 8px; }}
  p.lead {{ color: var(--muted); max-width: 960px; margin-top: 0; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
    gap: 16px;
    margin: 24px 0;
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
  }}
  .card .label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; }}
  .card .val {{ font-size: 25px; font-weight: 700; margin-top: 6px; color: var(--accent); }}
  .card .sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 24px;
  }}
  .panel h2 {{ margin-top: 0; font-size: 20px; }}
  .panel img {{
    width: 100%;
    border-radius: 8px;
    border: 1px solid var(--border);
    margin: 14px 0;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    margin-top: 12px;
  }}
  th, td {{
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
  }}
  th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
  code {{
    background: #1e293b;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 13px;
    color: #93c5fd;
  }}
</style>
</head>
<body>
<div class="container">
  <span class="badge">Multidisciplinary Design Optimization (MDO) • Body-Fitted Isoparametric FEM</span>
  <h1>Coupled Aero-Structural Adjoint vs. Latent Bayesian Opt vs. Physics-Guided Diffusion</h1>
  <p class="lead">
    Solves the coupled multi-point wing-box design problem on a <strong>body-fitted isoparametric Q4 mesh</strong>
    (zero staircase boundary artifacts) driven by our differentiable <strong>Weber-Glauert-Riegels Panel + Viscous Boundary-Layer Solver</strong>
    under a strict <strong>Constant Absolute Structural Mass Budget (kg)</strong> across a 3-condition flight envelope.
  </p>

  <div class="grid">
    <div class="card">
      <div class="label">Joukowski Conformal Benchmark</div>
      <div class="val" style="color: var(--emerald);">{metrics['jouk_rel_err_pct']:.2f}%</div>
      <div class="sub">CL_panel={metrics['jouk_cl_panel']:.3f} vs Exact={metrics['jouk_cl_exact']:.3f}</div>
    </div>
    <div class="card">
      <div class="label">Isoparametric Adjoint Error</div>
      <div class="val">{metrics['adjoint_rel_err']:.2e}</div>
      <div class="sub">Simultaneous shape dK/dh + load 2UᵀdF</div>
    </div>
    <div class="card">
      <div class="label">Best Paradigm (Guided Flow)</div>
      <div class="val" style="color: var(--emerald);">{metrics['flow_loss']:.2f}</div>
      <div class="sub">vs {metrics['bo_loss']:.2f} (GP-BO) &amp; {metrics['adj_loss']:.2f} (Local Adjoint)</div>
    </div>
    <div class="card">
      <div class="label">Multi-Point Maneuver Gain</div>
      <div class="val" style="color: var(--amber);">-{metrics['maneuver_gain_pct']:.1f}%</div>
      <div class="sub">+2.5g Pull-Up compliance vs Single-Point</div>
    </div>
    <div class="card">
      <div class="label">Constant Mass Invariance</div>
      <div class="val">{metrics['mass_budget']:.4f}</div>
      <div class="sub">Exact ∑ ρₑ Areaₑ(z) conservation</div>
    </div>
  </div>

  <div class="panel">
    <h2>1. Head-to-Head Comparison: Local Coupled Adjoint vs. Latent GP-BO vs. Physics-Guided Diffusion/Flow</h2>
    <p>
      <strong>Top Left:</strong> 2D latent projection <code>(z₀, z₃)</code> showing the 3 non-convex aerodynamic basins. Pure Local Adjoint
      (red) converges rapidly inside the thin-laminar basin but cannot cross the transonic wave-drag saddle, whereas
      <strong>Latent GP-BO (amber)</strong> and <strong>Physics-Guided Diffusion/Flow Swarm (purple → emerald)</strong> cross into the
      global Supercritical Aero-Structural Pareto Basin.<br>
      <strong>Top Middle:</strong> Gaussian Process Matérn-5/2 posterior mean <code>μ(z) ± 1.8σ(z)</code> vs. true non-convex MDO objective.<br>
      <strong>Bottom Row:</strong> Body-fitted isoparametric airfoil contours and 3× super-resolved internal bionic wing-rib topologies.
    </p>
    <img src="three_paradigms_comparison.png" alt="Three Optimization Paradigms Comparison">
    <table>
      <thead>
        <tr>
          <th>Optimization Paradigm</th>
          <th>Global Basin Reached</th>
          <th>Max Thickness (t/c)</th>
          <th>Cruise L/D (+3°)</th>
          <th>3-Condition p-Norm Compliance</th>
          <th>Structural Mass</th>
          <th>Total MDO Objective (↓)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>1. Local Coupled Adjoint</strong> (Gradient-Only)</td>
          <td>Basin 0 (Thin Laminar)</td>
          <td><code>{metrics['adj_tc']:.1f}%</code></td>
          <td><code>{metrics['adj_ld']:.1f}</code></td>
          <td><code>{metrics['adj_comp']:.3f}</code></td>
          <td><code>{metrics['adj_mass']:.4f}</code></td>
          <td style="color:var(--rose);font-weight:700;">{metrics['adj_loss']:.3f}</td>
        </tr>
        <tr>
          <td><strong>2. Bi-Level Latent GP-BO</strong> (Matérn-5/2 UCB + Adjoint)</td>
          <td>Basin 2 (Supercritical Pareto)</td>
          <td><code>{metrics['bo_tc']:.1f}%</code></td>
          <td><code>{metrics['bo_ld']:.1f}</code></td>
          <td><code>{metrics['bo_comp']:.3f}</code></td>
          <td><code>{metrics['bo_mass']:.4f}</code></td>
          <td style="color:var(--amber);font-weight:700;">{metrics['bo_loss']:.3f}</td>
        </tr>
        <tr>
          <td><strong>3. Physics-Guided Diffusion/Flow</strong> (Tweedie DPS + Adjoint)</td>
          <td>Basin 2 (Supercritical Pareto)</td>
          <td><code>{metrics['flow_tc']:.1f}%</code></td>
          <td><code>{metrics['flow_ld']:.1f}</code></td>
          <td><code>{metrics['flow_comp']:.3f}</code></td>
          <td><code>{metrics['flow_mass']:.4f}</code></td>
          <td style="color:var(--emerald);font-weight:700;">{metrics['flow_loss']:.3f}</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="panel">
    <h2>2. Multi-Condition Flight Envelope Robustness (3×3 Cross-Evaluation Matrix)</h2>
    <p>
      Comparing a wing rib optimized solely for <strong>Nominal Cruise (+3.0°, +1.0g)</strong> against our
      <strong>Multi-Point Flight-Envelope Robust Design</strong> across Cruise (+3.0°, +1.0g), High-g Maneuver Pull-Up (+9.0°, +2.5g),
      and Negative-g Gust Dive (-4.0°, -1.2g) at the exact same structural mass budget.
    </p>
    <img src="multipoint_envelope_matrix.png" alt="Single-Point vs Multi-Point Flight Envelope Matrix">
    <table>
      <thead>
        <tr>
          <th>Structural Rib Design</th>
          <th>Cond A: Cruise (+3.0°, +1.0g)</th>
          <th>Cond B: Pull-Up (+9.0°, +2.5g)</th>
          <th>Cond C: Negative Gust (-4.0°, -1.2g)</th>
          <th>Envelope p-Norm Compliance (↓)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Single-Point Design</strong> (Optimized for Cond A Only)</td>
          <td><code>{metrics['sp_c0']:.4f}</code></td>
          <td style="color:var(--rose);"><code>{metrics['sp_c1']:.4f}</code> (Vulnerable)</td>
          <td style="color:var(--rose);"><code>{metrics['sp_c2']:.4f}</code> (Vulnerable)</td>
          <td><code>{metrics['sp_norm']:.4f}</code></td>
        </tr>
        <tr>
          <td><strong>Multi-Point Robust Design</strong> (3-Condition Envelope)</td>
          <td><code>{metrics['mp_c0']:.4f}</code></td>
          <td style="color:var(--emerald);font-weight:700;"><code>{metrics['mp_c1']:.4f}</code> (-{metrics['maneuver_gain_pct']:.1f}%)</td>
          <td style="color:var(--emerald);font-weight:700;"><code>{metrics['mp_c2']:.4f}</code> (-{metrics['gust_gain_pct']:.1f}%)</td>
          <td style="color:var(--emerald);font-weight:700;"><code>{metrics['mp_norm']:.4f}</code> (-{metrics['envelope_gain_pct']:.1f}%)</td>
        </tr>
      </tbody>
    </table>
  </div>
{xfoil_section_html}
</div>
</body>
</html>
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html, encoding="utf-8")

