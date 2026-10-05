"""Section lofts for appearance previews, independent of AVL's thin-surface
model.
"""

import math
import numpy as np


def section_profile(section: dict, samples: int = 81) -> np.ndarray:
    """Common upper-TE / LE / lower-TE grid, retaining finite TE thickness.

    NACA thickness and vertical camber addition follow the formulas in the
    bundled XFOIL source/Xfoil/src/naca.f (NACA4, standard finite trailing
    edge).
    File sections use their validated, normalized coordinates, not a NACA
    proxy.
    """
    x = (1 - np.cos(np.linspace(0, math.pi, samples))) / 2
    if "naca" in section:
        code = str(section["naca"])
        m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
        thickness = (
            5
            * t
            * (
                0.2969 * np.sqrt(x)
                - 0.126 * x
                - 0.3516 * x**2
                + 0.2843 * x**3
                - 0.1015 * x**4
            )
        )
        camber = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x * x)
        if p:
            forward = x < p
            camber[forward] = m / p**2 * (2 * p * x[forward] - x[forward] ** 2)
        upper, lower = camber + thickness, camber - thickness
    else:
        points = np.asarray(section["coordinates"], dtype=float)
        leading = int(np.argmin(points[:, 0]))
        upper_points, lower_points = (
            points[: leading + 1][::-1],
            points[leading:],
        )
        upper = np.interp(x, upper_points[:, 0], upper_points[:, 1])
        lower = np.interp(x, lower_points[:, 0], lower_points[:, 1])
    return np.column_stack(
        (np.r_[x[::-1], x[1:]], np.r_[upper[::-1], lower[1:]])
    )


def section_ring(section: dict, span_axis, samples: int = 81) -> np.ndarray:
    """Illustrate incidence by rotating the physical contour about its LE.

    This display rotation is not written back to AVL geometry. Thickness
    follows
    the local normal, so vertical and dihedral surfaces do not collapse to
    lines.
    """
    axis = np.array(span_axis, dtype=float, copy=True)
    axis[0] = 0
    axis = axis / np.linalg.norm(axis)
    normal = np.cross([1.0, 0.0, 0.0], axis)
    angle = math.radians(section["twist"])
    chord_direction = (
        np.array([math.cos(angle), 0, 0]) - math.sin(angle) * normal
    )
    thickness_direction = (
        np.array([math.sin(angle), 0, 0]) + math.cos(angle) * normal
    )
    profile = section_profile(section, samples)
    origin = np.array([section[k] for k in ("x", "y", "z")])
    return origin + section["chord"] * (
        profile[:, :1] * chord_direction + profile[:, 1:] * thickness_direction
    )


def surface_lofts(
    surface: dict, samples: int = 81, span_steps: int = 17
) -> list[np.ndarray]:
    """Linearly join corresponding points on each section, then reflect if
    requested.
    """
    sections = surface["sections"]
    rings = []
    for i, section in enumerate(sections):
        left, right = (
            sections[max(0, i - 1)],
            sections[min(i + 1, len(sections) - 1)],
        )
        axis = np.array([0.0, right["y"] - left["y"], right["z"] - left["z"]])
        if np.linalg.norm(axis) == 0:
            neighbor = (
                sections[i + 1] if i + 1 < len(sections) else sections[i - 1]
            )
            axis = np.array(
                [
                    0.0,
                    neighbor["y"] - section["y"],
                    neighbor["z"] - section["z"],
                ]
            )
        rings.append(section_ring(section, axis, samples))
    stations = []
    for left, right in zip(rings, rings[1:]):
        for fraction in np.linspace(0, 1, span_steps, endpoint=False):
            stations.append((1 - fraction) * left + fraction * right)
    loft = np.array([*stations, rings[-1]])
    return [loft, loft * np.array([1, -1, 1])] if surface["mirror"] else [loft]


def loft_triangles(loft: np.ndarray, reflected: bool = False) -> np.ndarray:
    """Triangulate the skin, including the finite trailing-edge closure."""
    a = loft[:-1].reshape(-1, 3)
    b = loft[1:].reshape(-1, 3)
    c = np.roll(loft[1:], -1, axis=1).reshape(-1, 3)
    d = np.roll(loft[:-1], -1, axis=1).reshape(-1, 3)
    faces = np.concatenate(
        (np.stack((a, b, c), axis=1), np.stack((a, c, d), axis=1))
    )
    return faces[:, ::-1] if reflected else faces
