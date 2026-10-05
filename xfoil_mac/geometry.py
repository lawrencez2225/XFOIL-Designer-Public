"""Conservative Selig-coordinate inspection and explicitly saved repair
previews.
"""

from __future__ import annotations

import math
from pathlib import Path

from .data import atomic_json, atomic_text, sha256_file


def load_coordinates(path: Path) -> list[tuple[float, float]]:
    points = []
    for number, raw in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        text = raw.strip()
        if not text or text.startswith(("#", "!")):
            continue
        fields = text.split()
        try:
            point = tuple(float(v.replace("D", "E")) for v in fields[:2])
            if len(point) != 2:
                raise ValueError
        except ValueError:
            if not points:
                continue  # Header, before the first coordinate.
            raise ValueError(
                f"Invalid coordinate at line {number}: {path}"
            ) from None
        if not all(math.isfinite(v) for v in point):
            raise ValueError(f"Nonfinite coordinate at line {number}: {path}")
        points.append(point)
    if len(points) < 5:
        raise ValueError(
            "Need at least five coordinates in trailing-edge / "
            "leading-edge / trailing-edge order"
        )
    return points


def normalize(points):
    """Use the two contour ends as TE and the farthest contour point as LE."""
    if len(points) < 5 or not all(math.isfinite(v) for p in points for v in p):
        raise ValueError("Need at least five finite coordinate pairs")
    te = tuple((points[0][i] + points[-1][i]) / 2 for i in (0, 1))
    le = max(points, key=lambda p: math.dist(p, te))
    dx, dy = te[0] - le[0], te[1] - le[1]
    chord = math.hypot(dx, dy)
    if chord <= 1e-12:
        raise ValueError("Zero chord")
    clean = [
        (
            ((x - le[0]) * dx + (y - le[1]) * dy) / chord**2,
            (-(x - le[0]) * dy + (y - le[1]) * dx) / chord**2,
        )
        for x, y in points
    ]
    return clean, {
        "chord": chord,
        "rotation_deg": math.degrees(math.atan2(dy, dx)),
        "leading_edge": list(le),
    }


def _deduplicate(points):
    clean = []
    for p in points:
        if not clean or math.dist(p, clean[-1]) > 1e-10:
            clean.append(p)
    return clean


def _intersections(points):
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on(a, b, p):
        return abs(cross(a, b, p)) <= 1e-12 and all(
            min(a[k], b[k]) - 1e-12 <= p[k] <= max(a[k], b[k]) + 1e-12
            for k in (0, 1)
        )

    contour = (
        points
        if math.dist(points[0], points[-1]) < 1e-10
        else [*points, points[0]]
    )
    hits = []
    for i, (a, b) in enumerate(zip(contour, contour[1:])):
        for j in range(i + 2, len(contour) - 1):
            if i == 0 and j == len(contour) - 2:
                continue
            c, d = contour[j : j + 2]
            if (
                cross(a, b, c) * cross(a, b, d) < -1e-20
                and cross(c, d, a) * cross(c, d, b) < -1e-20
                or any((on(a, b, c), on(a, b, d), on(c, d, a), on(c, d, b)))
            ):
                hits.append([i, j])
    return hits


def inspect_points(points) -> tuple[dict, list]:
    import numpy as np

    normalized, transform = normalize(points)
    clean = _deduplicate(normalized)
    leading = min(range(len(clean)), key=lambda i: clean[i][0])
    issues, warnings = [], []
    if len(clean) < 5 or len(set(clean)) < 4:
        issues.append(
            "Too few distinct contour points after removing consecutive "
            "duplicates"
        )
    if leading in (0, len(clean) - 1):
        issues.append("Contour must start and end at the trailing edge")
    first, second = clean[: leading + 1], clean[leading:]
    if any(b[0] > a[0] + 1e-7 for a, b in zip(first, first[1:])) or any(
        b[0] < a[0] - 1e-7 for a, b in zip(second, second[1:])
    ):
        issues.append(
            "Surface x coordinates reverse direction; reorder or inspect "
            "the contour manually"
        )
    if (
        math.dist(clean[0], clean[-1]) > 0.1
        or min(clean[0][0], clean[-1][0]) < 0.9
    ):
        issues.append(
            "Contour endpoints do not form a plausible trailing edge"
        )
    crossings = _intersections(clean)
    if crossings:
        issues.append("Self-intersection or nonadjacent duplicate points")
    reversed_order = bool(
        first
        and second
        and sum(p[1] for p in first) / len(first)
        < sum(p[1] for p in second) / len(second)
    )
    if reversed_order:
        clean.reverse()
        leading = len(clean) - 1 - leading
    metrics = {}
    if not issues:
        upper, lower = clean[: leading + 1], clean[leading:]
        xs = np.linspace(0, min(upper[0][0], lower[-1][0]), 1001)
        yu = np.interp(
            xs, [p[0] for p in upper[::-1]], [p[1] for p in upper[::-1]]
        )
        yl = np.interp(xs, [p[0] for p in lower], [p[1] for p in lower])
        thickness = yu - yl
        if float(thickness.min()) < -1e-6:
            issues.append("Upper and lower surfaces cross")
        if float(thickness.max()) <= 1e-8:
            issues.append("Zero thickness")
        metrics = {
            "thickness_ratio": float(thickness.max()),
            "thickness_x": float(xs[thickness.argmax()]),
            "max_camber": float(
                ((yu + yl) / 2)[np.abs((yu + yl) / 2).argmax()]
            ),
            "trailing_edge_gap": math.dist(clean[0], clean[-1]),
        }
    duplicate_count = len(points) - len(clean)
    if duplicate_count:
        warnings.append(
            f"{duplicate_count} consecutive duplicate points can be removed"
        )
    if reversed_order:
        warnings.append("Order will be reversed to upper-TE / LE / lower-TE")
    if (
        abs(transform["chord"] - 1) > 1e-6
        or abs(transform["rotation_deg"]) > 0.01
        or math.hypot(*transform["leading_edge"]) > 1e-6
    ):
        warnings.append(
            "Translation, rotation and unit-chord normalization will be "
            "applied"
        )
    sharp = []
    for i in range(1, len(clean) - 1):
        u = (clean[i][0] - clean[i - 1][0], clean[i][1] - clean[i - 1][1])
        v = (clean[i + 1][0] - clean[i][0], clean[i + 1][1] - clean[i][1])
        angle = math.degrees(
            math.acos(
                max(
                    -1,
                    min(
                        1,
                        (u[0] * v[0] + u[1] * v[1])
                        / (math.hypot(*u) * math.hypot(*v)),
                    ),
                )
            )
        )
        if angle > 45 and abs(i - leading) > 2:
            sharp.append(i)
    if sharp:
        warnings.append(
            "Sharp contour corners need review; no automatic smoothing "
            "is applied"
        )
    return {
        "valid": not issues,
        "issues": issues,
        "warnings": warnings,
        "points": len(points),
        "repaired_points": len(clean),
        "duplicate_count": duplicate_count,
        "reversed": reversed_order,
        "intersections": crossings,
        "sharp_corners": sharp,
        "transform": transform,
        **metrics,
    }, clean


def inspect_file(
    source: Path, destination: Path, repair: bool = False
) -> Path:
    import matplotlib.pyplot as plt

    source, destination = (
        source.expanduser().resolve(),
        destination.expanduser().resolve(),
    )
    outputs = [
        destination / name
        for name in (
            "geometry_report.json",
            "geometry_preview.png",
            "repaired.dat",
        )
    ]
    if source in outputs:
        raise ValueError("Output must not overwrite the input coordinate file")
    points = load_coordinates(source)
    report, clean = inspect_points(points)
    report.update(
        source=str(source),
        source_sha256=sha256_file(source),
        repair_requested=repair,
        repaired_file=None,
    )
    destination.mkdir(parents=True, exist_ok=True)
    if repair and report["valid"]:
        atomic_text(
            destination / "repaired.dat",
            "XFOIL-MAC normalized copy\n"
            + "".join(f"{x:.10g} {y:.10g}\n" for x, y in clean),
        )
        report["repaired_file"] = "repaired.dat"
    else:
        (destination / "repaired.dat").unlink(missing_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)
    axes[0].plot(
        *zip(*points), ".-", label="Original coordinates", linewidth=1
    )
    axes[1].plot(
        *zip(*normalize(points)[0]),
        ".-",
        label="Original after coordinate transform",
        alpha=0.5,
    )
    axes[1].plot(*zip(*clean), "-", label="Proposed copy", linewidth=1.4)
    for ax in axes:
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_title(source.name)
    axes[1].set_title(
        "Normalization / order / duplicate removal only; shape is not smoothed"
    )
    fig.savefig(destination / "geometry_preview.png", dpi=160)
    plt.close(fig)
    atomic_json(destination / "geometry_report.json", report)
    return destination / "geometry_report.json"
