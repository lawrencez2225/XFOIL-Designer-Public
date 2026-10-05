"""Plot geometry, Cp, pressure vectors and polars from validated data."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, TypedDict

from .data import alpha_file_label, gapped_rows, parse_cp_file, row_for_alpha

GEOMETRY_COLOR = "#0000ce"
UPPER_SURFACE_COLOR = "#2563eb"
LOWER_SURFACE_COLOR = "#dc2626"
RESULT_PLOT_FILENAMES = (
    "cp_distribution.png",
    "cl_vs_alpha.png",
    "cd_vs_alpha.png",
    "cm_vs_alpha.png",
    "cl_vs_cd.png",
    "lift_to_drag_vs_alpha.png",
)


class PlotOutputs(TypedDict):
    result_plots: list[Path]
    pressure_vectors: list[Path]


def write_geometry_plot(
    points: Sequence[tuple[float, float]], out_dir: Path, title: str
) -> None:
    if not points:
        return
    import matplotlib.pyplot as plt

    xs, ys = zip(*points)
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    ax.plot(xs, ys, color=GEOMETRY_COLOR, linewidth=2.4)
    ax.axhline(0, color="#94a3b8", linewidth=1, linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.set_title(f"{title} geometry")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    fig.savefig(out_dir / "geometry.png", dpi=180)
    plt.close(fig)


def final_alpha_label(rows: Sequence[dict[str, float]]) -> str:
    if not rows:
        return "alpha = n/a"
    value = rows[-1].get("alpha")
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"alpha = {value:g} deg"
    return "alpha = n/a"


def pressure_vector_plot_limits(
    geometry_points: Sequence[tuple[float, float]],
    xs: Sequence[float],
    ys: Sequence[float],
    us: Sequence[float],
    vs: Sequence[float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_values = [point[0] for point in geometry_points]
    y_values = [point[1] for point in geometry_points]
    x_values.extend(xs)
    y_values.extend(ys)
    x_values.extend(x + u for x, u in zip(xs, us))
    y_values.extend(y + v for y, v in zip(ys, vs))
    if not x_values or not y_values:
        return (-0.05, 1.05), (-0.1, 0.1)

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    x_pad = max(0.04, x_span * 0.08)
    y_pad = max(0.035, y_span * 0.18)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def pressure_vector_components(
    cp_rows: Sequence[tuple[float, float]],
    geometry_points: Sequence[tuple[float, float]],
    alpha: float,
    max_vectors: int = 85,
) -> tuple[
    list[tuple[float, float]],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    n = min(len(cp_rows), len(geometry_points))
    angle = math.radians(alpha)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotated = [
        (cosine * x + sine * y, cosine * y - sine * x)
        for x, y in geometry_points[:n]
    ]
    if n < 3:
        return rotated, [], [], [], []

    cps = [row[1] for row in cp_rows[:n]]
    max_abs_cp = max(max(abs(value) for value in cps), 1e-9)
    vector_scale = 0.11 / max_abs_cp
    step = max(1, math.ceil((n - 2) / max(1, max_vectors)))
    xs: list[float] = []
    ys: list[float] = []
    us: list[float] = []
    vs: list[float] = []
    for index in range(1, n - 1, step):
        tangent_x = rotated[index + 1][0] - rotated[index - 1][0]
        tangent_y = rotated[index + 1][1] - rotated[index - 1][1]
        tangent_length = math.hypot(tangent_x, tangent_y)
        if tangent_length <= 0.0:
            continue
        normal_x = tangent_y / tangent_length
        normal_y = -tangent_x / tangent_length
        delta_x = -cps[index] * vector_scale * normal_x
        delta_y = -cps[index] * vector_scale * normal_y
        surface_x, surface_y = rotated[index]
        if cps[index] < 0.0:
            tail_x, tail_y = surface_x, surface_y
        else:
            tail_x, tail_y = surface_x - delta_x, surface_y - delta_y
        xs.append(tail_x)
        ys.append(tail_y)
        us.append(delta_x)
        vs.append(delta_y)
    return rotated, xs, ys, us, vs


def polar_row_for_alpha(
    rows: Sequence[dict[str, float]], alpha: float
) -> dict[str, float]:
    return row_for_alpha(rows, alpha) or {"alpha": alpha}


def write_pressure_vector_plot(
    cp_rows: Sequence[tuple[float, float]],
    geometry_points: Sequence[tuple[float, float]],
    polar_rows: Sequence[dict[str, float]],
    out_dir: Path,
    title: str,
    alpha: float,
) -> Path | None:
    if not cp_rows or not geometry_points:
        return None
    rotated_geometry, xs, ys, us, vs = pressure_vector_components(
        cp_rows,
        geometry_points,
        alpha,
    )
    if len(rotated_geometry) < 3:
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.5, 6.4), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.20)
    geometry_x, geometry_y = zip(*rotated_geometry)
    ax.plot(geometry_x, geometry_y, color=GEOMETRY_COLOR, linewidth=2.2)
    ax.quiver(
        xs,
        ys,
        us,
        vs,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#111827",
        width=0.0022,
        headwidth=3.6,
        headlength=4.8,
        clip_on=False,
    )
    xlim, ylim = pressure_vector_plot_limits(rotated_geometry, xs, ys, us, vs)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    alpha_rows = [polar_row_for_alpha(polar_rows, alpha)]
    ax.set_title(f"{title} pressure vectors ({final_alpha_label(alpha_rows)})")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    fig.text(
        0.5,
        0.055,
        f"{final_alpha_label(alpha_rows)}; signed Cp direction matches "
        f"XFOIL; vector length is proportional to |Cp|.",
        ha="center",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
        },
    )
    vector_dir = out_dir / "pressure_vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    path = vector_dir / f"pressure_vectors_alpha_{alpha_file_label(alpha)}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def split_cp_surfaces(
    cp_rows: Sequence[tuple[float, float]],
) -> dict[str, list[tuple[float, float]]]:
    if len(cp_rows) < 3:
        points = sorted(list(cp_rows), key=lambda point: point[0])
        return {"upper": points, "lower": []}
    leading_edge_index = min(
        range(len(cp_rows)), key=lambda index: cp_rows[index][0]
    )
    first_side = list(cp_rows[: leading_edge_index + 1])
    second_side = list(cp_rows[leading_edge_index:])
    if len(first_side) <= 1 or len(second_side) <= 1:
        return {
            "upper": sorted(list(cp_rows), key=lambda point: point[0]),
            "lower": [],
        }
    upper = sorted(first_side, key=lambda point: point[0])
    lower = sorted(second_side, key=lambda point: point[0])
    return {"upper": upper, "lower": lower}


def cp_surface_plot_series(
    cp_series: Sequence[tuple[float, Sequence[tuple[float, float]]]],
) -> list[dict[str, object]]:
    plot_series: list[dict[str, object]] = []
    for alpha, cp_rows in cp_series:
        surfaces = split_cp_surfaces(cp_rows)
        for surface_name, color in [
            ("upper", UPPER_SURFACE_COLOR),
            ("lower", LOWER_SURFACE_COLOR),
        ]:
            points = surfaces[surface_name]
            if not points:
                continue
            plot_series.append(
                {
                    "alpha": alpha,
                    "surface": surface_name,
                    "points": points,
                    "color": color,
                    "label": f"{surface_name} alpha={alpha:g} deg",
                }
            )
    return plot_series


def cp_point_near_x(
    points: Sequence[tuple[float, float]], target_x: float
) -> tuple[float, float]:
    return min(points, key=lambda point: abs(point[0] - target_x))


def cp_alpha_label_positions(
    surface_series: Sequence[dict[str, object]],
    label_surface: str = "upper",
    target_x: float = 0.18,
    text_x: float = 0.30,
    minimum_gap: float = 0.0,
) -> list[dict[str, object]]:
    candidates = [
        item for item in surface_series if item["surface"] == label_surface
    ]
    if not candidates:
        candidates = list(surface_series)
    labels: list[dict[str, object]] = []
    for item in sorted(candidates, key=lambda value: float(value["alpha"])):
        points = item["points"]
        if not points:
            continue
        xy = cp_point_near_x(points, target_x)
        labels.append(
            {
                "alpha": item["alpha"],
                "label": f"{float(item['alpha']):g} deg",
                "xy": xy,
                "xytext": (text_x, xy[1]),
                "color": item["color"],
            }
        )
    if len(labels) <= 1:
        return labels

    ordered_by_y = sorted(labels, key=lambda item: item["xy"][1])
    y_values = [float(item["xy"][1]) for item in ordered_by_y]
    y_span = max(max(y_values) - min(y_values), 1e-6)
    dense_labels = len(labels) > 12
    column_count = 2 if dense_labels else 1
    if dense_labels:
        column_start = (
            text_x - 0.03 if label_surface == "upper" else text_x + 0.07
        )
        for index, item in enumerate(ordered_by_y):
            item["xytext"] = (
                column_start + (index % column_count) * 0.05,
                item["xytext"][1],
            )

    min_gap = max(0.16 if dense_labels else 0.10, y_span * 0.045, minimum_gap)
    for column_index in range(column_count):
        previous_y: float | None = None
        for item in ordered_by_y[column_index::column_count]:
            xytext_x, xytext_y = item["xytext"]
            y_value = float(xytext_y)
            if previous_y is not None and y_value - previous_y < min_gap:
                y_value = previous_y + min_gap
            item["xytext"] = (xytext_x, y_value)
            previous_y = y_value
    return labels


def annotate_cp_alpha_labels(
    cp_ax, surface_series: Sequence[dict[str, object]]
) -> None:
    gap = abs(cp_ax.get_ylim()[1] - cp_ax.get_ylim()[0]) * 0.025
    positions = []
    for label_surface in ("upper", "lower"):
        labels = cp_alpha_label_positions(
            surface_series,
            label_surface=label_surface,
            text_x=0.30 if label_surface == "upper" else 0.55,
            minimum_gap=gap,
        )
        for label in labels:
            positions.append(label["xytext"])
            cp_ax.annotate(
                label["label"],
                label["xy"],
                xytext=label["xytext"],
                textcoords="data",
                fontsize=7,
                color=label["color"],
                ha="left",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": label["color"],
                    "alpha": 0.82,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": label["color"],
                    "linewidth": 0.6,
                    "alpha": 0.72,
                },
            )
    if positions:
        cp_ax.update_datalim(positions)
        cp_ax.autoscale_view()


def finite_lift_to_drag_rows(
    rows: Sequence[dict[str, float]],
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for row in rows:
        alpha = row.get("alpha")
        cl = row.get("CL")
        cd = row.get("CD")
        values = (alpha, cl, cd)
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in values
        ):
            continue
        if cd == 0.0:
            continue
        ratio = cl / cd
        if math.isfinite(ratio):
            result.append((float(alpha), ratio))
    return result


def write_cp_distribution_plot(
    cp_series: Sequence[tuple[float, Sequence[tuple[float, float]]]],
    out_dir: Path,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, cp_ax = plt.subplots(figsize=(11.8, 7.2), constrained_layout=True)
    fig.suptitle(f"{title}\nCp distribution and envelope", fontsize=13)
    if cp_series:
        surface_series = cp_surface_plot_series(cp_series)
        envelope_x = np.linspace(0, 1, 250)
        interpolated = []
        legend_seen = set()
        for item in surface_series:
            xs = np.array([p[0] for p in item["points"]])
            cps = np.array([p[1] for p in item["points"]])
            surface = item["surface"]
            legend_label = (
                f"{surface} surface" if surface not in legend_seen else None
            )
            legend_seen.add(surface)
            cp_ax.plot(
                xs,
                cps,
                color=item["color"],
                linewidth=1.4,
                alpha=0.78,
                label=legend_label,
            )
            unique_x, indices = np.unique(xs, return_index=True)
            if len(unique_x) >= 2:
                interpolated.append(
                    np.interp(
                        envelope_x,
                        unique_x,
                        cps[indices],
                        left=np.nan,
                        right=np.nan,
                    )
                )
        if interpolated:
            values = np.array(interpolated)
            valid = np.any(np.isfinite(values), axis=0)
            cp_ax.fill_between(
                envelope_x[valid],
                np.nanmin(values[:, valid], axis=0),
                np.nanmax(values[:, valid], axis=0),
                color="#64748b",
                alpha=0.18,
                label="Cp envelope",
            )
        labeled = surface_series
        if len(cp_series) > 18:
            indices = sorted(
                {round(i * (len(cp_series) - 1) / 8) for i in range(9)}
            )
            label_alphas = {cp_series[i][0] for i in indices}
            labeled = [
                item
                for item in surface_series
                if item["alpha"] in label_alphas
            ]
            cp_ax.text(
                0.02,
                0.04,
                f"{len(cp_series)} alpha curves; {len(label_alphas)} labeled",
                transform=cp_ax.transAxes,
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "#cbd5e1",
                },
            )
        annotate_cp_alpha_labels(cp_ax, labeled)
        cp_ax.legend(loc="upper right", fontsize=8)
    else:
        cp_ax.text(
            0.05, 0.5, "No converged Cp data available.", color="#b91c1c"
        )
    cp_ax.axhline(0, color="#94a3b8", linewidth=1)
    cp_ax.invert_yaxis()
    cp_ax.grid(True, color="#e2e8f0", linewidth=0.8)
    cp_ax.set(
        xlabel="x/c", ylabel="Cp", title="Upper/lower surface Cp with envelope"
    )
    path = out_dir / "cp_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_polar_chart(
    out_dir: Path,
    filename: str,
    x_values: Sequence[float],
    y_values: Sequence[float],
    alpha_labels: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
    scientific_x: bool = False,
    scientific_y: bool = False,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(8.6, 6.2), constrained_layout=True)
    if (
        x_values
        and y_values
        and any(
            math.isfinite(x) and math.isfinite(y)
            for x, y in zip(x_values, y_values)
        )
    ):
        ax.plot(
            x_values,
            y_values,
            color=color,
            marker="o",
            markersize=5.5,
            linewidth=1.8,
        )
        if len(alpha_labels) <= 9:
            for x_value, y_value, alpha in zip(
                x_values, y_values, alpha_labels
            ):
                if not (math.isfinite(x_value) and math.isfinite(y_value)):
                    continue
                ax.annotate(
                    f"{alpha:g}°",
                    (x_value, y_value),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    color="#334155",
                )
    else:
        ax.text(
            0.5,
            0.5,
            "No converged data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#b91c1c",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    if scientific_x:
        ax.ticklabel_format(
            axis="x", style="sci", scilimits=(-2, 2), useOffset=False
        )
    if scientific_y:
        ax.ticklabel_format(
            axis="y", style="sci", scilimits=(-2, 2), useOffset=False
        )
    if xlabel == "alpha (deg)":
        from .analysis.diagnostics import mark_missing

        mark_missing(
            ax,
            [
                a
                for a, y in zip(alpha_labels, y_values)
                if not math.isfinite(y)
            ],
        )
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_polar_result_plots(
    polar_rows: Sequence[dict[str, float]],
    out_dir: Path,
    title: str,
    requested_alphas: Sequence[float] | None = None,
) -> list[Path]:
    ordered = (
        gapped_rows(polar_rows, requested_alphas)
        if requested_alphas is not None
        else sorted(polar_rows, key=lambda row: row["alpha"])
    )
    alpha = [row["alpha"] for row in ordered]
    cl = [row["CL"] for row in ordered]
    cd = [row["CD"] for row in ordered]
    cm = [row["CM"] for row in ordered]
    ld_alpha = alpha
    lift_to_drag = [
        (
            row["CL"] / row["CD"]
            if row["CD"] > 0 and math.isfinite(row["CL"])
            else math.nan
        )
        for row in ordered
    ]
    return [
        write_polar_chart(
            out_dir,
            "cl_vs_alpha.png",
            alpha,
            cl,
            alpha,
            f"{title} CL vs alpha",
            "alpha (deg)",
            "CL",
            "#2563eb",
        ),
        write_polar_chart(
            out_dir,
            "cd_vs_alpha.png",
            alpha,
            cd,
            alpha,
            f"{title} CD vs alpha",
            "alpha (deg)",
            "CD",
            "#dc2626",
            scientific_y=True,
        ),
        write_polar_chart(
            out_dir,
            "cm_vs_alpha.png",
            alpha,
            cm,
            alpha,
            f"{title} CM vs alpha",
            "alpha (deg)",
            "CM",
            "#7c3aed",
        ),
        write_polar_chart(
            out_dir,
            "cl_vs_cd.png",
            cd,
            cl,
            alpha,
            f"{title} CL vs CD",
            "CD",
            "CL",
            "#0f766e",
            scientific_x=True,
        ),
        write_polar_chart(
            out_dir,
            "lift_to_drag_vs_alpha.png",
            ld_alpha,
            lift_to_drag,
            ld_alpha,
            f"{title} lift-to-drag ratio vs alpha",
            "alpha (deg)",
            "CL/CD",
            "#b45309",
        ),
    ]


def read_cp_series(
    cp_files_by_alpha: Sequence[tuple[float, Path]],
) -> list[tuple[float, list[tuple[float, float]]]]:
    cp_series = [
        (alpha, parse_cp_file(path))
        for alpha, path in cp_files_by_alpha
        if path.exists()
    ]
    return [(alpha, rows) for alpha, rows in cp_series if rows]


def write_cp_polar_outputs(
    cp_series: Sequence[tuple[float, Sequence[tuple[float, float]]]],
    geometry_points: Sequence[tuple[float, float]],
    polar_rows: Sequence[dict[str, float]],
    out_dir: Path,
    title: str,
    save_pressure_vectors: bool = True,
    requested_alphas: Sequence[float] | None = None,
) -> PlotOutputs:
    cp_series = [
        (alpha, points)
        for alpha, points in cp_series
        if row_for_alpha(polar_rows, alpha) is not None
    ]
    result_plots = [
        write_cp_distribution_plot(cp_series, out_dir, title),
        *write_polar_result_plots(
            polar_rows, out_dir, title, requested_alphas
        ),
    ]
    vectors: list[Path] = []
    if save_pressure_vectors:
        for alpha, cp_rows in cp_series:
            vector_path = write_pressure_vector_plot(
                cp_rows, geometry_points, polar_rows, out_dir, title, alpha
            )
            if vector_path is not None:
                vectors.append(vector_path)
    return {"result_plots": result_plots, "pressure_vectors": vectors}
