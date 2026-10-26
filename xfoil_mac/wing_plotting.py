"""Physical section-shape illustration plus a separate AVL lifting-surface
outline.
"""

from pathlib import Path


def write_lifting_surface_outline(config: dict, destination: Path) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from .plotting import (
        DARK_BG,
        DARK_GRID,
        DARK_MUTED,
        DARK_PANEL,
        DARK_SERIES_PALETTE,
        DARK_TEXT,
        GEOMETRY_COLOR,
        HUD_FONT_FAMILY,
        apply_dark_engineering_style,
        style_dark_legend,
    )

    fig = plt.figure(figsize=(12, 8), layout="constrained", facecolor=DARK_BG)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])
    top = fig.add_subplot(grid[0, :])
    space = fig.add_subplot(grid[1, 0], projection="3d")
    details = fig.add_subplot(grid[1, 1])
    apply_dark_engineering_style(fig, [top])
    space.set_facecolor(DARK_PANEL)
    details.set_facecolor(DARK_PANEL)
    details.axis("off")
    vertices, description = [], []
    for i, surface in enumerate(config["surfaces"]):
        color = DARK_SERIES_PALETTE[i % len(DARK_SERIES_PALETTE)]
        sections = surface["sections"]
        description.append(f"{surface['name']} | mirror: {surface['mirror']}")
        for j, s in enumerate(sections, 1):
            foil = (
                f"NACA {s['naca']}" if "naca" in s else Path(s["airfoil"]).name
            )
            description.append(
                f"  {j}: y={s['y']:g}, z={s['z']:g}, chord={s['chord']:g}\n"
                f"     {foil}; incidence={s['twist']:+g} deg"
            )
        for sign in [1, -1] if surface["mirror"] else [1]:
            le = np.array([[s["x"], sign * s["y"], s["z"]] for s in sections])
            te = le + np.array([[s["chord"], 0, 0] for s in sections])
            vertices.extend([*le, *te])
            quads = [
                [le[j], le[j + 1], te[j + 1], te[j]]
                for j in range(len(sections) - 1)
            ]
            space.add_collection3d(
                Poly3DCollection(
                    quads, facecolors=color, edgecolors=color, alpha=0.35
                )
            )
            for q in quads:
                q = np.array(q)
                top.fill(q[:, 1], q[:, 0], color=color, alpha=0.22)
            for edge in (le, te):
                top.plot(edge[:, 1], edge[:, 0], color=color, linewidth=1.6)
            for a, b in zip(le, te):
                top.plot([a[1], b[1]], [a[0], b[0]], color=color, alpha=0.75)

    points = np.array(vertices)
    lower, upper = points.min(axis=0), points.max(axis=0)
    extent = upper - lower
    # Each axis has identical units on screen; only the empty z range gets
    # padding.
    padded = np.maximum(extent * 1.1, max(float(extent.max()), 1e-6) * 0.06)
    center = (lower + upper) / 2
    space.set_xlim(center[0] - padded[0] / 2, center[0] + padded[0] / 2)
    space.set_ylim(center[1] - padded[1] / 2, center[1] + padded[1] / 2)
    space.set_zlim(center[2] - padded[2] / 2, center[2] + padded[2] / 2)
    space.set_box_aspect(padded)
    space.set_xticks([lower[0], upper[0]] if extent[0] else [lower[0]])
    space.set_yticks(
        [lower[1], center[1], upper[1]] if extent[1] else [lower[1]]
    )
    space.set_zticks([lower[2], upper[2]] if extent[2] else [lower[2]])
    space.view_init(elev=24, azim=-55)
    space.set(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        title="Lifting-surface outline | equal scale",
    )
    space.title.set_color(GEOMETRY_COLOR)
    space.xaxis.label.set_color(DARK_MUTED)
    space.yaxis.label.set_color(DARK_MUTED)
    space.zaxis.label.set_color(DARK_MUTED)
    space.tick_params(colors=DARK_MUTED)
    for axis in (space.xaxis, space.yaxis, space.zaxis):
        axis.pane.set_facecolor((17 / 255, 24 / 255, 39 / 255, 0.85))
        axis.pane.set_edgecolor(DARK_GRID)
    top.set_aspect("equal", adjustable="box")
    top.invert_yaxis()
    top.set_xlabel("Spanwise y (model length units)", color=DARK_MUTED)
    top.set_ylabel("Downstream x", color=DARK_MUTED)
    top.set_title(
        f"Top view | equal scale | y extent={extent[1]:g}, x "
        f"extent={extent[0]:g}",
        color=GEOMETRY_COLOR,
        fontweight="bold",
    )
    top.grid(True, color=DARK_GRID, alpha=0.85)
    reference = config["reference"]
    details.set_title(
        "Section inputs used by AVL",
        loc="left",
        color=GEOMETRY_COLOR,
        fontweight="bold",
    )
    text = (
        f"Sref={reference['area']:g}  Cref={reference['chord']:g}  "
        f"Bref={reference['span']:g}\n\n"
    ) + "\n".join(description)
    lines = text.splitlines()
    if len(lines) > 14:
        text = (
            "\n".join(lines[:13]) + "\n... full section list in wing_run.json"
        )
    details.text(
        0,
        0.98,
        text,
        va="top",
        fontsize=9.5,
        family=HUD_FONT_FAMILY,
        color=DARK_TEXT,
    )
    details.text(
        0,
        -0.03,
        "Incidence changes AVL's flow boundary condition.\n"
        "It does not rotate the section coordinates.\n"
        "This is a lifting-surface preview, without airfoil thickness.",
        va="bottom",
        fontsize=8.5,
        family=HUD_FONT_FAMILY,
        color=DARK_MUTED,
        linespacing=1.5,
    )
    fig.suptitle(config["name"], fontsize=13, color=GEOMETRY_COLOR, fontweight="bold")
    apply_dark_engineering_style(fig, [top])
    fig.savefig(
        destination / "wing_planform.png", dpi=160, facecolor=DARK_BG
    )
    plt.close(fig)


def write_wing_geometry(config: dict, destination: Path) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from .wing_geometry import surface_lofts, loft_triangles, section_profile
    from .plotting import (
        DARK_BG,
        DARK_GRID,
        DARK_MUTED,
        DARK_PANEL,
        DARK_SERIES_PALETTE,
        DARK_TEXT,
        GEOMETRY_COLOR,
        HUD_FONT_FAMILY,
        apply_dark_engineering_style,
        style_dark_legend,
    )

    write_lifting_surface_outline(config, destination)
    fig = plt.figure(figsize=(13, 8), layout="constrained", facecolor=DARK_BG)
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.25, 1], height_ratios=[1.15, 1]
    )
    whole = fig.add_subplot(grid[:, 0], projection="3d")
    closeup = fig.add_subplot(grid[0, 1], projection="3d")
    profiles = fig.add_subplot(grid[1, 1])
    apply_dark_engineering_style(fig, [profiles])
    whole.set_facecolor(DARK_PANEL)
    closeup.set_facecolor(DARK_PANEL)
    all_points = []
    first_lofts = None

    def draw_skin(ax, loft, color, reflected=False):
        ax.add_collection3d(
            Poly3DCollection(
                loft_triangles(loft, reflected),
                facecolors=color,
                edgecolors=color,
                linewidths=0,
                antialiased=False,
                shade=True,
                alpha=0.85,
            )
        )
        for ring in (loft[0], loft[-1]):
            closed = np.vstack((ring, ring[0]))
            ax.add_collection3d(
                Poly3DCollection(
                    [ring],
                    facecolors=color,
                    edgecolors=DARK_TEXT,
                    linewidths=0.7,
                )
            )
            ax.plot(*closed.T, color=DARK_TEXT, linewidth=0.9)

    def fit_equal(ax, points, elev, azim, zoom=1):
        low, high = points.min(axis=0), points.max(axis=0)
        size = np.maximum(
            (high - low) * 1.08,
            max(float(np.ptp(points, axis=0).max()), 1e-6) * 0.035,
        )
        middle = (high + low) / 2
        for index, setter in enumerate(
            (ax.set_xlim, ax.set_ylim, ax.set_zlim)
        ):
            setter(
                middle[index] - size[index] / 2,
                middle[index] + size[index] / 2,
            )
        ax.set_box_aspect(size, zoom=zoom)
        ax.view_init(elev=elev, azim=azim)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
            axis.set_major_formatter(FormatStrFormatter("%g"))
            axis.pane.set_facecolor((17 / 255, 24 / 255, 39 / 255, 0.85))
            axis.pane.set_edgecolor(DARK_GRID)
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        ax.xaxis.label.set_color(DARK_MUTED)
        ax.yaxis.label.set_color(DARK_MUTED)
        ax.zaxis.label.set_color(DARK_MUTED)
        ax.xaxis.label.set_fontfamily(HUD_FONT_FAMILY)
        ax.yaxis.label.set_fontfamily(HUD_FONT_FAMILY)
        ax.zaxis.label.set_fontfamily(HUD_FONT_FAMILY)
        ax.tick_params(labelsize=8, pad=1, colors=DARK_MUTED)
        ax.grid(False)

    for i, surface in enumerate(config["surfaces"]):
        color = DARK_SERIES_PALETTE[i % len(DARK_SERIES_PALETTE)]
        lofts = surface_lofts(surface)
        if first_lofts is None:
            first_lofts = lofts
        for side, loft in enumerate(lofts):
            draw_skin(whole, loft, color, reflected=side == 1)
            all_points.append(loft.reshape(-1, 3))
    fit_equal(whole, np.concatenate(all_points), elev=19, azim=-58, zoom=1.15)
    whole.set_axis_off()
    whole.set_title(
        "3D section loft | true proportions and thickness",
        fontsize=11,
        color=GEOMETRY_COLOR,
        fontweight="bold",
    )

    # Show the near tip at a larger scale so a slender wing's section remains
    # visible.
    tip = first_lofts[-1][-5:]
    draw_skin(
        closeup,
        tip,
        DARK_SERIES_PALETTE[0],
        reflected=len(first_lofts) == 2,
    )
    fit_equal(closeup, tip.reshape(-1, 3), elev=18, azim=-72)
    closeup.set_title(
        "Wing-tip close-up | enlarged view",
        fontsize=9.5,
        color=GEOMETRY_COLOR,
        fontweight="bold",
    )
    for index, section in enumerate(config["surfaces"][0]["sections"][:6]):
        xy = section_profile(section) * section["chord"]
        angle = np.radians(section["twist"])
        rotated = xy @ np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        label = (
            f"Section {index+1}: "
            + (
                f"NACA {section['naca']}"
                if "naca" in section
                else Path(section["airfoil"]).name
            )
            + f", c={section['chord']:g}, i={section['twist']:+g} deg"
        )
        profiles.plot(
            *rotated.T,
            label=label,
            linewidth=1.7,
            color=DARK_SERIES_PALETTE[index % len(DARK_SERIES_PALETTE)],
        )
    profiles.set_aspect("equal", adjustable="box")
    profiles.set_xlabel("X: CHORD DISTANCE (MODEL UNITS)", color=DARK_MUTED)
    profiles.set_ylabel("Z/C THICKNESS", color=DARK_MUTED)
    profiles.set_title(
        "Section contours | leading edges aligned",
        color=GEOMETRY_COLOR,
        fontweight="bold",
    )
    profiles.margins(y=0.25)
    profiles.grid(True, color=DARK_GRID, alpha=0.85)
    leg = profiles.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.65), fontsize=8
    )
    style_dark_legend(leg)
    fig.suptitle(
        config["name"] + " | airfoil-shape preview",
        fontsize=13,
        color=GEOMETRY_COLOR,
        fontweight="bold",
    )
    fig.supxlabel(
        "Appearance illustration: sections rotate about their leading "
        "edges. AVL still solves its thin lifting-surface model.",
        fontsize=8.5,
        color=DARK_MUTED,
    )
    apply_dark_engineering_style(fig, [profiles])
    fig.savefig(
        destination / "wing_geometry.png", dpi=170, facecolor=DARK_BG
    )
    plt.close(fig)
