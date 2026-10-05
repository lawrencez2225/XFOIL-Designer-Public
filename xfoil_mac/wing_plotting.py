"""Physical section-shape illustration plus a separate AVL lifting-surface
outline.
"""

from pathlib import Path


def write_lifting_surface_outline(config: dict, destination: Path) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])
    top = fig.add_subplot(grid[0, :])
    space = fig.add_subplot(grid[1, 0], projection="3d")
    details = fig.add_subplot(grid[1, 1])
    details.axis("off")
    vertices, description = [], []
    for i, surface in enumerate(config["surfaces"]):
        color = plt.get_cmap("tab10")(i % 10)
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
                    quads, facecolors=color, edgecolors=color, alpha=0.3
                )
            )
            for q in quads:
                q = np.array(q)
                top.fill(q[:, 1], q[:, 0], color=color, alpha=0.18)
            for edge in (le, te):
                top.plot(edge[:, 1], edge[:, 0], color=color)
            for a, b in zip(le, te):
                top.plot([a[1], b[1]], [a[0], b[0]], color=color, alpha=0.6)

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
    top.set_aspect("equal", adjustable="box")
    top.invert_yaxis()
    top.set(
        xlabel="Spanwise y (model length units)",
        ylabel="Downstream x",
        title=(
            f"Top view | equal scale | y extent={extent[1]:g}, x "
            f"extent={extent[0]:g}"
        ),
    )
    top.grid(alpha=0.2)
    reference = config["reference"]
    details.set_title("Section inputs used by AVL", loc="left")
    text = (
        f"Sref={reference['area']:g}  Cref={reference['chord']:g}  "
        f"Bref={reference['span']:g}\n\n"
    ) + "\n".join(description)
    lines = text.splitlines()
    if len(lines) > 14:
        text = (
            "\n".join(lines[:13]) + "\n... full section list in wing_run.json"
        )
    details.text(0, 0.98, text, va="top", fontsize=10, family="monospace")
    details.text(
        0,
        -0.03,
        "Incidence changes AVL's flow boundary condition.\n"
        "It does not rotate the section coordinates.\n"
        "This is a lifting-surface preview, without airfoil thickness.",
        va="bottom",
        fontsize=9,
        color="#52616b",
        linespacing=1.5,
    )
    fig.suptitle(config["name"], fontsize=15)
    fig.savefig(destination / "wing_planform.png", dpi=160)
    plt.close(fig)


def write_wing_geometry(config: dict, destination: Path) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from .wing_geometry import surface_lofts, loft_triangles, section_profile

    write_lifting_surface_outline(config, destination)
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.25, 1], height_ratios=[1.15, 1]
    )
    whole = fig.add_subplot(grid[:, 0], projection="3d")
    closeup = fig.add_subplot(grid[0, 1], projection="3d")
    profiles = fig.add_subplot(grid[1, 1])
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
            )
        )
        for ring in (loft[0], loft[-1]):
            closed = np.vstack((ring, ring[0]))
            ax.add_collection3d(
                Poly3DCollection(
                    [ring],
                    facecolors=color,
                    edgecolors="#16354b",
                    linewidths=0.7,
                )
            )
            ax.plot(*closed.T, color="#16354b", linewidth=0.8)

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
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        ax.tick_params(labelsize=8, pad=1)
        ax.grid(False)

    for i, surface in enumerate(config["surfaces"]):
        color = plt.get_cmap("tab10")(i % 10)
        lofts = surface_lofts(surface)
        if first_lofts is None:
            first_lofts = lofts
        for side, loft in enumerate(lofts):
            draw_skin(whole, loft, color, reflected=side == 1)
            all_points.append(loft.reshape(-1, 3))
    fit_equal(whole, np.concatenate(all_points), elev=19, azim=-58, zoom=1.15)
    whole.set_axis_off()
    whole.set_title(
        "3D section loft | true proportions and thickness", fontsize=13
    )

    # Show the near tip at a larger scale so a slender wing's section remains
    # visible.
    tip = first_lofts[-1][-5:]
    draw_skin(
        closeup, tip, plt.get_cmap("tab10")(0), reflected=len(first_lofts) == 2
    )
    fit_equal(closeup, tip.reshape(-1, 3), elev=18, azim=-72)
    closeup.set_title(
        "Wing-tip close-up | same thickness, enlarged view", fontsize=11
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
        profiles.plot(*rotated.T, label=label, linewidth=1.5)
    profiles.set_aspect("equal", adjustable="box")
    profiles.set(
        xlabel="Distance from leading edge (model units)",
        ylabel="Local thickness direction",
        title="Section contours | leading edges aligned",
        ymargin=0.25,
    )
    profiles.grid(alpha=0.2)
    profiles.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.65), fontsize=8
    )
    fig.suptitle(config["name"] + " | airfoil-shape preview", fontsize=16)
    fig.supxlabel(
        "Appearance illustration: sections rotate about their leading "
        "edges. AVL still solves its thin lifting-surface model.",
        fontsize=9,
        color="#52616b",
    )
    fig.savefig(destination / "wing_geometry.png", dpi=170)
    plt.close(fig)
