"""Read XFOIL DUMP surface/wake fields and retain their distinct identities."""

import math
from pathlib import Path

from ..data import atomic_json, cp_name, read_pairs
from ..results import write_records


def read_boundary_layer(path: Path, geometry):
    """Read a complete DUMP in PSAV panel order or raise a descriptive error.

    ``s``, ``x``, ``y``, ``Dstar`` and ``Theta`` use the input coordinate
    length units (normally chord-normalized). ``Ue_Vinf``, ``Cf`` and ``H``
    are dimensionless. XFOIL's DUMP column H is the kinematic shape factor
    HK; at nonzero Mach it need not equal Dstar / Theta.
    """
    if len(geometry) < 3 or not all(
        math.isfinite(value) for point in geometry for value in point
    ):
        raise ValueError(
            "Cannot validate boundary layer without finite panel geometry: "
            f"{path}"
        )
    leading = min(range(len(geometry)), key=lambda i: geometry[i][0])
    if leading in (0, len(geometry) - 1):
        raise ValueError(
            "Panel geometry must run from upper TE through LE to lower TE: "
            f"{path}"
        )
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.replace(",", " ").split()
        if not fields or fields[0].startswith("#"):
            continue
        if not records and [field.lower() for field in fields[:3]] == [
            "s",
            "x",
            "y",
        ]:
            continue
        try:
            values = [
                float(v.replace("D", "E").replace("d", "e")) for v in fields
            ]
        except ValueError as error:
            raise ValueError(
                f"Invalid boundary-layer data at {path}:{line_number}"
            ) from error
        if len(values) < 8 or not all(math.isfinite(v) for v in values):
            raise ValueError(
                "Expected eight finite boundary-layer fields at "
                f"{path}:{line_number}"
            )
        i = len(records)
        if records and values[0] < records[-1]["s"]:
            raise ValueError(
                "Boundary-layer arc length is out of order at "
                f"{path}:{line_number}"
            )
        if i < len(geometry):
            # DUMP prints x/y to five decimals; PSAV keeps seven significant
            # digits. Check both coordinates before assigning surface identity.
            if not all(
                math.isclose(value, expected, abs_tol=5.1e-6, rel_tol=5.1e-7)
                for value, expected in zip(values[1:3], geometry[i])
            ):
                raise ValueError(
                    f"Boundary-layer panel {i + 1} does not match geometry at "
                    f"{path}:{line_number}"
                )
        side = (
            "wake"
            if i >= len(geometry)
            else "upper" if i <= leading else "lower"
        )
        records.append(
            dict(
                zip(
                    ("s", "x", "y", "Ue_Vinf", "Dstar", "Theta", "Cf", "H"),
                    values,
                ),
                surface=side,
            )
        )
    if len(records) < len(geometry):
        raise ValueError(
            f"Incomplete boundary-layer surface in {path}: "
            f"{len(records)} rows for {len(geometry)} panels"
        )
    return records


def boundary_layer_error(path: Path, geometry) -> str | None:
    """Return the precise output error for checkpoint and retry accounting."""
    try:
        read_boundary_layer(path, geometry)
    except (OSError, ValueError) as error:
        return str(error)
    return None


def write_boundary_reports(root: Path, polar_rows):
    import matplotlib.pyplot as plt

    geometry = read_pairs(root / "geometry.dat")
    folder = root / "boundary_layer"
    if not folder.is_dir():
        return []
    index = []
    errors = []
    for point in polar_rows:
        alpha = point["alpha"]
        stem = cp_name(alpha).replace("cp_", "bl_", 1).removesuffix(".txt")
        try:
            records = read_boundary_layer(folder / f"{stem}.txt", geometry)
        except (OSError, ValueError) as error:
            errors.append(
                {
                    "alpha": alpha,
                    "status": "invalid_output",
                    "error": str(error),
                }
            )
            # A refreshed report must not leave an older successful plot for
            # data now known to be incomplete or corrupt.
            for suffix in (".csv", ".png"):
                (folder / f"{stem}{suffix}").unlink(missing_ok=True)
            continue
        write_records(folder / f"{stem}.csv", records)
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
        for ax, key in zip(axes.flat, ("Cf", "Dstar", "Theta", "H")):
            for side, color in (("upper", "#007f96"), ("lower", "#c66b24")):
                surface = [r for r in records if r["surface"] == side]
                ax.plot(
                    [r["x"] for r in surface],
                    [r[key] for r in surface],
                    label=side,
                    color=color,
                )
                transition = point.get(
                    "Top_Xtr" if side == "upper" else "Bot_Xtr"
                )
                if transition is not None:
                    ax.axvline(
                        transition, linestyle=":", color=color, alpha=0.6
                    )
            units = (
                "input coordinate length units"
                if key in ("Dstar", "Theta")
                else "dimensionless"
            )
            ax.set(
                xlabel="x (input coordinate length units)",
                ylabel=f"{key} ({units})",
            )
            ax.grid(alpha=0.2)
            if key == "Cf":
                ax.axhline(0, color="gray", linewidth=0.6)
                ax.legend()
        fig.suptitle(
            f"Boundary layer | alpha={alpha:g} deg | dotted lines: transition"
        )
        fig.savefig(folder / f"{stem}.png", dpi=140)
        plt.close(fig)
        index.append(
            {
                "alpha": alpha,
                "file": f"{stem}.csv",
                "plot": f"{stem}.png",
                "negative_cf_surface_points": sum(
                    r["Cf"] < 0 for r in records if r["surface"] != "wake"
                ),
                "top_transition": point.get("Top_Xtr"),
                "bottom_transition": point.get("Bot_Xtr"),
            }
        )
    atomic_json(
        folder / "index.json",
        {
            "points": index,
            "errors": errors,
            "note": (
                "Wake rows are retained in CSV and excluded from surface "
                "plots. Dstar and Theta use input coordinate length units; "
                "Ue/Vinf, Cf and H are "
                "dimensionless. H is XFOIL's kinematic shape factor HK."
            ),
        },
    )
    return index
