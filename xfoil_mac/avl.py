"""Optional AVL on macOS.

Reproducible geometry, bounded solves, forces and derivatives.
"""

from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import platform
import re
import shutil
import time
import urllib.request
from pathlib import Path

from .data import (
    atomic_json,
    atomic_text,
    fingerprint,
    positive_finite,
    sha256_file,
)
from .geometry import inspect_points, load_coordinates
from .results import write_records

AVL_URL = "https://web.mit.edu/drela/Public/web/avl/avl3.40_execs/DARWINM1/avl"
AVL_SHA256 = "b893039ccd7d322d1c997ae7e694a854ff36b149ce45d8209d8a394b68ec0033"


def install_avl(root: Path) -> Path:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError(
            "Automatic installation supports Apple Silicon macOS; "
            "use --avl-binary for another compatible build"
        )
    binary = root / "avl_runtime/bin/avl"
    if binary.is_file() and sha256_file(binary) == AVL_SHA256:
        pass
    else:
        if binary.exists():
            raise ValueError(
                "Existing AVL binary differs; preserve it and choose "
                "--avl-binary or another installation directory"
            )
        with urllib.request.urlopen(AVL_URL, timeout=60) as response:
            payload = response.read(10 * 1024 * 1024 + 1)
        if (
            len(payload) > 10 * 1024 * 1024
            or hashlib.sha256(payload).hexdigest() != AVL_SHA256
        ):
            raise ValueError(
                "Official AVL download does not match the tested build "
                "checksum; review the new release before installing"
            )
        binary.parent.mkdir(parents=True, exist_ok=True)
        temporary = binary.with_suffix(".download")
        temporary.write_bytes(payload)
        temporary.chmod(0o755)
        temporary.replace(binary)
    binary.chmod(binary.stat().st_mode | 0o100)
    atomic_json(
        root / "avl_runtime/install.json",
        {
            "url": AVL_URL,
            "sha256": AVL_SHA256,
            "version": "3.40b official ARM macOS distribution",
            "source_url": (
                "https://web.mit.edu/drela/Public/web/avl/avl3.40b.tgz"
            ),
            "license": "GPL",
            "license_url": "https://web.mit.edu/drela/Public/web/avl/",
        },
    )
    return binary


def find_avl(root: Path, explicit: Path | None = None) -> Path:
    candidate = (
        explicit.expanduser().resolve()
        if explicit
        else root / "avl_runtime/bin/avl"
    )
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    external = shutil.which("avl") if explicit is None else None
    if external:
        return Path(external)
    raise ValueError(
        "AVL is not installed. Run python -m xfoil_mac --install-avl, "
        "or set --avl-binary PATH"
    )


def _number(value, name, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    if positive:
        positive_finite(value, name)
    return value


def load_wing(path: Path) -> dict:
    config = json.loads(path.read_text())
    config = copy.deepcopy(config)
    allowed = {
        "name",
        "mach",
        "profile_drag",
        "reference",
        "surfaces",
        "cases",
    }
    if set(config) - allowed:
        raise ValueError(
            f"Unknown wing settings: {sorted(set(config)-allowed)}"
        )
    config.setdefault("name", path.stem)
    config.setdefault("mach", 0.0)
    config.setdefault("profile_drag", 0.0)
    if (
        not isinstance(config["name"], str)
        or "\n" in config["name"]
        or "\r" in config["name"]
    ):
        raise ValueError("Wing name must be a single line")
    if (
        not 0 <= _number(config["mach"], "Mach") < 1
        or _number(config["profile_drag"], "Profile drag") < 0
    ):
        raise ValueError("Require subsonic Mach and nonnegative profile drag")
    reference = config.get("reference", {})
    for key in ("area", "chord", "span"):
        _number(reference.get(key), f"Reference {key}", True)
    point = reference.get("point", [0.0, 0.0, 0.0])
    if not isinstance(point, list) or len(point) != 3:
        raise ValueError("Reference point needs x,y,z")
    for value in point:
        _number(value, "Reference point")
    reference["point"] = point
    surfaces = config.get("surfaces", [])
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError("At least one surface is required")
    vortices = 0
    for surface in surfaces:
        if set(surface) - {
            "name",
            "mirror",
            "chord_panels",
            "span_panels",
            "sections",
        }:
            raise ValueError("Unknown surface setting")
        if not isinstance(surface.get("name"), str) or any(
            c in surface["name"] for c in "\n\r"
        ):
            raise ValueError("Surface name must be a single line")
        surface.setdefault("mirror", True)
        if not isinstance(surface["mirror"], bool):
            raise ValueError("mirror must be true or false")
        for key, default in [("chord_panels", 12), ("span_panels", 24)]:
            surface.setdefault(key, default)
            if (
                not isinstance(surface[key], int)
                or not 2 <= surface[key] <= 200
            ):
                raise ValueError(f"{key} must be an integer from 2 to 200")
        vortices += (
            surface["chord_panels"]
            * surface["span_panels"]
            * (2 if surface["mirror"] else 1)
        )
        sections = surface.get("sections", [])
        if not isinstance(sections, list) or len(sections) < 2:
            raise ValueError("Each surface needs at least two sections")
        locations = []
        for section in sections:
            if set(section) - {
                "x",
                "y",
                "z",
                "chord",
                "twist",
                "naca",
                "airfoil",
            }:
                raise ValueError("Unknown section setting")
            for key in ("x", "y", "z", "chord"):
                _number(section.get(key), f"Section {key}", key == "chord")
            section.setdefault("twist", 0.0)
            _number(section["twist"], "Twist")
            locations.append((section["y"], section["z"]))
            if surface["mirror"] and section["y"] < 0:
                raise ValueError(
                    "Mirrored surfaces use the nonnegative y half only"
                )
            if ("naca" in section) == ("airfoil" in section):
                raise ValueError(
                    "Each section needs exactly one naca or airfoil"
                )
            if "naca" in section and not re.fullmatch(
                r"\d{4}", str(section["naca"])
            ):
                raise ValueError(
                    "AVL NACA sections require a four-digit string"
                )
            if "airfoil" in section:
                source = (
                    (path.parent / section["airfoil"]).expanduser().resolve()
                )
                report, points = inspect_points(load_coordinates(source))
                if not report["valid"]:
                    raise ValueError(
                        f"Invalid AVL section geometry: {source}: "
                        f'{report["issues"]}'
                    )
                section["airfoil"] = str(source)
                section["airfoil_sha256"] = sha256_file(source)
                section["coordinates"] = points
        if any(a == b for a, b in zip(locations, locations[1:])):
            raise ValueError(
                "Adjacent sections cannot have the same y,z location"
            )
    if vortices > 6000:
        raise ValueError(
            "This interface limits the model to 6000 vortices; "
            "reduce panel counts"
        )
    cases = config.get("cases", [])
    if not isinstance(cases, list) or not 1 <= len(cases) <= 100:
        raise ValueError("Supply between 1 and 100 wing cases")
    for case in cases:
        if set(case) - {"alpha", "cl", "beta"} or ("alpha" in case) == (
            "cl" in case
        ):
            raise ValueError(
                "Each wing case needs exactly one alpha or cl, "
                "and optionally beta"
            )
        case.setdefault("beta", 0.0)
        for k, v in case.items():
            _number(v, k)
    return config


def geometry_text(config: dict, destination: Path) -> str:
    ref = config["reference"]
    lines = [
        config["name"],
        str(config["mach"]),
        "0 0 0",
        f"{ref['area']} {ref['chord']} {ref['span']}",
        " ".join(map(str, ref["point"])),
        str(config["profile_drag"]),
    ]
    for i, surface in enumerate(config["surfaces"]):
        lines += [
            "SURFACE",
            surface["name"],
            f"{surface['chord_panels']} 1 {surface['span_panels']} 1",
        ]
        if surface["mirror"]:
            lines += ["YDUPLICATE", "0"]
        for j, section in enumerate(surface["sections"]):
            lines += [
                "SECTION",
                " ".join(
                    str(section[k]) for k in ("x", "y", "z", "chord", "twist")
                ),
            ]
            if "naca" in section:
                lines += ["NACA", str(section["naca"])]
            else:
                name = f"section_{i:02d}_{j:02d}.dat"
                atomic_text(
                    destination / name,
                    "Normalized section copy\n"
                    + "".join(
                        f"{x:.10g} {y:.10g}\n"
                        for x, y in section["coordinates"]
                    ),
                )
                lines += ["AFILE", name]
    return "\n".join(lines) + "\n"


def parse_values(text: str) -> dict:
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
    values = {}
    for name, value in re.findall(
        r"([A-Za-z][A-Za-z0-9_']*)\s*=\s*(" + number + r")", text
    ):
        # The final spiral-stability expression ends in "Cnb =";
        # do not overwrite Cnb.
        parsed = float(value.replace("D", "E").replace("d", "e"))
        if math.isfinite(parsed):
            values.setdefault(name, parsed)
    return values


def parse_strips(text: str, config: dict | None = None) -> list[dict]:
    """Read finite strip loads; reject damaged or incomplete surface tables.

    Chord and area use model length units and their square, respectively.
    Supplying the model also checks that no complete surface was lost from
    the output. Without it, a standalone table can only be checked against
    the dimensions that AVL printed in that table.
    """
    rows = []
    blocks = []
    block = None
    headers = None
    required_columns = {
        "j",
        "Xle",
        "Yle",
        "Zle",
        "Chord",
        "Area",
        "c_cl",
        "cl",
    }
    for line_number, line in enumerate(text.splitlines(), 1):
        if "Surface #" in line:
            label = line.split("Surface #", 1)[1].strip()
            match = re.match(r"(\d+)\s+", label)
            if not match:
                raise ValueError(
                    f"Invalid AVL surface header at line {line_number}"
                )
            block = {
                "surface": label,
                "index": int(match.group(1)),
                "rows": [],
            }
            blocks.append(block)
            headers = None
            continue
        if block is None:
            continue
        if "# Chordwise" in line:
            match = re.search(
                r"# Chordwise\s*=\s*(\d+)\s+# Spanwise\s*=\s*(\d+)"
                r"\s+First strip\s*=\s*(\d+)",
                line,
            )
            if not match or "span_panels" in block:
                raise ValueError(
                    f"Invalid AVL panel counts at line {line_number}"
                )
            block.update(
                zip(
                    ("chord_panels", "span_panels", "first_strip"),
                    map(int, match.groups()),
                )
            )
            continue
        if "Surface area Ssurf" in line:
            area = parse_values(line).get("Ssurf")
            if area is None or area <= 0 or "area" in block:
                raise ValueError(
                    f"Invalid AVL surface area at line {line_number}"
                )
            block["area"] = area
            continue
        fields = line.split()
        if fields and fields[0] == "j":
            if (
                headers
                or not required_columns.issubset(fields)
                or len(set(fields)) != len(fields)
            ):
                raise ValueError(
                    f"Invalid AVL strip columns at line {line_number}"
                )
            headers = fields
            continue
        if headers and fields and set(line.strip()) != {"-"}:
            if len(fields) != len(headers):
                raise ValueError(
                    f"Incomplete AVL strip row at line {line_number}"
                )
            try:
                values = [
                    float(v.replace("D", "E").replace("d", "e"))
                    for v in fields
                ]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid AVL strip number at line {line_number}"
                ) from exc
            if not all(math.isfinite(v) for v in values):
                raise ValueError(
                    f"Nonfinite AVL strip row at line {line_number}"
                )
            row = dict(zip(headers, values), surface=block["surface"])
            if (
                row["j"] != len(rows) + 1
                or row["Chord"] <= 0
                or row["Area"] <= 0
            ):
                raise ValueError(
                    "Invalid AVL strip index, chord or area "
                    f"at line {line_number}"
                )
            rows.append(row)
            block["rows"].append(row)
    if not blocks or not rows:
        raise ValueError("AVL strip output contains no usable surface table")
    if [b["index"] for b in blocks] != list(range(1, len(blocks) + 1)):
        raise ValueError("AVL strip output has missing or repeated surfaces")
    expected_surfaces = []
    if config is not None:
        for surface in config["surfaces"]:
            expected_surfaces.extend(
                [surface] * (2 if surface["mirror"] else 1)
            )
        if len(blocks) != len(expected_surfaces):
            raise ValueError(
                "AVL strip output does not contain every model surface"
            )
    for index, block in enumerate(blocks):
        count = len(block["rows"])
        if not count:
            raise ValueError(f"AVL surface {block['index']} has no strip rows")
        if "span_panels" in block and (
            count != block["span_panels"]
            or block["rows"][0]["j"] != block["first_strip"]
        ):
            raise ValueError(
                f"AVL surface {block['index']} has incomplete strip coverage"
            )
        # AVL prints Area to four decimals and Ssurf to six. Summing these
        # rounded strips need not equal Ssurf bit-for-bit. Sref can describe
        # the main wing alone, so it is not a completeness check for
        # all surfaces.
        area_tolerance = count * 0.00005 + 0.0000005
        if "area" in block and not math.isclose(
            sum(row["Area"] for row in block["rows"]),
            block["area"],
            rel_tol=1e-6,
            abs_tol=area_tolerance,
        ):
            raise ValueError(
                f"AVL surface {block['index']} strip areas do not sum to Ssurf"
            )
        if config is not None:
            surface = expected_surfaces[index]
            if any(
                key not in block
                for key in ("chord_panels", "span_panels", "area")
            ):
                raise ValueError(
                    f"AVL surface {block['index']} is missing its dimensions"
                )
            if any(
                block[key] != surface[key]
                for key in ("chord_panels", "span_panels")
            ):
                raise ValueError(
                    f"AVL surface {block['index']} panel counts "
                    "differ from the model"
                )
            sections = surface["sections"]
            span_length = sum(
                math.hypot(b["y"] - a["y"], b["z"] - a["z"])
                for a, b in zip(sections, sections[1:])
            )
            chords = [section["chord"] for section in sections]
            # With coarse panels across a section break, AVL's midpoint
            # quadrature can differ from the exact trapezoidal model area.
            lower = min(chords) * span_length
            upper = max(chords) * span_length
            if (
                not lower - area_tolerance
                <= block["area"]
                <= upper + area_tolerance
            ):
                raise ValueError(
                    f"AVL surface {block['index']} area is inconsistent "
                    "with the model"
                )
    return rows


def wing_commands(config: dict, case: dict) -> list[str]:
    # AVL 3.40 initializes OPER Mach/CD to zero even when geometry
    # has defaults.
    return [
        "LOAD wing.avl",
        "OPER",
        "M",
        f"MN {config['mach']:.10g}",
        f"CD {config['profile_drag']:.10g}",
        "",
        (
            f"A A {case['alpha']:.10g}"
            if "alpha" in case
            else f"A C {case['cl']:.10g}"
        ),
        f"B B {case['beta']:.10g}",
        "X",
        "FT",
        "totals.txt",
        "FS",
        "strips.txt",
        "ST",
        "stability.txt",
        "",
        "QUIT",
    ]


def run_wing(
    source: Path,
    destination: Path,
    binary: Path,
    timeout=120.0,
    resume=False,
    force=False,
) -> Path:
    from .execution import execute

    positive_finite(timeout, "AVL timeout")
    if resume and force:
        raise ValueError("Cannot combine force and resume")
    config = load_wing(source.expanduser().resolve())
    destination = destination.expanduser().resolve()
    metadata = {
        "config": config,
        "binary_sha256": sha256_file(binary),
        "implementation_sha256": sha256_file(Path(__file__)),
        "execution_implementation_sha256": sha256_file(
            Path(__file__).with_name("execution.py")
        ),
        "geometry_implementation_sha256": sha256_file(
            Path(__file__).with_name("geometry.py")
        ),
        "timeout": timeout,
    }
    identity = fingerprint(metadata)
    manifest_path = destination / "wing_run.json"
    previous = {}
    if destination.exists() and any(destination.iterdir()):
        if not manifest_path.is_file():
            raise ValueError("Use an empty directory for a new AVL study")
        previous = json.loads(manifest_path.read_text())
        if resume:
            if previous["fingerprint"] != identity:
                raise ValueError(
                    "Cannot resume AVL: configuration, binary "
                    "or implementation changed"
                )
        elif not force:
            raise ValueError("Wing study exists; use --resume or --force")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        **metadata,
        "fingerprint": identity,
        "solver": "AVL",
        "profile_drag_included": config["profile_drag"] > 0,
        "model_note": (
            "Linear vortex-lattice, small-angle attached-flow model; "
            "no viscous separation or stall prediction. Derivatives alone "
            "do not establish whole-aircraft stability."
        ),
        "status": "running",
        "cases": [],
    }
    atomic_json(manifest_path, manifest)
    deadline = time.monotonic() + timeout
    records = []
    loads = []
    for index, case in enumerate(config["cases"], 1):
        folder = destination / f"case_{index:03d}"
        folder.mkdir(exist_ok=True)
        required = [
            folder / name
            for name in ("totals.txt", "strips.txt", "stability.txt")
        ]
        saved_cases = previous.get("cases", [])
        old_case = saved_cases[index - 1] if index <= len(saved_cases) else {}
        cached = (
            resume
            and old_case.get("status") == "ok"
            and all(
                p.is_file()
                and old_case.get("raw_sha256", {}).get(p.name)
                == sha256_file(p)
                for p in required
            )
        )
        status = "finished"
        if not cached:
            for file in required:
                file.unlink(missing_ok=True)
            atomic_text(folder / "wing.avl", geometry_text(config, folder))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "timeout"
            else:
                status, _ = execute(
                    binary,
                    os.environ.copy(),
                    folder,
                    wing_commands(config, case),
                    remaining,
                    log_filename="avl.log",
                )
        totals = (
            parse_values(required[0].read_text())
            if required[0].is_file()
            else {}
        )
        derivatives = (
            parse_values(required[2].read_text())
            if required[2].is_file()
            else {}
        )
        strip_error = None
        try:
            strips = parse_strips(required[1].read_text(), config)
        except (OSError, ValueError) as exc:
            strips = []
            strip_error = str(exc)
        valid = (
            all(
                k in totals and math.isfinite(totals[k])
                for k in ("CLtot", "CDind", "Cmtot", "Alpha", "Mach", "Beta")
            )
            and bool(strips)
            and "CLa" in derivatives
        )
        if valid:
            valid = (
                abs(totals["Mach"] - config["mach"]) <= 0.00051
                and abs(totals["Beta"] - case["beta"]) < 1e-4
            )
            valid = valid and (
                abs(totals["Alpha"] - case["alpha"]) < 1e-4
                if "alpha" in case
                else abs(totals["CLtot"] - case["cl"]) < 1e-4
            )
        good = status == "finished" and valid
        record = {
            "case": index,
            "status": (
                "ok"
                if good
                else status if status != "finished" else "invalid_outputs"
            ),
            "raw_sha256": {
                p.name: sha256_file(p) for p in required if p.is_file()
            },
            **{f"target_{k}": v for k, v in case.items()},
            **{
                k: totals.get(k)
                for k in (
                    "Alpha",
                    "Beta",
                    "Mach",
                    "CLtot",
                    "CDtot",
                    "CDind",
                    "CDvis",
                    "Cmtot",
                    "e",
                )
            },
            **{
                k: derivatives.get(k)
                for k in (
                    "CLa",
                    "Cma",
                    "CLb",
                    "CYb",
                    "Clb",
                    "Cnb",
                    "Clp",
                    "Cmq",
                    "Cnr",
                    "Xnp",
                )
            },
        }
        if strip_error:
            record["strip_error"] = strip_error
        if good:
            for row in strips:
                loads.append(dict(row, case=index))
        records.append(record)
        manifest["cases"] = records
        manifest["status"] = "running"
        atomic_json(manifest_path, manifest)
        if status == "interrupted":
            manifest["status"] = "interrupted"
            atomic_json(manifest_path, manifest)
            raise KeyboardInterrupt
    manifest["status"] = (
        "ok"
        if all(r["status"] == "ok" for r in records)
        else "needs_attention"
    )
    atomic_json(manifest_path, manifest)
    write_records(
        destination / "wing_summary.csv",
        [{k: v for k, v in r.items() if k != "raw_sha256"} for r in records],
    )
    write_records(destination / "spanwise_loads.csv", loads)
    _plot_wing(config, records, loads, destination)
    from .ui.wing import write_wing_interactive
    from .analysis.report import report_page

    write_wing_interactive(config, destination / "wing_3d.html")
    report_page(
        destination / "report.html",
        config["name"],
        manifest["model_note"],
        records,
        [
            ("旋转三维机翼 / 查看截面", "wing_3d.html"),
            ("展向载荷", "spanwise_loads.csv"),
            ("计算输入和状态", "wing_run.json"),
        ],
        [
            ("三维外形与翼型厚度", "wing_geometry.png"),
            ("俯视投影", "wing_planform.png"),
            ("载荷与性能", "wing_performance.png"),
        ],
    )
    return manifest_path


def _plot_wing(config, records, loads, destination):
    import matplotlib.pyplot as plt

    from .wing_plotting import write_wing_geometry

    write_wing_geometry(config, destination)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for case in records:
        if case["status"] != "ok":
            continue
        group = [r for r in loads if r["case"] == case["case"]]
        color = plt.get_cmap("tab10")((case["case"] - 1) % 10)
        label = f"case {case['case']}: alpha={case['Alpha']:g}"
        for surface_index, surface in enumerate(
            dict.fromkeys(r["surface"] for r in group)
        ):
            rows = sorted(
                [r for r in group if r["surface"] == surface],
                key=lambda r: r["Yle"],
            )
            axes[0].plot(
                [r["Yle"] for r in rows],
                [r["c_cl"] for r in rows],
                color=color,
                label=label if surface_index == 0 else None,
            )
        axes[1].scatter(case["CDind"], case["CLtot"], color=color, label=label)
    axes[0].set(
        xlabel="Spanwise y",
        ylabel="Chord × sectional cl",
        title="Spanwise loading",
    )
    axes[1].set(
        xlabel="Induced CD only", ylabel="CL", title="Wing induced drag"
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        if ax.has_data():
            ax.legend(fontsize=7)
    fig.savefig(destination / "wing_performance.png", dpi=160)
    plt.close(fig)
