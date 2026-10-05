"""Wing construction, editing, and offline rotatable appearance preview."""

import copy
import json
import math
from pathlib import Path

from ..data import atomic_json, atomic_text, timestamp_label
from ..avl import load_wing
from ..wing_geometry import surface_lofts, loft_triangles, section_ring


def simple_wing(
    *,
    name="My wing",
    span=8.0,
    root_chord=1.0,
    tip_chord=1.0,
    sweep=0.0,
    dihedral=0.0,
    root_twist=0.0,
    tip_twist=-1.0,
    naca="0012",
    tip_naca=None,
    airfoil=None,
    mach=0.1,
    alphas=(0.0, 2.0, 4.0, 6.0),
    span_panels=24,
    chord_panels=12,
):
    if (
        not all(
            math.isfinite(v)
            for v in (
                span,
                root_chord,
                tip_chord,
                sweep,
                dihedral,
                root_twist,
                tip_twist,
            )
        )
        or min(span, root_chord, tip_chord) <= 0
    ):
        raise ValueError("Wing dimensions must be finite and positive")
    if (
        abs(sweep) >= 75
        or abs(dihedral) >= 75
        or max(abs(root_twist), abs(tip_twist)) > 30
    ):
        raise ValueError("Require sweep/dihedral <75°, section incidence ≤30°")
    root = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "chord": root_chord,
        "twist": root_twist,
    }
    tip = {
        "x": span / 2 * math.tan(math.radians(sweep)),
        "y": span / 2,
        "z": span / 2 * math.tan(math.radians(dihedral)),
        "chord": tip_chord,
        "twist": tip_twist,
    }
    if airfoil:
        root["airfoil"] = tip["airfoil"] = str(
            Path(airfoil).expanduser().resolve()
        )
    else:
        root["naca"] = str(naca)
        tip["naca"] = str(tip_naca or naca)
    config = {
        "name": name,
        "mach": mach,
        "profile_drag": 0.0,
        "surfaces": [
            {
                "name": "Wing",
                "mirror": True,
                "chord_panels": chord_panels,
                "span_panels": span_panels,
                "sections": [root, tip],
            }
        ],
        "cases": [{"alpha": a} for a in alphas],
    }
    config["reference"] = reference_geometry(config["surfaces"][0])
    return config


def reference_geometry(surface):
    """Integrate a main wing whose sections increase in spanwise y.

    Sref is the projected lifting area in model length units squared; Cref
    (mean aerodynamic chord) and Bref (tip-to-tip span) use model length units.
    Section chords vary linearly. A gap between mirrored roots contributes no
    lifting area, but remains part of the full span. The suggested moment
    reference is the area-weighted quarter-chord x, with y=z=0; an existing
    model's chosen reference point must be preserved separately by callers.
    """
    if len(surface["sections"]) < 2:
        raise ValueError(
            "Automatic reference geometry needs at least two sections"
        )
    area = chord2 = quarter = 0.0
    for a, b in zip(surface["sections"], surface["sections"][1:]):
        dy = b["y"] - a["y"]
        if dy <= 0:
            raise ValueError(
                "Automatic reference geometry requires increasing spanwise y"
            )
        c, d = a["chord"], b["chord"]
        area += dy * (c + d) / 2
        chord2 += dy * (c * c + c * d + d * d) / 3
        q0 = a["x"] + 0.25 * c
        q1 = b["x"] + 0.25 * d
        quarter += dy * (2 * c * q0 + c * q1 + d * q0 + 2 * d * q1) / 6
    mirrored = surface.get("mirror", True)
    root_y = surface["sections"][0]["y"]
    tip_y = surface["sections"][-1]["y"]
    if mirrored and root_y < 0:
        raise ValueError("Mirrored surfaces use the nonnegative y half only")
    # The reflected tip lies at -tip_y even when the root is offset from y=0.
    span = 2 * tip_y if mirrored else tip_y - root_y
    return {
        "area": area * (2 if mirrored else 1),
        "chord": chord2 / area,
        "span": span,
        "point": [quarter / area, 0.0, 0.0],
    }


def clean_model(config):
    config = copy.deepcopy(config)
    for surface in config["surfaces"]:
        for section in surface["sections"]:
            section.pop("coordinates", None)
            section.pop("airfoil_sha256", None)
    return config


def wing_mesh(config):
    triangles = []
    sections = []
    for surface in config["surfaces"]:
        for loft in surface_lofts(surface, samples=35, span_steps=3):
            triangles.extend(loft_triangles(loft).tolist())
        for i, s in enumerate(surface["sections"]):
            left = surface["sections"][max(0, i - 1)]
            right = surface["sections"][
                min(i + 1, len(surface["sections"]) - 1)
            ]
            ring = section_ring(
                s,
                [0, right["y"] - left["y"], right["z"] - left["z"]],
                samples=51,
            )
            sections.append(
                {
                    "surface": surface["name"],
                    "index": i + 1,
                    "label": (
                        f"{surface['name']} · {i+1} · y={s['y']:g}, "
                        f"c={s['chord']:g}, i={s['twist']:+g}° · "
                        f"{s.get('naca', Path(s.get('airfoil', '')).name)}"
                    ),
                    "ring": ring.tolist(),
                }
            )
    return {
        "name": config["name"],
        "triangles": triangles,
        "sections": sections,
        "reference": config["reference"],
    }


def write_wing_interactive(config, destination):
    payload = (
        json.dumps(wing_mesh(config), ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    template = (
        Path(__file__).with_name("assets").joinpath("wing.html").read_text()
    )
    atomic_text(destination, template.replace("__WING_DATA__", payload))
    return destination


def wing_wizard(app):
    from ..avl import find_avl, run_wing

    print("机翼向导：直接输入尺寸，回车保留默认；所有长度使用同一单位。")
    source = input("载入已有机翼 JSON（回车新建）: ").strip().strip("\"'")
    if source:
        model = clean_model(load_wing(Path(source).expanduser().resolve()))
        for surface in model["surfaces"]:
            for index, section in enumerate(surface["sections"], 1):
                print(f"{surface['name']} 截面 {index}")
                for key in ("x", "y", "z", "chord", "twist"):
                    value = input(f"  {key} [{section[key]:g}]: ").strip()
                    if value:
                        section[key] = float(value)
                foil = input(
                    "  NACA 或坐标路径 ["
                    f"{section.get('naca', section.get('airfoil', ''))}"
                    "]: "
                ).strip()
                if foil:
                    section.pop("naca", None)
                    section.pop("airfoil", None)
                    section["naca" if foil.isdigit() else "airfoil"] = foil
        print(
            f"保留已有参考量：{model['reference']}；长度为模型单位，面积为模型单位²。"
        )
        if (
            input("按修改后的主翼重新计算参考面积、平均气动弦和翼展？[y/N]: ")
            .strip()
            .lower()
            == "y"
        ):
            reference = reference_geometry(model["surfaces"][0])
            if (
                input(
                    "同时重置力矩参考点为主翼加权 1/4 弦位置（y=z=0）？[y/N]: "
                )
                .strip()
                .lower()
                != "y"
            ):
                reference["point"] = model["reference"]["point"]
            model["reference"] = reference
    else:
        values = {}
        for key, label, default in (
            ("span", "全翼展", 8.0),
            ("root_chord", "根弦长", 1.0),
            ("tip_chord", "尖弦长", 1.0),
            ("sweep", "前缘后掠角", 0.0),
            ("dihedral", "上反角", 0.0),
            ("root_twist", "根部安装角", 0.0),
            ("tip_twist", "翼尖安装角", -1.0),
            ("mach", "Mach", 0.1),
        ):
            values[key] = float(
                input(f"{label} [{default:g}]: ").strip() or default
            )
        foil = input("NACA 四位编号或坐标文件 [0012]: ").strip() or "0012"
        values["naca" if foil.isdigit() else "airfoil"] = foil
        model = simple_wing(**values)
    angles = input("计算攻角，空格分隔 [0 2 4 6]: ").split() or [0, 2, 4, 6]
    model["cases"] = [{"alpha": float(a)} for a in angles]
    folder = app.run_root / "projects" / f"wing_{timestamp_label()}"
    source = folder / "model.json"
    atomic_json(source, model)
    validated = load_wing(source)
    preview = write_wing_interactive(validated, folder / "preview.html")
    import webbrowser

    webbrowser.open(preview.as_uri())
    print(f"预览：{preview}\n参考量：{model['reference']}")
    if input("开始 AVL 计算？[Y/n]: ").strip().lower() != "n":
        output = run_wing(source, folder / "results", find_avl(app.app_root))
        print(f"结果：{output.parent / 'report.html'}")
    return source
