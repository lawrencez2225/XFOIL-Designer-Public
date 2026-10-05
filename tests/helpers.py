"""Small deterministic solver fixture shared by workflow tests."""

from xfoil_mac.data import POLAR_FIELDS, alpha_values

GEOMETRY = "sample\n1 0\n0.5 0.08\n0 0\n0.5 -0.05\n1 0\n"


def fake_solver(binary, env, work_dir, commands, timeout):
    sequence = next(c for c in commands if c.startswith("ASEQ ")).split()
    angles = alpha_values(*map(float, sequence[1:]))
    (work_dir / "geometry.dat").write_text(GEOMETRY)
    (work_dir / "commands.txt").write_text("\n".join(commands))
    (work_dir / "xfoil.log").write_text("fake solver\n")
    (work_dir / "polar.txt").write_text(
        " ".join(POLAR_FIELDS)
        + "\n"
        + "\n".join(f"{a} {a * .1} .01 .003 -.02 .5 .6 50 60" for a in angles)
    )
    for i, alpha in enumerate(angles, 1):
        (work_dir / f"cp_sequence_{i:04d}.txt").write_text(
            "1 0\n.5 -1\n0 0\n.5 .2\n1 0\n"
        )
    return "finished", 0
