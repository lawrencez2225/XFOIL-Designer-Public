"""Application paths, flow assumptions, and the bundled macOS runtime."""

from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from .data import positive_finite

# Mach-O magic numbers, used to recognise the shipped macOS binaries without
# shelling out to `file`. Little-endian 64-bit is what the arm64 build uses;
# the others are accepted so a universal or x86_64 build is classified as
# Mach-O rather than mistaken for a native binary of some other platform.
_MACHO_MAGICS = {
    b"\xcf\xfa\xed\xfe",  # 64-bit little-endian (arm64, x86_64)
    b"\xce\xfa\xed\xfe",  # 32-bit little-endian
    b"\xfe\xed\xfa\xcf",  # 64-bit big-endian
    b"\xfe\xed\xfa\xce",  # 32-bit big-endian
    b"\xca\xfe\xba\xbe",  # universal, big-endian header
    b"\xbe\xba\xfe\xca",  # universal, little-endian header
}


@dataclass(frozen=True)
class AppPaths:
    app_root: Path
    xfoil_home: Path
    xfoil_bin: Path
    runtime_lib: Path
    run_root: Path
    single_root: Path
    batch_root: Path
    xfoil_is_bundled: bool = True
    """Whether ``xfoil_bin`` is the shipped binary or one found on PATH.

    The bundled binary needs ``runtime_lib`` on the loader path; a solver
    the user installed themselves brings its own libraries and must not be
    handed ours.
    """


@dataclass(frozen=True)
class FlowAssumptions:
    chord_m: float = 1.0
    air_density: float = 1.225
    air_viscosity: float = 1.7894e-5
    sound_speed: float = 340.3

    def __post_init__(self) -> None:
        for name in ("chord_m", "air_density", "air_viscosity", "sound_speed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite positive number")
            positive_finite(value, name)

    @classmethod
    def from_env(cls, **overrides: float | None) -> "FlowAssumptions":
        names = {
            "chord_m": "XFOIL_CHORD_M",
            "air_density": "XFOIL_AIR_DENSITY",
            "air_viscosity": "XFOIL_AIR_VISCOSITY",
            "sound_speed": "XFOIL_SOUND_SPEED",
        }
        if set(overrides) - names.keys():
            raise ValueError("Unknown flow assumption")
        return cls(
            **{
                name: (
                    overrides[name]
                    if overrides.get(name) is not None
                    else float(os.environ.get(env_name, getattr(cls, name)))
                )
                for name, env_name in names.items()
            }
        )

    def describe(self) -> str:
        return (
            f"chord={self.chord_m:g} m, rho={self.air_density:g} kg/m^3, "
            f"mu={self.air_viscosity:g} kg/(m*s), a={self.sound_speed:g} m/s"
        )


def die(message: str) -> NoReturn:
    raise ValueError(message)


def validate_xfoil_mach(mach: float) -> None:
    if not math.isfinite(mach):
        die("Mach must be a finite number")
    if mach < 0:
        die("Mach must be non-negative")
    if mach >= 1:
        die(
            "XFOIL only accepts subsonic freestream Mach values (Mach < 1). "
            f"Current Mach={mach:g}. Enter a subsonic value below 1 "
            "(0.3 is a reasonable conservative choice), or leave "
            "Mach blank and use a lower Re."
        )


def estimate_re_from_mach(mach: float, assumptions: FlowAssumptions) -> int:
    return round(
        assumptions.air_density
        * mach
        * assumptions.sound_speed
        * assumptions.chord_m
        / assumptions.air_viscosity
    )


def estimate_mach_from_re(
    reynolds: float, assumptions: FlowAssumptions
) -> float:
    return (
        reynolds
        * assumptions.air_viscosity
        / (
            assumptions.air_density
            * assumptions.chord_m
            * assumptions.sound_speed
        )
    )


FLOW_RELATIONS = (
    "defaults",
    "derived_re_from_mach",
    "derived_mach_from_re",
    "both_matched",
    "independent",
)
"""How a run's Reynolds number and Mach number came to be paired.

Re and Mach are independent dimensionless numbers: Re scales with chord
and viscosity, Mach does not. They are therefore only linked once a
chord and a fluid are fixed. A run that carried both in as free inputs
can hold a pair that no single chord and fluid could produce, and XFOIL
will still solve it, so the pairing is recorded rather than inferred.

``independent`` is accepted deliberately, because a non-physical pair is
a legitimate thing to compute. It is recorded so that a reader can tell
it apart from a pair derived from a stated physical assumption.
"""


def resolve_flow_inputs(
    reynolds: float | None,
    mach: float | None,
    assumptions: FlowAssumptions | None = None,
) -> tuple[float, float, str, str]:
    assumptions = assumptions or FlowAssumptions.from_env()
    for name in ("chord_m", "air_density", "air_viscosity", "sound_speed"):
        positive_finite(getattr(assumptions, name), name)
    if reynolds is not None:
        positive_finite(reynolds, "Re")
    if mach is not None:
        validate_xfoil_mach(mach)
    note = ""
    relation = "defaults"
    if reynolds is None and mach is None:
        reynolds = float(os.environ.get("XFOIL_DEFAULT_RE", "1000000"))
        mach = float(os.environ.get("XFOIL_DEFAULT_MACH", "0.0"))
        note = f"Using defaults: Re={reynolds:g}, Mach={mach:g}."
    elif reynolds is None:
        if mach == 0:
            reynolds = float(os.environ.get("XFOIL_DEFAULT_RE", "1000000"))
            note = f"Mach is zero, so using default Re={reynolds:g}."
        else:
            reynolds = float(estimate_re_from_mach(mach, assumptions))
            note = f"Estimated Re from Mach using {assumptions.describe()}."
            relation = "derived_re_from_mach"
    elif mach is None:
        mach = estimate_mach_from_re(reynolds, assumptions)
        note = f"Estimated Mach from Re using {assumptions.describe()}."
        relation = "derived_mach_from_re"
    elif mach > 0:
        expected_mach = estimate_mach_from_re(reynolds, assumptions)
        relation = "independent"
        if math.isclose(mach, expected_mach, rel_tol=0.001, abs_tol=1e-8):
            relation = "both_matched"
        else:
            note = (
                f"Re/Mach supplied independently: keeping "
                f"Re={reynolds:g}, Mach={mach:g}. "
                f"With {assumptions.describe()}, this Re implies "
                f"Mach={expected_mach:.6g}. "
                "For one physical operating condition, supply only Re or "
                "only Mach."
            )
    positive_finite(reynolds, "Re")
    validate_xfoil_mach(mach)
    return float(reynolds), float(mach), note, relation


def bundled_solver_diagnosis(binary: Path) -> str | None:
    """Why ``binary`` cannot run on this machine, or None when it can.

    The shipped XFOIL, PPLOT and PXPLOT are arm64 macOS executables linked
    against the libraries in ``runtime/lib``. On any other platform the
    kernel refuses to start them, and what reaches the user is a bare
    ``Bad CPU type in executable`` or ``Exec format error`` that says
    nothing about this project. This turns that into a sentence naming the
    mismatch.

    A binary that is not Mach-O is accepted: it is either a rebuild done
    for this platform in place, or something the operating system will
    judge for itself. Only the cases where a wrong-architecture failure is
    guaranteed are reported.
    """
    if not binary.is_file():
        return f"{binary} is missing"
    try:
        header = binary.read_bytes()[:4]
    except OSError as error:
        return f"{binary} cannot be read ({error})"
    if header not in _MACHO_MAGICS:
        if not os.access(binary, os.X_OK):
            return f"{binary} is not executable"
        return None
    if sys.platform != "darwin":
        return (
            f"the bundled XFOIL is a macOS executable, but this system "
            f"reports {sys.platform}"
        )
    machine = platform.machine()
    if machine != "arm64":
        return (
            f"the bundled XFOIL is an arm64 macOS binary, but this Mac "
            f"reports {machine}"
        )
    return None


def find_xfoil(xfoil_home: Path) -> tuple[Path, bool]:
    """The solver to run, and whether it is the bundled one.

    Falls back to an ``xfoil`` on PATH so that a user on another platform
    can supply their own build instead of being stopped by a binary that
    cannot start.
    """
    bundled = xfoil_home / "bin" / "xfoil"
    problem = bundled_solver_diagnosis(bundled)
    if problem is None:
        return bundled, True
    external = shutil.which("xfoil")
    if external:
        return Path(external), False
    raise ValueError(
        f"Cannot run XFOIL: {problem}, and no 'xfoil' was found on PATH.\n"
        "Options:\n"
        "  1. Run this on Apple Silicon macOS, where the bundled solver "
        "works.\n"
        "  2. Build XFOIL from the Fortran source in "
        f"{xfoil_home / 'source' / 'Xfoil'}, then put it on PATH.\n"
        "  3. Point at an existing solver by adding its directory to "
        "PATH.\n"
        "See CONTRIBUTING.md for what a platform port involves."
    )


def discover_app(start: Path | None = None) -> AppPaths:
    base = (start or Path(__file__)).expanduser().resolve()
    script_dir = base.parent if base.is_file() else base
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "bin" / "xfoil").is_file() and (
            candidate / "runtime" / "lib"
        ).is_dir():
            xfoil_home = candidate
            app_root = candidate.parent
            break
        home = candidate / "XFOIL_6.996"
        if (home / "bin" / "xfoil").is_file() and (
            home / "runtime" / "lib"
        ).is_dir():
            app_root = candidate
            xfoil_home = home
            break
    else:
        die(
            f"Could not find the XFOIL_6.996 directory from {script_dir}. "
            "The solver and its runtime libraries ship in the repository, "
            "so this usually means the package was installed somewhere "
            "other than the checkout. Install it in place with "
            "`pip install -e .` from the repository root."
        )
    runtime_lib = xfoil_home / "runtime" / "lib"
    run_root = app_root / "xfoil_runs"
    xfoil_bin, is_bundled = find_xfoil(xfoil_home)
    return AppPaths(
        app_root=app_root,
        xfoil_home=xfoil_home,
        xfoil_bin=xfoil_bin,
        runtime_lib=runtime_lib,
        run_root=run_root,
        single_root=run_root / "single_calculations",
        batch_root=run_root / "batch_calculations",
        xfoil_is_bundled=is_bundled,
    )


def configure_matplotlib_cache(app: AppPaths) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    cache_dir = app.run_root / "_cache" / "matplotlib"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    except OSError:
        pass


def batch_preview_enabled() -> bool:
    disabled = (
        os.environ.get("XFOIL_DISABLE_BATCH_PREVIEW", "").strip().lower()
    )
    value = os.environ.get("XFOIL_BATCH_PREVIEW", "1").strip().lower()
    return disabled not in {"1", "true", "yes", "on"} and value not in {
        "0",
        "false",
        "no",
        "off",
    }


def xfoil_env(app: AppPaths) -> dict[str, str]:
    env = os.environ.copy()
    if app.xfoil_is_bundled:
        # The shipped binary loads its libraries from runtime/lib. A solver
        # the user installed themselves brings its own, and prepending ours
        # could shadow them with an incompatible copy.
        variable = (
            "DYLD_LIBRARY_PATH"
            if sys.platform == "darwin"
            else "LD_LIBRARY_PATH"
        )
        env[variable] = f"{app.runtime_lib}:{env.get(variable, '')}"
        # Resource locations follow the checkout when it is moved. The
        # neutral /opt/X11 defaults in libX11 belong to XQuartz; prefer
        # the matching bundled locale and colour database here.
        resources = app.runtime_lib.parent / "share" / "X11"
        env["XLOCALEDIR"] = str(resources / "locale")
        env["XCMSDB"] = str(resources / "Xcms.txt")
    env.setdefault("DISPLAY", ":0")
    # Xplot11 uses its native reverse-video (black) background when unset.
    env.pop("XPLOT11_BACKGROUND", None)
    return env


def start_xquartz() -> None:
    xquartz = Path("/Applications/Utilities/XQuartz.app")
    if not xquartz.exists():
        die(
            "XQuartz is not installed. Install XQuartz before running "
            "plotted XFOIL workflows."
        )
    subprocess.run(
        ["open", "-a", "XQuartz"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    time.sleep(float(os.environ.get("XFOIL_XQUARTZ_START_WAIT", "2")))


def live_cpx_pause_seconds() -> float:
    return max(0.0, float(os.environ.get("XFOIL_LIVE_CPX_PAUSE", "0.0")))
