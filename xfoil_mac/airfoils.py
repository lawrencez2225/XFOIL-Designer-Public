"""The UIUC airfoil coordinate database, fetched rather than redistributed.

The database is published by the UIUC Applied Aerodynamics Group under a
copyright notice with no redistribution grant, so this repository does not
carry a copy of it. It is downloaded on first use, checked against a pinned
checksum, and unpacked into the directory the rest of the program already
reads shapes from. This mirrors how ``avl.py`` handles AVL.

Deleting the directory and running the installer again is always safe; the
database holds no state that a run depends on beyond the coordinates
themselves, and every run copies the geometry it actually used into its own
output.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .data import atomic_json

DATABASE_NAME = "coord_seligFmt"
"""Directory, relative to the application root, that holds the .dat files."""

DATABASE_URL = (
    "https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip"
)

DATABASE_SHA256 = (
    "5a17e5400322d7d1b0171554cee3764bcf34aa2cc349e0e4e95aeafe5062fda7"
)
"""Checksum of the archive this project was tested against.

Upstream revises the database in place: entries are added and existing
coordinates are corrected. A file set that differs from the tested one can
change a computed result, so a mismatch is refused rather than accepted.
When upstream publishes a new revision, review the changes, update this
constant, and note the new revision in THIRD_PARTY_NOTICES.md.
"""

DATABASE_SOURCE = "https://m-selig.ae.illinois.edu/ads/coord_database.html"

DATABASE_ATTRIBUTION = (
    "UIUC Airfoil Coordinates Database, Michael S. Selig, "
    "University of Illinois at Urbana-Champaign"
)

DATABASE_MAX_BYTES = 64 * 1024 * 1024
"""Refuse anything larger than this; the real archive is about 1 MB."""


def database_dir(app_root: Path) -> Path:
    """Where the coordinate files belong, installed or not."""
    return Path(app_root) / DATABASE_NAME


def coordinate_count(root: Path) -> int:
    """Count coordinate files; return zero if the directory is unreadable."""
    try:
        return sum(1 for _ in Path(root).glob("*.dat"))
    except OSError:
        return 0


def find_database(app_root: Path) -> Path | None:
    """The populated database directory, or None when it is not installed."""
    root = database_dir(app_root)
    return root if coordinate_count(root) else None


def require_database(app_root: Path) -> Path:
    """The database directory, or an error saying how to obtain it."""
    root = find_database(app_root)
    if root is None:
        raise ValueError(
            "The airfoil coordinate database is not installed.\n"
            "Fetch it with: python -m xfoil_mac --install-airfoils\n"
            f"Source: {DATABASE_SOURCE}\n"
            f"Attribution: {DATABASE_ATTRIBUTION}"
        )
    return root


def describe(app_root: Path) -> str:
    """One line on the state of the database, for a status report."""
    root = database_dir(app_root)
    count = coordinate_count(root)
    if not count:
        return f"{root}: not installed"
    return f"{root}: {count} coordinate files"


def _extract(payload: bytes, destination: Path) -> int:
    """Write the archive's coordinate files into ``destination``.

    Entry names are flattened to their basename and only ``.dat`` files are
    taken, so a crafted archive cannot write outside the destination.
    """
    written = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            name = Path(entry.filename).name
            if not name.endswith(".dat") or name.startswith("."):
                continue
            with archive.open(entry) as source:
                with open(destination / name, "wb") as target:
                    shutil.copyfileobj(source, target)
            written += 1
    if not written:
        raise ValueError("The downloaded archive contains no .dat files")
    return written


def install_database(app_root: Path, *, force: bool = False) -> Path:
    """Download, verify, and unpack the database; return its directory.

    Unpacking goes to a staging directory beside the destination so an
    interrupted download cannot leave a half-populated database behind.
    """
    root = database_dir(app_root)
    present = coordinate_count(root)
    if present and not force:
        raise ValueError(
            f"{root} already holds {present} coordinate files. "
            "Pass --force to download and replace it."
        )
    with urllib.request.urlopen(DATABASE_URL, timeout=120) as response:
        payload = response.read(DATABASE_MAX_BYTES + 1)
    if len(payload) > DATABASE_MAX_BYTES:
        raise ValueError(
            f"The database download exceeded {DATABASE_MAX_BYTES} bytes; "
            "refusing it. Check the URL before retrying."
        )
    digest = hashlib.sha256(payload).hexdigest()
    if digest != DATABASE_SHA256:
        raise ValueError(
            "The published database does not match the tested checksum.\n"
            f"  expected {DATABASE_SHA256}\n"
            f"  received {digest}\n"
            "Upstream may have published a new revision. Review the changes "
            "before updating DATABASE_SHA256 in xfoil_mac/airfoils.py."
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{DATABASE_NAME}-", dir=root.parent)
    )
    try:
        written = _extract(payload, staging)
        if root.exists():
            shutil.rmtree(root)
        staging.replace(root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    atomic_json(
        root / "install.json",
        {
            "url": DATABASE_URL,
            "sha256": DATABASE_SHA256,
            "files": written,
            "source_url": DATABASE_SOURCE,
            "attribution": DATABASE_ATTRIBUTION,
        },
    )
    return root
