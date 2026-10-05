"""The airfoil database is fetched, and fetched safely.

The database is copyrighted upstream with no redistribution grant, so the
repository no longer carries it. That makes the download path load-bearing:
it has to verify what it got, refuse a payload it cannot vouch for, and
never let a crafted archive write outside the directory it owns. These
tests pin those properties without touching the network.
"""

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from xfoil_mac import airfoils


def archive(entries: dict) -> bytes:
    """A zip payload built from ``{name: text}``."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        for name, text in entries.items():
            handle.writestr(name, text)
    return buffer.getvalue()


class FakeResponse:
    """Stands in for the object urlopen returns."""

    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self, _limit: int = -1) -> bytes:
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class DatabaseStateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.app = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_absent_database_is_reported_as_not_installed(self):
        self.assertIsNone(airfoils.find_database(self.app))
        self.assertIn("not installed", airfoils.describe(self.app))

    def test_a_directory_without_coordinates_counts_as_absent(self):
        airfoils.database_dir(self.app).mkdir()
        self.assertIsNone(airfoils.find_database(self.app))

    def test_present_database_is_counted(self):
        root = airfoils.database_dir(self.app)
        root.mkdir()
        for name in ("a.dat", "b.dat"):
            (root / name).write_text("1 0\n")
        self.assertEqual(airfoils.find_database(self.app), root)
        self.assertIn("2 coordinate files", airfoils.describe(self.app))

    def test_require_names_the_command_that_installs_it(self):
        with self.assertRaises(ValueError) as caught:
            airfoils.require_database(self.app)
        message = str(caught.exception)
        self.assertIn("--install-airfoils", message)
        self.assertIn(airfoils.DATABASE_SOURCE, message)
        self.assertIn("Selig", message)


class DatabaseInstallTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.app = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.payload = archive(
            {
                "coord_seligFmt/alpha.dat": "1 0\n",
                "coord_seligFmt/beta.dat": "0 0\n",
                "coord_seligFmt/readme.txt": "not a coordinate file\n",
            }
        )
        self.digest = hashlib.sha256(self.payload).hexdigest()

    def install(self, payload=None, digest=None, **kwargs):
        """Install from ``payload``, accepting it unless told otherwise."""
        payload = self.payload if payload is None else payload
        digest = (
            hashlib.sha256(payload).hexdigest() if digest is None else digest
        )
        with patch.object(airfoils, "DATABASE_SHA256", digest):
            with patch.object(
                airfoils.urllib.request,
                "urlopen",
                return_value=FakeResponse(payload),
            ):
                return airfoils.install_database(self.app, **kwargs)

    def test_installs_and_ignores_non_coordinate_entries(self):
        root = self.install()
        self.assertEqual(airfoils.coordinate_count(root), 2)
        self.assertFalse((root / "readme.txt").exists())
        self.assertTrue((root / "alpha.dat").is_file())

    def test_records_provenance(self):
        root = self.install()
        record = json.loads((root / "install.json").read_text())
        self.assertEqual(record["url"], airfoils.DATABASE_URL)
        self.assertEqual(record["sha256"], self.digest)
        self.assertEqual(record["files"], 2)
        self.assertIn("Selig", record["attribution"])

    def test_refuses_to_overwrite_an_existing_database(self):
        (airfoils.database_dir(self.app)).mkdir()
        (airfoils.database_dir(self.app) / "keep.dat").write_text("1 0\n")
        with self.assertRaises(ValueError) as caught:
            self.install()
        self.assertIn("--force", str(caught.exception))
        self.assertTrue(
            (airfoils.database_dir(self.app) / "keep.dat").is_file()
        )

    def test_force_replaces_the_existing_database(self):
        root = airfoils.database_dir(self.app)
        root.mkdir()
        (root / "stale.dat").write_text("1 0\n")
        self.install(force=True)
        self.assertFalse((root / "stale.dat").is_file())
        self.assertTrue((root / "alpha.dat").is_file())

    def test_a_checksum_mismatch_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.install(digest="0" * 64)
        self.assertIn("checksum", str(caught.exception))
        self.assertFalse(airfoils.database_dir(self.app).exists())

    def test_an_oversized_payload_is_refused(self):
        with patch.object(airfoils, "DATABASE_MAX_BYTES", 8):
            with self.assertRaises(ValueError) as caught:
                self.install()
        self.assertIn("exceeded", str(caught.exception))

    def test_an_archive_without_coordinates_is_refused(self):
        payload = archive({"coord_seligFmt/readme.txt": "nothing here\n"})
        with self.assertRaises(ValueError) as caught:
            self.install(payload)
        self.assertIn("no .dat files", str(caught.exception))
        self.assertFalse(airfoils.database_dir(self.app).exists())


class ArchiveExtractionTest(unittest.TestCase):
    """A crafted archive must not choose where its files land."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.staging = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_traversal_names_are_flattened_into_the_destination(self):
        payload = archive({"../../escaped.dat": "evil\n"})
        written = airfoils._extract(payload, self.staging)
        self.assertEqual(written, 1)
        self.assertTrue((self.staging / "escaped.dat").is_file())
        self.assertFalse((self.staging.parent / "escaped.dat").exists())

    def test_nested_names_land_flat(self):
        payload = archive({"a/b/c/nested.dat": "1 0\n"})
        airfoils._extract(payload, self.staging)
        self.assertTrue((self.staging / "nested.dat").is_file())
        self.assertFalse((self.staging / "a").exists())

    def test_hidden_and_non_coordinate_files_are_skipped(self):
        payload = archive(
            {
                "ok.dat": "1 0\n",
                ".hidden.dat": "1 0\n",
                "ignored.csv": "1,0\n",
            }
        )
        self.assertEqual(airfoils._extract(payload, self.staging), 1)
        self.assertTrue((self.staging / "ok.dat").is_file())
        self.assertFalse((self.staging / ".hidden.dat").exists())

    def test_directories_are_skipped(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as handle:
            handle.writestr("folder/", "")
            handle.writestr("folder/one.dat", "1 0\n")
        self.assertEqual(airfoils._extract(buffer.getvalue(), self.staging), 1)


if __name__ == "__main__":
    unittest.main()
