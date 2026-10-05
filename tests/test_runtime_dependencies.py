"""The trimmed solver runtime must include every non-system dependency."""

import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    sys.platform == "darwin" and os.environ.get("XFOIL_NATIVE_TESTS") == "1",
    "Requires macOS and the local solver binaries",
)
class RuntimeDependenciesTest(unittest.TestCase):
    def native_artifacts(self):
        return [
            ROOT / "XFOIL_6.996/bin" / name
            for name in ("xfoil", "pplot", "pxplot")
        ] + sorted((ROOT / "XFOIL_6.996/runtime/lib").glob("*.dylib"))

    def test_native_artifacts_do_not_embed_home_directories(self):
        for artifact in self.native_artifacts():
            with self.subTest(artifact=artifact.name):
                self.assertNotIn(b"/Users/", artifact.read_bytes())

    def test_native_code_signatures_are_valid(self):
        for artifact in self.native_artifacts():
            with self.subTest(artifact=artifact.name):
                result = subprocess.run(
                    [
                        "/usr/bin/codesign",
                        "--verify",
                        "--strict",
                        str(artifact),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_cleaned_artifacts_match_packaging_record(self):
        home = ROOT / "XFOIL_6.996"
        record = json.loads((home / "runtime/packaging.json").read_text())
        for name, expected in record["files"].items():
            with self.subTest(artifact=name):
                digest = hashlib.sha256((home / name).read_bytes()).hexdigest()
                self.assertEqual(digest, expected["clean_sha256"])

    def test_all_native_library_dependencies_are_bundled(self):
        library_root = ROOT / "XFOIL_6.996/runtime/lib"
        pending = [
            ROOT / "XFOIL_6.996/bin" / name
            for name in ("xfoil", "pplot", "pxplot")
        ]
        visited = set()
        while pending:
            binary = pending.pop().resolve()
            if binary in visited:
                continue
            visited.add(binary)
            result = subprocess.run(
                ["/usr/bin/otool", "-L", str(binary)],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.splitlines()[1:]:
                dependency = line.strip().split(" (", 1)[0]
                if dependency.startswith(("/usr/lib/", "/System/Library/")):
                    continue
                bundled = library_root / Path(dependency).name
                self.assertTrue(
                    bundled.is_file(),
                    f"{binary.name} requires missing {dependency}",
                )
                pending.append(bundled)


if __name__ == "__main__":
    unittest.main()
