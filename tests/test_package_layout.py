"""Check launchers and runtime discovery after modularization."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from xfoil_mac import __version__
from xfoil_mac.runtime import discover_app, xfoil_env

ROOT = Path(__file__).resolve().parents[1]


class PackageLayoutTest(unittest.TestCase):
    def test_bundled_x11_resources_follow_a_moved_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            app = replace(
                discover_app(),
                runtime_lib=Path(folder) / "runtime/lib",
                xfoil_is_bundled=True,
            )
            with patch.dict(
                os.environ,
                {
                    "XLOCALEDIR": "/external/locale",
                    "XCMSDB": "/external/Xcms.txt",
                },
            ):
                environment = xfoil_env(app)
            resources = app.runtime_lib.parent / "share/X11"
            self.assertEqual(
                environment["XLOCALEDIR"], str(resources / "locale")
            )
            self.assertEqual(
                environment["XCMSDB"], str(resources / "Xcms.txt")
            )

    def test_external_solver_keeps_its_x11_resource_configuration(self):
        app = replace(discover_app(), xfoil_is_bundled=False)
        external = {
            "XLOCALEDIR": "/external/locale",
            "XCMSDB": "/external/Xcms.txt",
        }
        with patch.dict(os.environ, external):
            environment = xfoil_env(app)
        for name, value in external.items():
            self.assertEqual(environment[name], value)

    def test_original_launcher_works_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(
                [sys.executable, str(ROOT / "xfoil_work.py"), "--version"],
                cwd=folder,
                capture_output=True,
                text=True,
                timeout=10,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), __version__)

    def test_module_and_script_expose_the_same_commands(self):
        script = subprocess.run(
            [sys.executable, "xfoil_work.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        package = subprocess.run(
            [sys.executable, "-m", "xfoil_mac", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(script.returncode, 0, script.stderr)
        self.assertEqual(package.returncode, 0, package.stderr)
        self.assertEqual(
            script.stdout[script.stdout.index("options:") :],
            package.stdout[package.stdout.index("options:") :],
        )

    def test_runtime_discovery_handles_nested_package_and_native_directory(
        self,
    ):
        expected = discover_app(ROOT / "xfoil_work.py")
        self.assertEqual(discover_app(ROOT / "xfoil_mac" / "cli.py"), expected)
        self.assertEqual(discover_app(ROOT / "XFOIL_6.996"), expected)
        self.assertEqual(discover_app(), expected)

    def test_import_does_not_initialize_matplotlib(self):
        code = (
            "import sys,json,xfoil_mac; "
            "print(json.dumps('matplotlib.pyplot' in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout))


if __name__ == "__main__":
    unittest.main()
