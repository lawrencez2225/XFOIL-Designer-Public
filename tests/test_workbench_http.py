"""Opt-in HTTP boundary checks against the genuine loopback application."""

import http.client
import json
import os
import tempfile
import threading
import unittest
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path
from xfoil_mac.runtime import discover_app
from xfoil_mac.ui.server import Workbench, make_handler


@unittest.skipUnless(
    os.environ.get("XFOIL_HTTP_TESTS") == "1",
    "Set XFOIL_HTTP_TESTS=1 to bind a loopback test server",
)
class WorkbenchHttpTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = replace(discover_app(), run_root=self.root)
        self.workbench = Workbench(self.app)
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(self.workbench)
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()
        self.addCleanup(self.stop)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.workbench.close()
        self.thread.join(timeout=2)

    def request(self, path="/", method="GET", data=None, headers=None):
        c = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=3
        )
        try:
            c.request(
                method,
                path,
                body=json.dumps(data) if data is not None else None,
                headers=headers or {},
            )
            r = c.getresponse()
            return r.status, r.read(), dict(r.getheaders())
        finally:
            c.close()

    def test_login_cookie_and_token_required(self):
        self.assertEqual(self.request("/api/state")[0], 403)
        status, body, headers = self.request("/?token=" + self.workbench.token)
        self.assertEqual(status, 200)
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        self.assertIn("SameSite=Strict", headers["Set-Cookie"])
        self.assertIn("XFOIL".encode(), body)
        self.assertEqual(
            self.request(
                "/api/state", headers={"X-Xfoil-Token": self.workbench.token}
            )[0],
            200,
        )

    def test_cross_origin_and_host_rejected(self):
        spec = {"kind": "polar", "speed": 20}
        h = {
            "X-Xfoil-Token": self.workbench.token,
            "Origin": "https://example.com",
        }
        self.assertEqual(
            self.request("/api/preflight", "POST", spec, h)[0], 403
        )
        h = {"X-Xfoil-Token": self.workbench.token, "Host": "evil.example"}
        self.assertEqual(self.request("/api/state", headers=h)[0], 403)

    def test_path_escape_and_unlisted_file_rejected(self):
        h = {"X-Xfoil-Token": self.workbench.token}
        self.assertEqual(
            self.request("/files/../private.txt", headers=h)[0], 400
        )
        (self.root / "link").symlink_to("/etc")
        self.assertEqual(self.request("/files/link/passwd", headers=h)[0], 400)
        (self.root / "private.env").write_text("not served")
        self.assertEqual(self.request("/files/private.env", headers=h)[0], 404)

    def test_authenticated_preflight(self):
        h = {
            "X-Xfoil-Token": self.workbench.token,
            "Origin": f"http://127.0.0.1:{self.server.server_port}",
        }
        status, body, _ = self.request(
            "/api/preflight",
            "POST",
            {"kind": "polar", "speed": 20, "aseq": [0, 4, 2]},
            h,
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["requested_points"], 3)
