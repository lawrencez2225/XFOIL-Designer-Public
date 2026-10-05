"""Authenticated loopback workbench with a serial queue and cancellable solver
jobs.
"""

import hmac
import json
import mimetypes
import os
import queue
import secrets
import signal
import subprocess
import sys
import threading
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit, quote

from .. import airfoils
from ..data import atomic_json, atomic_text, timestamp_label
from ..storage.catalog import (
    history,
    presets,
    save_preset,
    within,
    export_report,
    spec_from_history,
)
from .service import preflight, new_job_root


class Workbench:
    def __init__(self, app):
        self.app = app
        self.token = secrets.token_urlsafe(32)
        self.jobs = {}
        self.lock = threading.RLock()
        self.pending = queue.Queue()
        self.closing = False
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def check_paths(self, spec):
        """Browser requests only consume project-local coordinates and saved
        results.
        """

        def visit(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "airfoil" and child:
                        path = within(self.app.app_root, str(child))
                        if (
                            path.suffix.lower() not in {".dat", ".txt"}
                            or not path.is_file()
                            or path.stat().st_size > 2 * 1024**2
                        ):
                            raise ValueError("请选择项目内的翼型坐标文件")
                        value[key] = str(path)
                    else:
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(spec)
        if spec.get("source_run"):
            spec["source_run"] = str(
                within(self.app.run_root, spec["source_run"])
            )
        if "folders" in spec:
            spec["folders"] = [
                str(within(self.app.run_root, p)) for p in spec["folders"]
            ]
        return spec

    def submit(self, spec, folder=None):
        spec = self.check_paths(spec)
        preflight(spec)
        with self.lock:
            if (
                sum(
                    v["status"] in {"queued", "running"}
                    for v in self.jobs.values()
                )
                >= 32
            ):
                raise ValueError("队列已满，请等待当前计算完成")
            root = (
                within(self.app.run_root, folder)
                if folder
                else new_job_root(self.app, spec)
            )
            key = str(root.relative_to(self.app.run_root))
            if key in self.jobs and self.jobs[key]["status"] in {
                "queued",
                "running",
            }:
                raise ValueError("此计算已经在队列中")
            if folder:
                old = json.loads((root / "job.json").read_text())
                if old["spec"] != spec:
                    raise ValueError("继续计算必须保持原设置")
            root.mkdir(parents=True, exist_ok=True)
            atomic_json(
                root / "job.json",
                {
                    "spec": spec,
                    "title": spec.get("title", root.name),
                    "created": timestamp_label(),
                    "status": "queued",
                },
            )
            self.jobs[key] = {
                "id": key,
                "root": root,
                "status": "queued",
                "process": None,
            }
            self.pending.put(key)
            return key

    def work(self):
        while not self.closing:
            key = self.pending.get()
            if key is None:
                return
            with self.lock:
                job = self.jobs[key]
                if job["status"] == "cancelled":
                    continue
                log = (job["root"] / "task.log").open("a")
                env = os.environ.copy()
                env["MPLBACKEND"] = "Agg"
                env["PYTHONUNBUFFERED"] = "1"
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "xfoil_mac",
                        "--job",
                        str(job["root"]),
                        "--resume",
                    ],
                    cwd=self.app.app_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                job.update(status="running", process=process)
            code = process.wait()
            log.close()
            with self.lock:
                file = job["root"] / "job.json"
                saved = json.loads(file.read_text())
                status = (
                    "cancelled"
                    if job["status"] == "cancelled"
                    else saved.get("status", "failed")
                )
                if status in {"running", "queued"}:
                    status = "failed" if code else "ok"
                job.update(status=status, process=None)
                saved["status"] = status
                atomic_json(file, saved)

    def cancel(self, key):
        with self.lock:
            job = self.jobs.get(key)
            if not job:
                raise ValueError("此任务不在当前队列中")
            if job["status"] not in {"queued", "running"}:
                return
            job["status"] = "cancelled"
            file = job["root"] / "job.json"
            saved = json.loads(file.read_text())
            saved["status"] = "cancelled"
            atomic_json(file, saved)
            p = job["process"]
            if p and p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)

                def reap():
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(p.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

                threading.Thread(target=reap, daemon=True).start()

    def snapshot(self):
        items = []
        with self.lock:
            jobs = list(self.jobs.values())
        for job in jobs:
            count = total = 0
            for path in list(job["root"].rglob("run.json"))[:200]:
                try:
                    saved = json.loads(path.read_text())
                    count += saved.get("summary", {}).get("points", 0)
                    total += len(
                        saved.get(
                            "requested_alphas", saved["config"]["alphas"]
                        )
                    )
                except (OSError, ValueError, KeyError):
                    pass
            for path in list(job["root"].rglob("lift_run.json"))[:200]:
                try:
                    saved = json.loads(path.read_text())
                    count += sum(
                        p.get("status") == "ok"
                        for p in saved.get("points", [])
                    )
                    total += len(saved["config"]["targets"])
                except (OSError, ValueError, KeyError):
                    pass
            log = job["root"] / "task.log"
            tail = ""
            if log.exists():
                with log.open("rb") as f:
                    f.seek(max(0, log.stat().st_size - 3500))
                    tail = f.read().decode(errors="replace")
            items.append(
                {
                    "id": job["id"],
                    "status": job["status"],
                    "completed": count,
                    "total": total,
                    "log": tail,
                }
            )
        return {
            "jobs": items,
            "history": history(self.app),
            "presets": presets(self.app),
        }

    def close(self):
        self.closing = True
        for key in list(self.jobs):
            self.cancel(key)
        self.pending.put(None)


def make_handler(workbench):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def allowed_host(self):
            return (
                self.headers.get("Host")
                == f"127.0.0.1:{self.server.server_port}"
            )

        def authorized(self):
            cookie = SimpleCookie()
            try:
                cookie.load(self.headers.get("Cookie", ""))
            except Exception:
                return False
            token = self.headers.get("X-Xfoil-Token") or (
                cookie["xfoil_session"].value
                if "xfoil_session" in cookie
                else ""
            )
            return hmac.compare_digest(token, workbench.token)

        def send(
            self,
            payload,
            status=200,
            mime="application/json; charset=utf-8",
            cookie=False,
            download=None,
        ):
            body = (
                json.dumps(
                    payload, ensure_ascii=False, allow_nan=False
                ).encode()
                if not isinstance(payload, bytes)
                else payload
            )
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self' data: blob:; script-src 'self' "
                "'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                "object-src 'none'; frame-ancestors 'self'; base-uri "
                "'none'",
            )
            if cookie:
                self.send_header(
                    "Set-Cookie",
                    f"xfoil_session={workbench.token}; HttpOnly; "
                    f"SameSite=Strict; Path=/",
                )
            if download:
                self.send_header(
                    "Content-Disposition",
                    'attachment; filename="xfoil-report.zip"',
                )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                if not self.allowed_host():
                    return self.send({"error": "Invalid host"}, 403)
                parsed = urlsplit(self.path)
                auth = parse_qs(parsed.query).get("token", [""])[0]
                login = parsed.path == "/" and hmac.compare_digest(
                    auth, workbench.token
                )
                if not login and not self.authorized():
                    return self.send(
                        {"error": "请使用启动时给出的本地工作台地址"}, 403
                    )
                if parsed.path == "/":
                    template = (
                        Path(__file__)
                        .with_name("assets")
                        .joinpath("workbench.html")
                        .read_text()
                    )
                    return self.send(
                        template.replace(
                            "__SESSION_TOKEN__", workbench.token
                        ).encode(),
                        mime="text/html; charset=utf-8",
                        cookie=login,
                    )
                if parsed.path == "/api/state":
                    return self.send(workbench.snapshot())
                if parsed.path == "/api/files":
                    root = airfoils.database_dir(workbench.app.app_root)
                    files = sorted(
                        str(p.relative_to(workbench.app.app_root))
                        for p in root.glob("*.dat")
                    )
                    return self.send({"airfoils": files})
                if parsed.path.startswith("/files/"):
                    path = within(
                        workbench.app.run_root, unquote(parsed.path[7:])
                    )
                    if (
                        path.suffix.lower()
                        not in {
                            ".html",
                            ".json",
                            ".csv",
                            ".txt",
                            ".dat",
                            ".png",
                            ".svg",
                            ".zip",
                        }
                        or not path.is_file()
                    ):
                        return self.send({"error": "File not available"}, 404)
                    if path.stat().st_size > 500 * 1024**2:
                        raise ValueError("File too large")
                    return self.send(
                        path.read_bytes(),
                        mime=mimetypes.guess_type(path.name)[0]
                        or "text/plain",
                        download=path.suffix == ".zip",
                    )
                return self.send({"error": "Not found"}, 404)
            except (ValueError, OSError, KeyError) as e:
                self.send({"error": str(e)}, 400)

        def do_POST(self):
            try:
                if (
                    not self.allowed_host()
                    or not self.authorized()
                    or not hmac.compare_digest(
                        self.headers.get("X-Xfoil-Token", ""), workbench.token
                    )
                ):
                    return self.send({"error": "Unauthorized"}, 403)
                origin = self.headers.get("Origin")
                if (
                    origin
                    and origin != f"http://127.0.0.1:{self.server.server_port}"
                ):
                    return self.send({"error": "Invalid origin"}, 403)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("Request must be 1 byte..1 MB")
                value = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path == "/api/preflight":
                    return self.send(preflight(workbench.check_paths(value)))
                if path == "/api/submit":
                    return self.send({"id": workbench.submit(value)})
                if path == "/api/cancel":
                    workbench.cancel(value["id"])
                    return self.send({"ok": True})
                if path == "/api/preset":
                    preflight(value["spec"])
                    with workbench.lock:
                        save_preset(
                            workbench.app, value["name"], value["spec"]
                        )
                    return self.send({"ok": True})
                if path == "/api/load":
                    return self.send(
                        spec_from_history(
                            within(workbench.app.run_root, value["id"])
                        )
                    )
                if path == "/api/repeat":
                    folder = within(workbench.app.run_root, value["id"])
                    spec = spec_from_history(folder)
                    if (
                        value.get("resume")
                        and not (folder / "job.json").exists()
                    ):
                        raise ValueError(
                            "旧版计算请用“复制设置后重跑”；原结果继续保留。原参数续算也可使用原命令。"
                        )
                    return self.send(
                        {
                            "id": workbench.submit(
                                spec,
                                value["id"] if value.get("resume") else None,
                            )
                        }
                    )
                if path == "/api/export":
                    folder = within(workbench.app.run_root, value["id"])
                    file = export_report(folder, folder / "report.zip")
                    return self.send(
                        {
                            "url": "/files/"
                            + quote(
                                str(file.relative_to(workbench.app.run_root))
                            )
                        }
                    )
                if path == "/api/preview":
                    from .wing import (
                        simple_wing,
                        write_wing_interactive,
                        reference_geometry,
                    )
                    from ..avl import load_wing

                    recalculate = value.get("recalculate_reference", False)
                    reset_point = value.get("reset_reference_point", False)
                    if not isinstance(recalculate, bool) or not isinstance(
                        reset_point, bool
                    ):
                        raise ValueError("参考量重算选项必须为 true 或 false")
                    if reset_point and not recalculate:
                        raise ValueError(
                            "重置力矩参考点时必须明确选择重新计算参考量"
                        )
                    model = (
                        value["model"]
                        if "model" in value
                        else simple_wing(
                            **{
                                key: item
                                for key, item in value.items()
                                if key
                                not in {
                                    "recalculate_reference",
                                    "reset_reference_point",
                                }
                            }
                        )
                    )
                    workbench.check_paths(model)
                    if "reference" in model and not isinstance(
                        model["reference"], dict
                    ):
                        raise ValueError("已有参考量必须为对象")
                    if "reference" not in model or recalculate:
                        reference = reference_geometry(model["surfaces"][0])
                        if "reference" in model and not reset_point:
                            previous_reference = model["reference"]
                            # AVL's omitted reference point means the origin,
                            # not the automatically suggested quarter chord.
                            reference["point"] = previous_reference.get(
                                "point", [0.0, 0.0, 0.0]
                            )
                        model["reference"] = reference
                    folder = (
                        workbench.app.run_root
                        / "_cache"
                        / "previews"
                        / timestamp_label()
                    )
                    source = folder / "model.json"
                    atomic_json(source, model)
                    write_wing_interactive(
                        load_wing(source), folder / "preview.html"
                    )
                    return self.send(
                        {
                            "model": model,
                            "url": "/files/"
                            + quote(
                                str(
                                    (folder / "preview.html").relative_to(
                                        workbench.app.run_root
                                    )
                                )
                            ),
                        }
                    )
                if path == "/api/upload":
                    from ..geometry import inspect_points, load_coordinates
                    from ..data import safe_label

                    name = (
                        safe_label(Path(value["name"]).stem)
                        + "_"
                        + timestamp_label()
                        + ".dat"
                    )
                    text = value["text"]
                    if not isinstance(text, str) or len(text) > 200000:
                        raise ValueError("坐标文件过大")
                    folder = workbench.app.run_root / "inputs"
                    file = folder / name
                    atomic_text(file, text)
                    try:
                        report, _ = inspect_points(load_coordinates(file))
                        if not report["valid"]:
                            raise ValueError(str(report["issues"]))
                    except Exception:
                        file.unlink(missing_ok=True)
                        raise
                    return self.send(
                        {"path": str(file.relative_to(workbench.app.app_root))}
                    )
                return self.send({"error": "Not found"}, 404)
            except (ValueError, TypeError, OSError, KeyError) as e:
                self.send({"error": str(e)}, 400)

    return Handler


def serve(app, port=0, open_browser=True):
    workbench = Workbench(app)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(workbench))
    url = f"http://127.0.0.1:{server.server_port}/?token={workbench.token}"
    print(
        f"本地工作台：{url}\n关闭终端或按 Ctrl+C 将停止服务与当前计算。",
        flush=True,
    )
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.3)
    finally:
        workbench.close()
        server.server_close()
