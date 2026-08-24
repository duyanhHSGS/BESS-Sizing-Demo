"""Local Flask control panel for the debloated BESS DRL runner.

The UI is deliberately a thin process/control layer.  It launches the same
``main.py train`` and ``main.py run`` commands used from the terminal, tails
plain-text logs, and reads plain result/state files.  No database is involved.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from flask import Flask, jsonify, render_template, request
from og_ui_compat import index_context, register_og_routes
from og_ui_runtime import register_og_runtime_routes
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

HERE = Path(__file__).resolve().parent
MAIN_PY = HERE / "main.py"
RESULTS_DIR = HERE / "results"
UPLOADS_DIR = HERE / "uploads"
LOGS_DIR = HERE / "logs"
STATE_DIR = HERE / "state"
SETPOINT_LOG = LOGS_DIR / "setpoints.jsonl"
RUNTIME_STATE = STATE_DIR / "runtime_state.json"

_SAFE_TAG = re.compile(r"^[A-Za-z0-9_.-]+$")
_ALLOWED_UPLOADS = {
    ".csv",
    ".json",
    ".pt",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _display_command(command: list[str]) -> str:
    redacted: list[str] = []
    hide_next = False
    sensitive_flags = {"--api-key", "--mqtt-password"}
    for part in command:
        if hide_next:
            redacted.append("***")
            hide_next = False
            continue
        redacted.append(part)
        if part in sensitive_flags:
            hide_next = True
    return " ".join(shlex.quote(part) for part in redacted)


@dataclass
class ProcessSlot:
    name: str
    process: subprocess.Popen[str] | None = None
    status: str = "idle"
    command: list[str] = field(default_factory=list)
    started_at: str | None = None
    ended_at: str | None = None
    return_code: int | None = None
    stop_requested: bool = False
    sequence: int = 0
    logs: deque[tuple[int, str]] = field(default_factory=lambda: deque(maxlen=4000))
    lock: threading.RLock = field(default_factory=threading.RLock)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            if running and self.status != "running":
                self.status = "running"
            return {
                "name": self.name,
                "status": self.status,
                "running": running,
                "command": _display_command(self.command) if self.command else "",
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "return_code": self.return_code,
                "last_sequence": self.sequence,
            }

    def append_log(self, line: str) -> None:
        clean = line.rstrip("\r\n")
        with self.lock:
            self.sequence += 1
            self.logs.append((self.sequence, clean))

    def log_since(self, after: int) -> list[dict[str, Any]]:
        with self.lock:
            return [
                {"seq": seq, "line": line}
                for seq, line in self.logs
                if seq > after
            ]


class ProcessManager:
    def __init__(self) -> None:
        self.slots = {
            "training": ProcessSlot("training"),
            "runtime": ProcessSlot("runtime"),
        }

    def start(self, name: str, command: list[str], *, env: dict[str, str] | None = None) -> dict:
        slot = self.slots[name]
        with slot.lock:
            if slot.process is not None and slot.process.poll() is None:
                raise RuntimeError(f"{name} is already running")

            slot.status = "starting"
            slot.command = list(command)
            slot.started_at = _utc_now()
            slot.ended_at = None
            slot.return_code = None
            slot.stop_requested = False
            slot.logs.clear()
            slot.sequence = 0
            slot.append_log(f"[ui] starting: {_display_command(command)}")

            kwargs: dict[str, Any] = {
                "cwd": str(HERE),
                "env": env or os.environ.copy(),
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 1,
            }
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True

            try:
                slot.process = subprocess.Popen(command, **kwargs)
            except Exception:
                slot.status = "error"
                slot.ended_at = _utc_now()
                raise
            slot.status = "running"

            threading.Thread(
                target=self._pump_output,
                args=(slot,),
                name=f"bess-ui-{name}-output",
                daemon=True,
            ).start()
            return slot.snapshot()

    def stop(self, name: str) -> dict:
        slot = self.slots[name]
        with slot.lock:
            process = slot.process
            if process is None or process.poll() is not None:
                return slot.snapshot()
            slot.stop_requested = True
            slot.status = "stopping"
            slot.append_log("[ui] stop requested")

        self._terminate_process_tree(process)
        return slot.snapshot()

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return

        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    @staticmethod
    def _pump_output(slot: ProcessSlot) -> None:
        process = slot.process
        if process is None or process.stdout is None:
            return
        return_code = -1
        try:
            for line in process.stdout:
                slot.append_log(line)
            return_code = process.wait()
        except (OSError, ValueError) as exc:
            slot.append_log(f"[ui] output reader failed: {exc}")
            return_code = process.poll() if process.poll() is not None else -1
        finally:
            try:
                process.stdout.close()
            except (OSError, ValueError) as exc:
                slot.append_log(f"[ui] stdout close warning: {exc}")

        with slot.lock:
            slot.return_code = return_code
            slot.ended_at = _utc_now()
            if slot.stop_requested:
                slot.status = "stopped"
            elif return_code == 0:
                slot.status = "done"
            else:
                slot.status = "error"
            slot.append_log(f"[ui] process exited with code {return_code}")


MANAGER = ProcessManager()


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, UPLOADS_DIR, LOGS_DIR, STATE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _save_upload(file: FileStorage | None, *, prefix: str) -> Path | None:
    if file is None or not file.filename:
        return None
    filename = secure_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_UPLOADS:
        raise ValueError(f"unsupported upload type {suffix!r}; use CSV, JSON, or PT")
    target = UPLOADS_DIR / f"{prefix}_{uuid4().hex[:10]}_{filename}"
    file.save(target)
    return target.resolve()


def _path_from_form(form_key: str, upload_key: str, *, prefix: str, required: bool) -> Path | None:
    uploaded = _save_upload(request.files.get(upload_key), prefix=prefix)
    if uploaded is not None:
        return uploaded
    raw = request.form.get(form_key, "").strip()
    if not raw:
        if required:
            raise ValueError(f"{form_key} is required (path or upload)")
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (HERE / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    return path


def _int_field(name: str, default: int, *, minimum: int = 0) -> int:
    raw = request.form.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _tail_text_lines(path: Path, limit: int) -> list[str]:
    """Return the final text lines without reading a potentially huge log into RAM."""
    if not path.is_file() or limit <= 0:
        return []
    block_size = 8192
    chunks: list[bytes] = []
    newline_count = 0
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        while position > 0 and newline_count <= limit:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
    text = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
    return text.splitlines()[-limit:]


def _setpoint_tail(limit: int = 30) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in _tail_text_lines(SETPOINT_LOG, limit):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _artifact_payload() -> dict[str, list[dict[str, Any]]]:
    _ensure_dirs()
    groups = {"policies": [], "evaluations": [], "curves": []}
    for path in RESULTS_DIR.iterdir():
        if not path.is_file():
            continue
        stat = path.stat()
        item = {
            "name": path.name,
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        }
        if path.suffix.lower() == ".pt":
            groups["policies"].append(item)
        elif path.name.startswith("evaluation_") and path.suffix.lower() == ".json":
            groups["evaluations"].append(item)
        elif path.name.startswith("training_curve_") and path.suffix.lower() == ".csv":
            groups["curves"].append(item)
    for items in groups.values():
        items.sort(key=lambda item: item["mtime"], reverse=True)
    return groups


def _latest_curve(max_rows: int = 300) -> dict[str, Any]:
    curves = _artifact_payload()["curves"]
    if not curves:
        return {"name": None, "rows": []}
    path = Path(curves[0]["path"])
    rows: deque[dict[str, str]] = deque(maxlen=max_rows)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(dict(row))
    except OSError:
        return {"name": path.name, "rows": []}
    return {"name": path.name, "rows": list(rows)}


def _error_response(exc: Exception, status: int = 400):
    return jsonify({"ok": False, "error": str(exc)}), status


def create_app() -> Flask:
    _ensure_dirs()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

    @app.get("/")
    def index():
        return render_template("index.html", **index_context())

    @app.get("/api/status")
    def status():
        setpoints = _setpoint_tail(1)
        return jsonify(
            {
                "ok": True,
                "training": MANAGER.slots["training"].snapshot(),
                "runtime": MANAGER.slots["runtime"].snapshot(),
                "artifacts": _artifact_payload(),
                "runtime_state": _json_file(RUNTIME_STATE),
                "latest_setpoint": setpoints[-1] if setpoints else None,
            }
        )

    @app.get("/api/logs/<name>")
    def logs(name: str):
        if name not in MANAGER.slots:
            return _error_response(ValueError("unknown process"), 404)
        try:
            after = max(0, int(request.args.get("after", "0")))
        except ValueError:
            after = 0
        slot = MANAGER.slots[name]
        return jsonify({"ok": True, "lines": slot.log_since(after), "status": slot.snapshot()})

    @app.get("/api/setpoints")
    def setpoints():
        try:
            limit = min(200, max(1, int(request.args.get("limit", "30"))))
        except ValueError:
            limit = 30
        return jsonify({"ok": True, "rows": _setpoint_tail(limit)})

    @app.get("/api/training/curve")
    def training_curve():
        return jsonify({"ok": True, **_latest_curve()})

    @app.post("/api/train/start")
    def train_start():
        try:
            dataset = _path_from_form("dataset_path", "dataset_file", prefix="dataset", required=True)
            config = _path_from_form("config_path", "config_file", prefix="config", required=True)
            steps = _int_field("steps", 1_500_000, minimum=1)
            seed = _int_field("seed", 0, minimum=0)
            tag = request.form.get("tag", "ui_policy").strip() or "ui_policy"
            if not _SAFE_TAG.fullmatch(tag):
                raise ValueError("tag may contain only letters, numbers, dot, underscore, and dash")

            command = [
                sys.executable,
                str(MAIN_PY),
                "train",
                "--csv",
                str(dataset),
                "--config-json",
                str(config),
                "--steps",
                str(steps),
                "--tag",
                tag,
                "--seed",
                str(seed),
            ]
            extra = request.form.get("extra_args", "").strip()
            if extra:
                command.extend(shlex.split(extra, posix=os.name != "nt"))
            snapshot = MANAGER.start("training", command)
            return jsonify({"ok": True, "training": snapshot})
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            return _error_response(exc)

    @app.post("/api/train/stop")
    def train_stop():
        return jsonify({"ok": True, "training": MANAGER.stop("training")})

    @app.post("/api/runtime/start")
    def runtime_start():
        try:
            policy = _path_from_form("policy_path", "policy_file", prefix="policy", required=True)
            config = _path_from_form("config_path", "config_file", prefix="runtime_config", required=False)
            plan = _path_from_form("plan_path", "plan_file", prefix="plan", required=False)
            mode = request.form.get("mode", "shadow").strip().lower()
            if mode not in {"shadow", "closed"}:
                raise ValueError("mode must be shadow or closed")

            interval = _int_field("interval", 15, minimum=1)
            offset = _int_field("offset", 2, minimum=0)
            mqtt_port = _int_field("mqtt_port", 1883, minimum=1)
            mqtt_host = request.form.get("mqtt_host", "localhost").strip() or "localhost"
            timezone_name = request.form.get("timezone", "Asia/Ho_Chi_Minh").strip() or "Asia/Ho_Chi_Minh"
            controller_url = request.form.get("controller_url", "http://localhost:8001").strip()
            api_key = request.form.get("api_key", "dev-api-key-change-me")

            command = [
                sys.executable,
                str(MAIN_PY),
                "run",
                "--policy",
                str(policy),
                "--mode",
                mode,
                "--timezone",
                timezone_name,
                "--interval",
                str(interval),
                "--offset",
                str(offset),
                "--mqtt-host",
                mqtt_host,
                "--mqtt-port",
                str(mqtt_port),
                "--controller-url",
                controller_url,
                "--api-key",
                api_key,
            ]
            if config is not None:
                command.extend(["--config", str(config)])
            if plan is not None:
                command.extend(["--plan", str(plan)])
            mqtt_username = request.form.get("mqtt_username", "")
            mqtt_password = request.form.get("mqtt_password", "")
            if mqtt_username:
                command.extend(["--mqtt-username", mqtt_username])
            if mqtt_password:
                command.extend(["--mqtt-password", mqtt_password])

            snapshot = MANAGER.start("runtime", command)
            return jsonify({"ok": True, "runtime": snapshot})
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            return _error_response(exc)

    @app.post("/api/runtime/stop")
    def runtime_stop():
        return jsonify({"ok": True, "runtime": MANAGER.stop("runtime")})

    # The browser source is the literal original Sizing Demo index.html. These
    # adapters satisfy its historical route contract with files + subprocesses,
    # while the older compact /api/train/* and /api/runtime/* endpoints remain
    # available for scripts that already use them.
    register_og_routes(app, MANAGER)
    register_og_runtime_routes(app)
    return app


def run_ui(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Flask UI for bess-drl-debloated")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    app = create_app()
    # Disable the reloader: it would create a second process manager and duplicate
    # training/runtime launches. Debug tracebacks can still be enabled.
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_ui())
