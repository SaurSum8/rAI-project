"""Local live-camera dashboard and persistent sorting history."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import logging
import math
from pathlib import Path
import sqlite3
from threading import Condition, RLock, Thread
import time
from typing import Iterator, Optional
from urllib.parse import urlsplit

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from werkzeug.serving import make_server, WSGIRequestHandler


LOG = logging.getLogger("camera_sorter.dashboard")
MAX_NOTE_LENGTH = 2000
PAGE = Path(__file__).resolve().parent / "web" / "dashboard.html"


class DashboardState:
    """Share one camera's frames, next-item note, and history across threads."""

    def __init__(self, history_file: Path):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._frames_changed = Condition(self._lock)
        self._pending_note = ""
        self._latest_result: Optional[dict] = None
        self._status = "Waiting for camera"
        self._camera_online = False
        self._recalibration_requested = False
        self._jpeg: Optional[bytes] = None
        self._frame_number = 0
        self._last_frame_time = -math.inf
        self._closed = False
        with self._database() as database:
            database.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL UNIQUE,
                    record TEXT NOT NULL
                )
            """)
            database.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    name TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                )
            """)
            database.execute("INSERT OR IGNORE INTO metadata VALUES ('epoch', 0)")
            self._epoch = database.execute("SELECT value FROM metadata WHERE name = 'epoch'").fetchone()[0]

    @contextmanager
    def _database(self):
        database = None
        try:
            database = sqlite3.connect(str(self.history_file), timeout=5)
            with database:
                yield database
        except sqlite3.Error as exc:
            raise RuntimeError(f"Cannot access sorting history: {exc}") from exc
        finally:
            if database is not None:
                database.close()

    def get_snapshot(self) -> dict:
        with self._lock:
            with self._database() as database:
                rows = database.execute("SELECT record FROM history ORDER BY sequence DESC").fetchall()
            items = [json.loads(row[0]) for row in rows]
            counts = {
                "trash": sum(item["decision"] == "trash" for item in items),
                "recycle": sum(item["decision"] == "recycle" for item in items),
                "total": len(items),
            }
            return {
                "counts": counts,
                "items": items,
                "pending_note": self._pending_note,
                "latest_result": deepcopy(self._latest_result),
                "status": self._status,
                "camera_online": self._camera_online,
            }

    def set_next_note(self, note: str) -> None:
        if not isinstance(note, str):
            raise ValueError("The next-item note must be text.")
        if len(note) > MAX_NOTE_LENGTH:
            raise ValueError(f"The next-item note must be {MAX_NOTE_LENGTH} characters or fewer.")
        with self._lock:
            self._pending_note = note.strip()

    def claim_next_item(self) -> tuple[str, int]:
        """Consume the note once, recording which reset epoch owns the item."""
        with self._lock:
            note = self._pending_note
            self._pending_note = ""
            return note, self._epoch

    def record_result(self, record: dict, epoch: int) -> None:
        with self._lock:
            if epoch != self._epoch:
                return
            stored = deepcopy(record)
            if stored.get("decision") in ("trash", "recycle") and not stored.get("dry_run", False):
                item_id = stored.get("item_id") or stored.get("capture_id")
                if not isinstance(item_id, str) or not item_id:
                    raise ValueError("A sorted item needs an item_id or capture_id.")
                encoded = json.dumps(stored, ensure_ascii=False, allow_nan=False)
                with self._database() as database:
                    database.execute(
                        """INSERT INTO history (item_id, record) VALUES (?, ?)
                           ON CONFLICT(item_id) DO UPDATE SET record = excluded.record""",
                        (item_id, encoded),
                    )
            self._latest_result = stored

    def reset_history(self) -> None:
        with self._lock:
            next_epoch = self._epoch + 1
            with self._database() as database:
                database.execute("DELETE FROM history")
                database.execute("UPDATE metadata SET value = ? WHERE name = 'epoch'", (next_epoch,))
            self._epoch = next_epoch
            self._latest_result = None

    def request_recalibration(self) -> None:
        with self._lock:
            self._recalibration_requested = True

    def consume_recalibration(self) -> bool:
        with self._lock:
            requested = self._recalibration_requested
            self._recalibration_requested = False
            return requested

    def set_status(self, status: str, online: bool) -> None:
        with self._lock:
            self._status = status
            self._camera_online = online

    def publish_frame(self, frame: np.ndarray, status: str) -> None:
        now = time.monotonic()
        with self._frames_changed:
            if self._closed:
                return
            self._status = status
            self._camera_online = True
            if now - self._last_frame_time < 1 / 15:
                return
            height, width = frame.shape[:2]
            if width > 960:
                frame = cv2.resize(frame, (960, max(1, round(height * 960 / width))), interpolation=cv2.INTER_AREA)
            success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                raise RuntimeError("Could not encode the live camera frame.")
            self._jpeg = encoded.tobytes()
            self._last_frame_time = now
            self._frame_number += 1
            self._frames_changed.notify_all()

    def frames(self) -> Iterator[bytes]:
        """Yield each new JPEG once per viewer; wake immediately on shutdown."""
        last_frame = -1
        while True:
            with self._frames_changed:
                self._frames_changed.wait_for(
                    lambda: self._closed or (self._jpeg is not None and self._frame_number != last_frame)
                )
                if self._closed:
                    return
                jpeg = self._jpeg
                last_frame = self._frame_number
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n")

    def close(self) -> None:
        with self._frames_changed:
            self._closed = True
            self._camera_online = False
            self._frames_changed.notify_all()


def create_app(state: DashboardState) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

    @app.before_request
    def local_requests_only():
        try:
            host = urlsplit("http://" + request.host).hostname
        except ValueError:
            host = None
        if host not in ("localhost", "127.0.0.1", "::1"):
            return jsonify(error="Use the localhost dashboard address."), 403
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin:
                try:
                    parsed_origin = urlsplit(origin)
                    same_origin = (parsed_origin.scheme in ("http", "https")
                                   and parsed_origin.netloc.lower() == request.host.lower())
                except ValueError:
                    same_origin = False
                if not same_origin:
                    return jsonify(error="Cross-origin dashboard changes are not allowed."), 403
            if not request.is_json:
                return jsonify(error="Send a JSON request."), 415
        return None

    @app.after_request
    def disable_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return send_file(PAGE, mimetype="text/html")

    @app.get("/video_feed")
    def video_feed():
        return Response(state.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/state")
    def snapshot():
        return jsonify(state.get_snapshot())

    @app.post("/api/next-item")
    def next_item():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "note" not in payload:
            return jsonify(error="Provide a JSON object with a note field."), 400
        try:
            state.set_next_note(payload["note"])
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(state.get_snapshot())

    @app.post("/api/reset")
    def reset():
        state.reset_history()
        return jsonify(state.get_snapshot())

    @app.post("/api/recalibrate")
    def recalibrate():
        state.request_recalibration()
        return jsonify(state.get_snapshot())

    return app


class _QuietRequestHandler(WSGIRequestHandler):
    def log(self, kind: str, message: str, *args) -> None:
        LOG.debug("%s: %s", kind, message % args)


class DashboardServer:
    def __init__(self, state: DashboardState, port: int = 8000):
        if not 0 <= port <= 65535:
            raise ValueError("Dashboard port must be between 0 and 65535.")
        self.state = state
        self.port = port
        self._server = None
        self._thread: Optional[Thread] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        if self._server is not None:
            return
        try:
            self._server = make_server(
                "127.0.0.1", self.port, create_app(self.state), threaded=True,
                request_handler=_QuietRequestHandler,
            )
        except (OSError, SystemExit) as exc:
            raise OSError(f"Cannot start dashboard on 127.0.0.1:{self.port}; try another --port.") from exc
        self.port = self._server.server_port
        self._thread = Thread(target=self._server.serve_forever, name="sorter-dashboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.state.close()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
