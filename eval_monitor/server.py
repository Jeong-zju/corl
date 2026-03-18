from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from repository import EvalMonitorStore, ensure_relative_to


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_ROOT = REPO_ROOT / "main" / "outputs" / "eval"
STATIC_ROOT = Path(__file__).resolve().parent / "static"


class MonitorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        store: EvalMonitorStore,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.store = store
        self.static_root = STATIC_ROOT.resolve()
        self.eval_root = store.eval_root.resolve()


class MonitorRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: MonitorHTTPServer

    def do_GET(self) -> None:
        self._dispatch_get(send_body=True)

    def do_HEAD(self) -> None:
        self._dispatch_get(send_body=False)

    def _dispatch_get(self, send_body: bool) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/":
            self._serve_static("index.html", send_body=send_body)
            return
        if parsed.path.startswith("/static/"):
            self._serve_static(parsed.path.removeprefix("/static/"), send_body=send_body)
            return
        if parsed.path.startswith("/media/"):
            self._serve_media(parsed.path.removeprefix("/media/"), send_body=send_body)
            return
        if parsed.path == "/api/data":
            self._send_json(HTTPStatus.OK, self.server.store.get_snapshot(), send_body=send_body)
            return
        if parsed.path == "/api/run":
            run_id = self._query_value(parsed.query, "id")
            if not run_id:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "Missing run id", send_body=send_body)
                return
            detail = self.server.store.get_run_detail(run_id)
            if detail is None:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"Run not found: {run_id}", send_body=send_body)
                return
            self._send_json(HTTPStatus.OK, detail, send_body=send_body)
            return
        if parsed.path == "/api/stream":
            if not send_body:
                self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
                self.send_header("Allow", "GET")
                self.end_headers()
                return
            self._handle_stream()
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown path: {parsed.path}", send_body=send_body)

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return

        if parsed.path == "/api/delete-run":
            run_id = payload.get("id")
            if not isinstance(run_id, str) or not run_id:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "Missing run id")
                return
            try:
                deleted = self.server.store.delete_run(run_id)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            if not deleted:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"Run not found: {run_id}")
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "deleted": run_id})
            return

        if parsed.path == "/api/delete-series":
            series_id = payload.get("id")
            if not isinstance(series_id, str) or not series_id:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "Missing series id")
                return
            try:
                deleted = self.server.store.delete_series(series_id)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            if not deleted:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"Series not found: {series_id}")
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "deleted": series_id})
            return

        if parsed.path == "/api/refresh":
            self.server.store.refresh(force=True)
            self._send_json(HTTPStatus.OK, {"ok": True, "snapshot_version": self.server.store.get_snapshot()["version"]})
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown path: {parsed.path}")

    def _handle_stream(self) -> None:
        snapshot = self.server.store.get_snapshot()
        listener = self.server.store.subscribe()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            self._write_sse("snapshot", snapshot)
            while True:
                try:
                    event = listener.get(timeout=15.0)
                    self._write_sse("update", event)
                except Empty:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return
        finally:
            self.server.store.unsubscribe(listener)

    def _write_sse(self, event_name: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        message = f"event: {event_name}\ndata: {data}\n\n".encode("utf-8")
        self.wfile.write(message)
        self.wfile.flush()

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any], send_body: bool = True) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _send_error_json(self, status: HTTPStatus, message: str, send_body: bool = True) -> None:
        self._send_json(status, {"ok": False, "error": message}, send_body=send_body)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("JSON body must be an object")
        return parsed

    def _query_value(self, query: str, key: str) -> str | None:
        values = parse_qs(query).get(key)
        if not values:
            return None
        return values[0]

    def _serve_static(self, relative_path: str, send_body: bool) -> None:
        safe_relative = relative_path.strip("/") or "index.html"
        target = (self.server.static_root / safe_relative).resolve()
        try:
            ensure_relative_to(self.server.static_root, target)
        except ValueError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid static path", send_body=send_body)
            return
        if not target.exists() or not target.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Static file not found: {relative_path}", send_body=send_body)
            return
        self._serve_file(target, allow_ranges=False, cache_control="no-store", send_body=send_body)

    def _serve_media(self, relative_path: str, send_body: bool) -> None:
        decoded = unquote(relative_path).lstrip("/")
        target = (self.server.eval_root / decoded).resolve()
        try:
            ensure_relative_to(self.server.eval_root, target)
        except ValueError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Invalid media path", send_body=send_body)
            return
        if not target.exists() or not target.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Media file not found: {decoded}", send_body=send_body)
            return
        self._serve_file(target, allow_ranges=True, cache_control="no-store", send_body=send_body)

    def _serve_file(self, target: Path, allow_ranges: bool, cache_control: str, send_body: bool) -> None:
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        file_size = target.stat().st_size
        range_header = self.headers.get("Range") if allow_ranges else None

        start = 0
        end = file_size - 1
        status = HTTPStatus.OK

        if range_header and range_header.startswith("bytes="):
            requested = range_header.removeprefix("bytes=").split(",", 1)[0].strip()
            start_text, _, end_text = requested.partition("-")
            if start_text:
                start = int(start_text)
            if end_text:
                end = int(end_text)
            if start_text == "" and end_text:
                suffix_size = int(end_text)
                start = max(file_size - suffix_size, 0)
                end = file_size - 1
            start = max(start, 0)
            end = min(end, file_size - 1)
            if start > end or start >= file_size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT

        content_length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache_control)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        if not send_body:
            return

        with target.open("rb") as handle:
            handle.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = handle.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, fmt: str, *args: Any) -> None:
        client = self.client_address[0]
        message = fmt % args
        print(f"[eval-monitor] {client} {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live dashboard for main/outputs/eval.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind. Default: 8765")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=DEFAULT_EVAL_ROOT,
        help=f"Eval output root. Default: {DEFAULT_EVAL_ROOT}",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Filesystem polling interval in seconds. Default: 2.0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.expanduser()
    if not eval_root.is_absolute():
        eval_root = (REPO_ROOT / eval_root).resolve()

    os.chdir(REPO_ROOT)
    store = EvalMonitorStore(eval_root=eval_root, poll_interval_seconds=args.poll_interval)
    store.start()

    server = MonitorHTTPServer((args.host, args.port), MonitorRequestHandler, store)
    print(f"Eval monitor listening on http://{args.host}:{args.port}")
    print(f"Watching eval root: {eval_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.stop()


if __name__ == "__main__":
    main()
