"""Local page for correcting Plus AI drafts. Binds to localhost only.

The working sheet is annotations/plus_sheet.jsonl. Gold stays false
until a save says so. Restarting the server keeps those saves and
refreshes unlabeled rows from the starter file.

    python -m src.locomo_eval.interaction_patterns.plus_label_server
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .plus_label_sheet import load_or_build_sheet, update_row, write_jsonl

ROOT = Path(__file__).resolve().parents[3]
PAGE = Path(__file__).with_name("plus_label_dashboard.html")
STARTER = ROOT / "annotations" / "plus_starter.jsonl"
SHEET = ROOT / "annotations" / "plus_sheet.jsonl"


def make_handler(rows: list[dict]):
    """Closure so the handler can write the same list the process loaded."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/sheet":
                payload = json.dumps(rows).encode("utf-8")
                self._send(200, payload, "application/json; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/item":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length") or "0")
            try:
                fields = json.loads(self.rfile.read(length).decode("utf-8"))
                saved = update_row(rows, str(fields.get("record_id") or ""), fields)
            except (json.JSONDecodeError, KeyError) as exc:
                message = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send(400, message, "application/json; charset=utf-8")
                return
            write_jsonl(SHEET, rows)
            self._send(200, json.dumps(saved).encode("utf-8"), "application/json; charset=utf-8")

    return Handler


def main() -> None:
    rows = load_or_build_sheet(STARTER, SHEET)
    drafted = sum(1 for row in rows if row.get("label_status") == "ai_draft")
    server = ThreadingHTTPServer(("127.0.0.1", 8765), make_handler(rows))
    print(f"Label sheet: http://127.0.0.1:8765  ({drafted} AI drafts, {len(rows)} rows)", flush=True)
    print(f"Saves go to {SHEET}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
