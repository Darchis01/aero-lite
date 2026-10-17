"""
Single stateless API endpoint for AERO-Lite.

POST /api/simulate
  body: {"tick": int, "injectedFailures": ["s3-s4", ...], "predictiveFailoverEnabled": true}
  -> full topology + per-link metrics + ML predictions + stats for that tick

Vercel's Python runtime (@vercel/python) natively supports a BaseHTTPRequestHandler
subclass named `handler` in any file under /api -- no Flask or other framework
needed, which keeps this function's cold start fast and its dependency list
(stdlib only) at zero.
"""
import json
from http.server import BaseHTTPRequestHandler

from forest_predict import predict as rf_predict
from simulator import build_state


class handler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        # Convenience: a plain GET returns tick=0 with no injected failures,
        # useful for a first paint before the frontend starts polling.
        self._handle({"tick": 0, "injectedFailures": [], "predictiveFailoverEnabled": True})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw or b"{}")
        except Exception:
            body = {}
        self._handle(body)

    def _handle(self, body):
        try:
            tick = int(body.get("tick", 0))
            injected = set(body.get("injectedFailures", []) or [])
            failover_enabled = bool(body.get("predictiveFailoverEnabled", True))

            state = build_state(tick, injected, failover_enabled, rf_predict)
            self._send_json(state)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
