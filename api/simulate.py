"""
Single stateless API endpoint for AERO-Lite.

POST /api/simulate
  body: {"tick": int, "injectedFailures": ["s3-s4", ...], "predictiveFailoverEnabled": true}
  -> full topology + per-link metrics + ML predictions + stats for that tick

Helper modules live in api/_lib/ (underscore-prefixed folders are not turned
into separate serverless functions by Vercel). Only this file is an endpoint.
"""
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_lib"))

IMPORT_ERROR = None
try:
    from forest_predict import predict as rf_predict
    from simulator import build_state
except Exception:  # surfaced as JSON instead of an opaque 500
    IMPORT_ERROR = traceback.format_exc()


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
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
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
        if IMPORT_ERROR:
            self._send_json({"error": "import failed", "detail": IMPORT_ERROR}, status=500)
            return
        try:
            tick = int(body.get("tick", 0))
            injected = set(body.get("injectedFailures", []) or [])
            failover_enabled = bool(body.get("predictiveFailoverEnabled", True))
            state = build_state(tick, injected, failover_enabled, rf_predict)
            self._send_json(state)
        except Exception:
            self._send_json({"error": "simulation failed", "detail": traceback.format_exc()}, status=500)
