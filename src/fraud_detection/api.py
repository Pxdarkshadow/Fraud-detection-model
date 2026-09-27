"""Minimal HTTP server for fraud scoring (stdlib only)."""

import json
import traceback

import pandas as pd
from http.server import HTTPServer, BaseHTTPRequestHandler

from .predict import predict, load_model, default_threshold, validate_input
from .config import INPUT_COLS

# ponytail: global state for model/threshold. Per-process, no concurrent access concern.

_model = None
_threshold = None


def init(model_path=None, threshold=None):
    global _model, _threshold
    from .config import MODEL_PATH
    _model = load_model(model_path or MODEL_PATH)
    _threshold = threshold if threshold is not None else default_threshold()


def _parse_body(body):
    data = json.loads(body)
    if isinstance(data, dict) and "transactions" in data:
        data = data["transactions"]
    if isinstance(data, list):
        return pd.DataFrame(data)
    raise ValueError("Request body must be a JSON array of transaction objects")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # ponytail: silence default stderr logging

    def _send(self, code, obj, content_type="application/json"):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/predict":
            self._send(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            df = _parse_body(body)
            validate_input(df)
        except (json.JSONDecodeError, ValueError) as e:
            self._send(400, {"error": str(e)})
            return
        except Exception as e:
            self._send(400, {"error": str(e)})
            return
        try:
            result = predict(df, threshold=_threshold, model=_model)
            records = result[INPUT_COLS + ["fraud_probability", "prediction"]].to_dict(orient="records")
            self._send(200, {"results": records, "count": len(records)})
        except Exception as e:
            self._send(500, {"error": f"Prediction failed: {e}"})

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "model_loaded": _model is not None})
            return
        self._send(404, {"error": "Not found"})

    def do_PUT(self):
        self._send(405, {"error": "Method not allowed"})

    def do_DELETE(self):
        self._send(405, {"error": "Method not allowed"})

    def do_PATCH(self):
        self._send(405, {"error": "Method not allowed"})


def run(host="127.0.0.1", port=8000):
    server = HTTPServer((host, port), Handler)
    print(f"Serving fraud API on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
