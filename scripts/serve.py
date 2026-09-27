"""Entrypoint for fraud scoring API server.

Usage:
    python scripts/serve.py
    python scripts/serve.py --port 8080 --model models/fraud_model.txt
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fraud_detection.api import run, init


def main():
    p = argparse.ArgumentParser(description="Fraud scoring HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default=None, help="Path to model file")
    p.add_argument("--threshold", type=float, default=None, help="Fraud threshold override")
    args = p.parse_args()
    init(model_path=args.model, threshold=args.threshold)
    run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
