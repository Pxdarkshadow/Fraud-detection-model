#!/usr/bin/env python
"""Run the fraud detection training pipeline.

Usage:
    .venv\Scripts\python.exe scripts/run_pipeline.py
"""

import sys
import os

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fraud_detection.pipeline import FraudPipeline

if __name__ == "__main__":
    FraudPipeline().run()
