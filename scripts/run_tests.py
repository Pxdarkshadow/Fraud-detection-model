#!/usr/bin/env python
"""Run the test suite.

Usage:
    .venv\Scripts\python.exe scripts/run_tests.py
"""

import sys
import os

# Add src/ to path so tests can import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Run test file
test_path = os.path.join(os.path.dirname(__file__), "..", "tests", "test_predict.py")
exec(compile(open(test_path).read(), test_path, "exec"))
