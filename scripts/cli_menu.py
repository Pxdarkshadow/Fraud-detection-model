#!/usr/bin/env python
"""Interactive CLI menu for the fraud detection project.

Usage:
    python scripts/cli_menu.py
"""

from __future__ import annotations

import os
import sys
import time
import math
import argparse
from pathlib import Path

import psutil

# Add src/ to path so we can import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fraud_detection.config import FEATURES, INPUT_COLS, TARGET  # type: ignore
from fraud_detection.pipeline import FraudPipeline, load_data, chrono_split, evaluate, benchmark_latency  # type: ignore
from fraud_detection.predict import (
    validate_input,
    load_model,
    default_threshold,
)  # type: ignore


def _is_finite_number(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def ask_float(label: str, *, allow_negative: bool = True) -> float:
    while True:
        raw = input(f"{label}: ").strip()
        try:
            v = float(raw)
        except ValueError:
            print("Invalid number. Try again.")
            continue

        if not _is_finite_number(v):
            print("Number must be finite. Try again.")
            continue
        if not allow_negative and v < 0:
            print("Value must be >= 0. Try again.")
            continue
        return v


def get_cpu_and_ram_metrics(*, wall_start: float, cpu_start: float) -> tuple[float, float]:
    proc = psutil.Process(os.getpid())
    wall_end = time.perf_counter()
    cpu_end = proc.cpu_times().user + proc.cpu_times().system

    wall = wall_end - wall_start
    cpu_pct = 0.0
    if wall > 0:
        cpu_pct = (cpu_end - cpu_start) / wall * 100.0

    ram_mb = proc.memory_info().rss / (1024.0 * 1024.0)
    return cpu_pct, ram_mb


def get_nvidia_gpu_metrics() -> tuple[float | None, float | None]:
    """Return (gpu_util_percent, used_memory_mb) for GPU 0 if available."""
    try:
        from pynvml import (
            nvmlInit,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetUtilizationRates,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetCount,
            nvmlShutdown,
        )
    except Exception:
        return None, None

    try:
        nvmlInit()
        try:
            count = nvmlDeviceGetCount()
            if count <= 0:
                return None, None

            h = nvmlDeviceGetHandleByIndex(0)
            util = nvmlDeviceGetUtilizationRates(h)
            mem = nvmlDeviceGetMemoryInfo(h)
            return float(util.gpu), float(mem.used) / (1024.0 * 1024.0)
        finally:
            nvmlShutdown()
    except Exception:
        return None, None


def risk_level_from_prob(p: float) -> str:
    # Simple 3-band risk scale (tuned for human readability, not calibrated policy).
    if p >= 0.65:
        return "High"
    if p >= 0.45:
        return "Medium"
    return "Low"


def top_features_from_shap(model, df_row, feature_cols: list[str]) -> tuple[list[str], float, bool]:
    """Returns (top_features, explanation_ms, shap_used). If SHAP unavailable, returns ([], 0.0, False)."""
    try:
        import shap  # type: ignore
    except Exception:
        return [], 0.0, False

    expl_start = time.perf_counter()
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_row[feature_cols])

        # shap.TreeExplainer may return (n, m) or list[(n, m)] depending on model.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # fraud class

        vals = shap_values[0]
        order = sorted(range(len(feature_cols)), key=lambda i: abs(float(vals[i])), reverse=True)
        top_n = order[:5]
        top = [f"{feature_cols[i]} ({float(vals[i]):+.4f})" for i in top_n]
        return top, (time.perf_counter() - expl_start) * 1000.0, True
    except Exception:
        # If SHAP fails for any reason, don't break the prediction flow.
        return [], (time.perf_counter() - expl_start) * 1000.0, True


def option_train() -> None:
    FraudPipeline().run()


def option_evaluate() -> None:
    import pandas as pd

    model = load_model()
    threshold = default_threshold()

    df = load_data()
    train, val, test = chrono_split(df)
    _ = train  # unused: evaluation is model/threshold only

    print("\nEvaluating on validation and test using models/threshold.txt")
    evaluate(model, val[FEATURES], val[TARGET], threshold, "Validation")
    evaluate(model, test[FEATURES], test[TARGET], threshold, "Test")


def option_benchmark() -> None:
    df = load_data()
    _, _, test = chrono_split(df)

    model = load_model()
    benchmark_latency(model, test[FEATURES])


def option_predict_custom() -> None:
    import pandas as pd

    print("\nEnter values for the required features.")
    print("(All inputs are used as-is; invalid numbers are rejected.)\n")

    row: dict[str, float] = {}
    for col in INPUT_COLS:
        allow_negative = True
        if col == "Time":
            allow_negative = False
        if col == "Amount":
            allow_negative = False
        row[col] = ask_float(col, allow_negative=allow_negative)

    df = pd.DataFrame([row])
    validate_input(df)

    threshold = default_threshold()
    model = load_model()

    cpu_proc = psutil.Process(os.getpid())
    cpu_start = cpu_proc.cpu_times().user + cpu_proc.cpu_times().system
    wall_start = time.perf_counter()

    infer_start = time.perf_counter()
    probs = model.predict(df[INPUT_COLS])
    prob = float(probs[0])
    pred = int(prob >= threshold)
    infer_ms = (time.perf_counter() - infer_start) * 1000.0

    shap_top, shap_ms, shap_used = top_features_from_shap(model, df, INPUT_COLS)

    total_ms = (time.perf_counter() - wall_start) * 1000.0
    cpu_pct, ram_mb = get_cpu_and_ram_metrics(wall_start=wall_start, cpu_start=cpu_start)
    gpu_util, gpu_mem_mb = get_nvidia_gpu_metrics()

    risk = risk_level_from_prob(prob)
    label = "Fraud" if pred == 1 else "Legitimate"

    print("\n" + "=" * 60)
    print("PREDICTION")
    print("=" * 60)
    print(f"Fraud/Legitimate  : {label}")
    print(f"Fraud probability  : {prob:.6f}")
    print(f"Risk level        : {risk}")

    if shap_top:
        print("\nTop contributing features (SHAP):")
        for f in shap_top:
            print(f"  - {f}")
        print(f"Explanation time  : {shap_ms:.2f} ms")
    else:
        print("\nTop contributing features (SHAP):")
        if shap_used:
            print("  - SHAP explanation failed")
        else:
            print("  - SHAP not available (install shap)")
        print("Explanation time  : 0.00 ms")

    print("\nPerformance")
    print(f"Inference time    : {infer_ms:.2f} ms")
    print(f"Total time        : {total_ms:.2f} ms")
    print(f"CPU usage         : {cpu_pct:.2f}%")
    print(f"RAM usage         : {ram_mb:.1f} MB")

    if gpu_util is None or gpu_mem_mb is None:
        print("GPU usage         : N/A (pynvml not installed or no NVIDIA GPU)")
    else:
        print(f"GPU usage         : {gpu_util:.0f}%")
        print(f"GPU memory (used) : {gpu_mem_mb:.1f} MB")


def main() -> None:
    p = argparse.ArgumentParser(description="Fraud detection interactive CLI")
    p.add_argument("--no-loop", action="store_true", help="Run once then exit (useful for testing)")
    args = p.parse_args()

    while True:
        print("\n" + "-" * 35)
        print("Fraud Detection Menu")
        print("-" * 35)
        print("1. Train model")
        print("2. Evaluate model")
        print("3. Predict custom transaction (manual entry)")
        print("4. Benchmark")
        print("5. Exit")

        choice = input("Select an option (1-5): ").strip()
        if choice == "1":
            option_train()
        elif choice == "2":
            option_evaluate()
        elif choice == "3":
            option_predict_custom()
        elif choice == "4":
            option_benchmark()
        elif choice == "5":
            print("Bye.")
            return
        else:
            print("Invalid selection. Please enter 1-5.")

        if args.no_loop:
            return


if __name__ == "__main__":
    main()
