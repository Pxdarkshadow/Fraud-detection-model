"""Drift detection via Population Stability Index (PSI).

PSI measures how much a feature's distribution has shifted between a reference
and a new sample.  Per-feature PSI is reported, plus an overall boolean flag
when any feature exceeds ``DRIFT_THRESHOLD``.

Method: for each numeric feature, both reference and sample are binned into
the same quantile-based bins (derived from the reference).  PSI is then
    sum((p_i - q_i) * ln(p_i / q_i))
where p = reference proportions, q = sample proportions.  Standard bands:
    PSI < 0.1  -> no significant drift
    0.1 <= PSI < 0.2 -> moderate drift
    PSI >= 0.2 -> significant drift

So ``DRIFT_THRESHOLD = 0.2`` flags "significant drift".
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from .config import NUMERIC_FEATURES, DATA_PATH

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_BINS = 10  # quantile bins for PSI
DRIFT_THRESHOLD = 0.2  # per-feature PSI above this = "drifted"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def _psi(reference: np.ndarray, sample: np.ndarray, bins: np.ndarray) -> float:
    """Compute PSI for one feature given shared bin edges."""
    eps = 1e-4  # floor to avoid div-by-zero / log(0)
    p = np.histogram(reference, bins=bins)[0].astype(float) / len(reference) + eps
    q = np.histogram(sample, bins=bins)[0].astype(float) / len(sample) + eps
    return float(np.sum((p - q) * np.log(p / q)))


def check_drift(
    reference: pd.DataFrame,
    sample: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = DRIFT_THRESHOLD,
) -> dict:
    """Compare *sample* distributions to *reference*.

    Returns a dict with:
        - per_feature: {feature: psi_value, ...}
        - drifted_features: [feature, ...]  (only those >= threshold)
        - overall_drift: bool (any drifted)
    """
    # ponytail: quantile PSI only on numeric cols; categoricals need a separate chi2 PSI.
    features = features or NUMERIC_FEATURES
    # align on shared columns
    features = [f for f in features if f in reference.columns and f in sample.columns]

    per_feature: dict[str, float] = {}
    for col in features:
        ref = reference[col].dropna().values
        sam = sample[col].dropna().values
        # quantile bins from reference
        bins = np.quantile(ref, np.linspace(0, 1, N_BINS + 1))
        bins = np.unique(bins)  # collapse duplicate edges
        if len(bins) < 3:
            per_feature[col] = 0.0
            continue
        per_feature[col] = round(_psi(ref, sam, bins), 6)

    drifted = [f for f, v in per_feature.items() if v >= threshold]
    return {
        "per_feature": per_feature,
        "drifted_features": drifted,
        "overall_drift": len(drifted) > 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Drift check (PSI) for numeric fraud features")
    ap.add_argument("--reference", default=str(DATA_PATH), help="Reference CSV (default: training data)")
    ap.add_argument("--sample", required=True, help="New-sample CSV to compare against reference")
    ap.add_argument("--threshold", type=float, default=DRIFT_THRESHOLD, help="Per-feature PSI drift threshold")
    args = ap.parse_args()

    ref = pd.read_csv(args.reference)
    sam = pd.read_csv(args.sample)

    result = check_drift(ref, sam, threshold=args.threshold)

    # pretty-print
    print(json.dumps(result, indent=2))

    # top-5 by PSI
    ranked = sorted(result["per_feature"].items(), key=lambda kv: kv[1], reverse=True)
    print("\nTop drifted features (by PSI):")
    for feat, psi in ranked[:5]:
        marker = " <-- DRIFTED" if psi >= args.threshold else ""
        print(f"  {feat:>8s}: {psi:.4f}{marker}")
    print(f"\nOverall drift: {result['overall_drift']}")


if __name__ == "__main__":
    main()
