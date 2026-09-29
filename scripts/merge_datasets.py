"""Merge Sparkov, BankSim, and FDH Handbook datasets into one clean CSV.

Output: data/fraud_merged.csv
Schema (all columns present, NaN where source lacks data):
  transaction_id       str
  tx_datetime          str   (ISO-8601, UTC inferred)
  amount               float
  category             str   (normalized category slug)
  merchant             str
  gender               str   (M/F/U)
  age_bucket           str   (0-10,10-20,...,70+ or U)
  city                 str
  state                str
  lat                  float
  long                 float
  merch_lat            float
  merch_long           float
  channel              str   (pos / online / unknown)
  hour                 float (0-23.99)
  day_of_week          int   (0=Mon)
  is_weekend           int
  source               str   (sparkov / banksim / handbook)
  is_fraud             int8

Rebalancing target: 35% fraud / 65% legit
  - SMOTE-free: random undersample legit + plausible oversample fraud
    via Gaussian noise on numeric cols (bounded to realistic ranges).
  - ponytail: simple oversampling with bounded jitter; upgrade to SMOTE
    via imbalanced-learn if distribution quality proves insufficient.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
import random

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "fraud_merged.csv"

RANDOM_SEED = 42
TARGET_FRAUD_RATE = 0.35  # 35% fraud / 65% legit

# ---------------------------------------------------------------------------
# Category normalisation maps
# ---------------------------------------------------------------------------
BANKSIM_CAT_MAP = {
    "es_transportation": "gas_transport",
    "es_food": "food_dining",
    "es_barsandrestaurants": "food_dining",
    "es_health": "health_fitness",
    "es_wellnessandbeauty": "personal_care",
    "es_hyper": "grocery_pos",
    "es_fashion": "shopping_pos",
    "es_sportsandtoys": "kids_pets",
    "es_home": "home",
    "es_hotelservices": "travel",
    "es_leisure": "entertainment",
    "es_travel": "travel",
    "es_tech": "misc_pos",
    "es_contents": "misc_net",
    "es_otherservices": "misc_pos",
}

# BankSim age: '0'=<18,'1'=18-30,'2'=30-40,'3'=40-50,'4'=50-60,'5'=60-70,'6'=70+,'U'=unknown
BANKSIM_AGE_MAP = {
    "'0'": "<18", "'1'": "18-30", "'2'": "30-40",
    "'3'": "40-50", "'4'": "50-60", "'5'": "60-70", "'6'": "70+",
    "'U'": "U",
}
BANKSIM_GENDER_MAP = {
    "'M'": "M", "'F'": "F", "'E'": "U", "'U'": "U",
}


# ---------------------------------------------------------------------------
# Pickle loader (FDH handbook uses old pandas Int64Index)
# ---------------------------------------------------------------------------
class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name in ("Int64Index", "Float64Index", "UInt64Index"):
            return pd.Index
        if "pandas.core.indexes.numeric" in module:
            return pd.Index
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            if "indexes" in module:
                return pd.Index
            raise


def _load_pickle(path: Path) -> pd.DataFrame:
    with open(path, "rb") as fh:
        return _CompatUnpickler(fh).load()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_sparkov() -> pd.DataFrame:
    chunks = []
    for fname in ("fraudTrain.csv", "fraudTest.csv"):
        p = DATA / fname
        if not p.exists():
            print(f"[warn] {fname} not found — skipping", file=sys.stderr)
            continue
        df = pd.read_csv(p, usecols=[
            "trans_date_trans_time", "merchant", "category", "amt",
            "gender", "city", "state", "lat", "long",
            "merch_lat", "merch_long", "unix_time", "is_fraud",
        ])
        out = pd.DataFrame()
        out["transaction_id"] = "sp_" + df.index.astype(str) + "_" + fname[5:8]
        out["tx_datetime"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        out["amount"] = df["amt"].astype("float32")
        out["category"] = df["category"].str.strip().str.lower()
        out["merchant"] = df["merchant"].str.strip().str.lower()
        out["gender"] = df["gender"].str.strip().str.upper().map({"M": "M", "F": "F"}).fillna("U")
        out["age_bucket"] = "U"  # Sparkov has DOB but we skip parsing PII
        out["city"] = df["city"].str.strip().str.lower().fillna("unknown")
        out["state"] = df["state"].str.strip().str.lower().fillna("unknown")
        out["lat"] = df["lat"].astype("float32")
        out["long"] = df["long"].astype("float32")
        out["merch_lat"] = df["merch_lat"].astype("float32")
        out["merch_long"] = df["merch_long"].astype("float32")
        out["channel"] = "pos"  # Sparkov is card-present unless category says net
        out.loc[out["category"].str.contains("_net", na=False), "channel"] = "online"
        out["source"] = "sparkov"
        out["is_fraud"] = df["is_fraud"].astype("int8")
        chunks.append(out)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def load_banksim() -> pd.DataFrame:
    p = DATA / "bs140513_032310.csv"
    if not p.exists():
        print("[warn] bs140513_032310.csv not found — skipping", file=sys.stderr)
        return pd.DataFrame()

    df = pd.read_csv(p)
    # BankSim wraps strings in single-quotes: "'M'" etc.
    def _clean(col):
        return df[col].astype(str).str.strip().str.strip("'")

    cat_raw = _clean("category").str.lower()
    out = pd.DataFrame()
    out["transaction_id"] = "bs_" + df.index.astype(str)
    # step = hour since simulation start; no wall-clock date available
    out["tx_datetime"] = pd.NaT
    out["amount"] = _clean("amount").astype("float32")
    out["category"] = cat_raw.map(BANKSIM_CAT_MAP).fillna("misc_pos")
    out["merchant"] = _clean("merchant").str.lower()
    out["gender"] = _clean("gender").map(
        {"M": "M", "F": "F", "E": "U", "U": "U"}
    ).fillna("U")
    out["age_bucket"] = _clean("age").map(
        {"0": "<18","1": "18-30","2": "30-40","3": "40-50",
         "4": "50-60","5": "60-70","6": "70+","U": "U"}
    ).fillna("U")
    out["city"] = "unknown"
    out["state"] = "unknown"
    out["lat"] = np.nan
    out["long"] = np.nan
    out["merch_lat"] = np.nan
    out["merch_long"] = np.nan
    out["channel"] = "pos"
    out["source"] = "banksim"
    out["is_fraud"] = _clean("fraud").astype(int).astype("int8")
    return out


def load_handbook() -> pd.DataFrame:
    pkl_dir = DATA / "data"
    if not pkl_dir.exists():
        print("[warn] data/data/ not found — skipping handbook", file=sys.stderr)
        return pd.DataFrame()

    frames = []
    for pkl in sorted(pkl_dir.glob("*.pkl")):
        try:
            df = _load_pickle(pkl)
        except Exception as exc:
            print(f"[warn] skip {pkl.name}: {exc}", file=sys.stderr)
            continue
        out = pd.DataFrame()
        out["transaction_id"] = "hb_" + df["TRANSACTION_ID"].astype(str)
        out["tx_datetime"] = pd.to_datetime(df["TX_DATETIME"], errors="coerce")
        out["amount"] = pd.to_numeric(df["TX_AMOUNT"], errors="coerce").astype("float32")
        out["category"] = "unknown"  # handbook has no merchant category
        out["merchant"] = "terminal_" + df["TERMINAL_ID"].astype(str)
        out["gender"] = "U"
        out["age_bucket"] = "U"
        out["city"] = "unknown"
        out["state"] = "unknown"
        out["lat"] = np.nan
        out["long"] = np.nan
        out["merch_lat"] = np.nan
        out["merch_long"] = np.nan
        out["channel"] = "unknown"
        out["source"] = "handbook"
        out["is_fraud"] = df["TX_FRAUD"].astype("int8")
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour, day_of_week, is_weekend from tx_datetime where present."""
    dt = pd.to_datetime(df["tx_datetime"], errors="coerce")
    df["hour"] = dt.dt.hour + dt.dt.minute / 60.0
    # Fill missing hour from sparkov unix_time if we carried it over; else median
    df["hour"] = df["hour"].fillna(12.0).astype("float32")
    df["day_of_week"] = dt.dt.dayofweek.fillna(2).astype("int8")  # 2=Wed fallback
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    return df


def clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df["category"] = df["category"].fillna("unknown").astype(str).str.strip()
    df["merchant"] = df["merchant"].fillna("unknown").astype(str).str.strip()
    df["gender"] = df["gender"].fillna("U")
    df["age_bucket"] = df["age_bucket"].fillna("U")
    df["city"] = df["city"].fillna("unknown")
    df["state"] = df["state"].fillna("unknown")
    df["channel"] = df["channel"].fillna("unknown")
    df["amount"] = df["amount"].clip(lower=0.0)
    df = df.dropna(subset=["amount", "is_fraud"])
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap amount at 99.9th percentile to kill extreme outliers."""
    cap = df["amount"].quantile(0.999)
    df["amount"] = df["amount"].clip(upper=cap)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort dedup on (transaction_id). Source columns are unique by construction."""
    return df.drop_duplicates(subset=["transaction_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rebalancing
# ---------------------------------------------------------------------------

def _jitter_fraud(df_fraud: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Oversample fraud rows with bounded Gaussian noise on amount/lat/long."""
    needed = n
    if needed <= 0:
        return df_fraud

    samples = df_fraud.sample(n=needed, replace=True, random_state=rng.integers(0, 2**31))
    samples = samples.copy()

    # Jitter amount by ±5%, floored at 0
    noise = rng.normal(loc=1.0, scale=0.05, size=len(samples))
    samples["amount"] = (samples["amount"] * noise).clip(lower=0.01).astype("float32")

    # Jitter coordinates by tiny delta (~0-2 km in degrees)
    for col in ("lat", "long", "merch_lat", "merch_long"):
        mask = samples[col].notna()
        delta = rng.normal(0, 0.01, size=mask.sum())
        samples.loc[mask, col] = (samples.loc[mask, col] + delta).astype("float32")

    # Jitter hour by ±15 min
    noise_h = rng.normal(0, 0.25, size=len(samples))
    samples["hour"] = (samples["hour"] + noise_h).clip(0, 23.99).astype("float32")

    samples["transaction_id"] = samples["transaction_id"] + "_aug" + np.arange(needed).astype(str)
    samples["is_fraud"] = 1
    # ponytail: simple Gaussian jitter; upgrade to SMOTE (imbalanced-learn) for higher fidelity
    return samples


def rebalance(df: pd.DataFrame, target_fraud_rate: float = TARGET_FRAUD_RATE) -> pd.DataFrame:
    """Undersample legit + oversample fraud to reach target_fraud_rate."""
    rng = np.random.default_rng(RANDOM_SEED)

    fraud = df[df["is_fraud"] == 1]
    legit = df[df["is_fraud"] == 0]

    n_fraud = len(fraud)
    n_legit = len(legit)

    # Target: fraud / total = target_fraud_rate
    # total = n_fraud_final / target_fraud_rate
    # First determine if we need to oversample fraud or undersample legit (or both)
    # Strategy: keep all fraud, augment if needed, then match legit count
    target_legit = int(n_fraud / target_fraud_rate * (1 - target_fraud_rate))

    aug_fraud = pd.DataFrame()
    n_aug = 0

    if n_fraud < 1000:
        # Too few fraud samples to reach target_legit without absurd ratio — augment fraud
        target_fraud_final = max(n_fraud, int(target_legit * target_fraud_rate / (1 - target_fraud_rate)))
        n_aug = target_fraud_final - n_fraud
        aug_fraud = _jitter_fraud(fraud, n_aug, rng)

    all_fraud = pd.concat([fraud, aug_fraud], ignore_index=True) if n_aug else fraud
    n_fraud_final = len(all_fraud)
    target_legit = int(n_fraud_final / target_fraud_rate * (1 - target_fraud_rate))

    if len(legit) > target_legit:
        legit_sampled = legit.sample(n=target_legit, random_state=RANDOM_SEED)
    else:
        legit_sampled = legit  # already fewer legit than target — use all

    result = pd.concat([all_fraud, legit_sampled], ignore_index=True)
    result = result.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    actual_rate = result["is_fraud"].mean()
    print(f"Rebalanced: {len(result):,} rows | fraud rate: {actual_rate:.2%} "
          f"(target {target_fraud_rate:.0%}) | augmented: {n_aug:,}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FINAL_COLS = [
    "transaction_id", "tx_datetime", "amount", "category", "merchant",
    "gender", "age_bucket", "city", "state",
    "lat", "long", "merch_lat", "merch_long",
    "channel", "hour", "day_of_week", "is_weekend",
    "source", "is_fraud",
]


def main(out: Path = OUT, target_rate: float = TARGET_FRAUD_RATE) -> None:
    print("Loading Sparkov...")
    sp = load_sparkov()
    print(f"  {len(sp):,} rows | fraud {sp['is_fraud'].sum() if len(sp) else 0:,}")

    print("Loading BankSim...")
    bs = load_banksim()
    print(f"  {len(bs):,} rows | fraud {bs['is_fraud'].sum() if len(bs) else 0:,}")

    print("Loading FDH Handbook...")
    hb = load_handbook()
    print(f"  {len(hb):,} rows | fraud {hb['is_fraud'].sum() if len(hb) else 0:,}")

    merged = pd.concat([sp, bs, hb], ignore_index=True)
    print(f"\nMerged raw: {len(merged):,} rows | "
          f"fraud {int(merged['is_fraud'].sum()):,} ({merged['is_fraud'].mean():.4%})")

    merged = add_time_features(merged)
    merged = clean_nulls(merged)
    merged = remove_outliers(merged)
    merged = remove_duplicates(merged)
    print(f"After clean: {len(merged):,} rows | "
          f"fraud {int(merged['is_fraud'].sum()):,} ({merged['is_fraud'].mean():.4%})")

    balanced = rebalance(merged, target_rate)

    # Final column order
    for col in FINAL_COLS:
        if col not in balanced.columns:
            balanced[col] = np.nan

    balanced = balanced[FINAL_COLS]
    balanced["is_fraud"] = balanced["is_fraud"].astype("int8")
    balanced["tx_datetime"] = balanced["tx_datetime"].astype(str)
    balanced["amount"] = balanced["amount"].round(2)

    out.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(out, index=False)
    print(f"\nSaved {len(balanced):,} rows to {out}")
    print(f"  is_fraud distribution:\n{balanced['is_fraud'].value_counts()}")
    print(f"  source distribution:\n{balanced['source'].value_counts()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Merge fraud datasets into one CSV")
    p.add_argument("--out", default=str(OUT), help="Output CSV path")
    p.add_argument("--target-rate", type=float, default=TARGET_FRAUD_RATE,
                   help="Target fraud rate (default 0.35)")
    args = p.parse_args()
    main(out=Path(args.out), target_rate=args.target_rate)
