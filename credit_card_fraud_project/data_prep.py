
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (ARTIFACTS, DATA_FILE, FEATURES, FRAUD_RATE, N_LEGIT, SEED, TARGET, TEST_SIZE)


def load_data(path):
    
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(f"\nDataset not found: {path}\n"
                         "Download creditcard_2023.csv (link in README.md) and put it in the data/ folder,\n"
                         "or test with fake data:  python make_demo_data.py\n")
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"The CSV is missing columns: {missing}. Is it the 2023 dataset?")
    return df


def clean_data(df):
    """Clean the data and return (clean_df, report). Missing values are NOT filled here:
    filling must use training data only, so it happens inside the model pipeline."""
    report = {"rows_before": len(df)}
    df = df[[c for c in FEATURES + [TARGET] if c in df.columns]].copy()       # drops 'id' etc.
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")                         # text -> NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    report["negative_amounts"] = int((df["Amount"] < 0).sum())
    df.loc[df["Amount"] < 0, "Amount"] = np.nan                               # impossible value
    bad = ~df[TARGET].isin([0, 1])
    report["invalid_label_rows"] = int(bad.sum())
    df = df[~bad].copy()
    df[TARGET] = df[TARGET].astype(int)
    report["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    report["rows_after"] = len(df)
    report["missing_values_left"] = int(df.isna().sum().sum())
    report["raw_fraud_share"] = round(float(df[TARGET].mean()), 4)
    return df, report


def make_study_set(df, n_legit=N_LEGIT, fraud_rate=FRAUD_RATE):
    """The 2023 file is ~50% fraud, which is unrealistic. Keep n_legit normal rows and only
    enough fraud rows to reach `fraud_rate`, so the class-imbalance problem is real again."""
    legit, fraud = df[df[TARGET] == 0], df[df[TARGET] == 1]
    n_legit = min(n_legit, len(legit))
    n_fraud = min(int(round(n_legit * fraud_rate / (1 - fraud_rate))), len(fraud))
    study = pd.concat([legit.sample(n_legit, random_state=SEED), fraud.sample(n_fraud, random_state=SEED)])
    return study.sample(frac=1, random_state=SEED).reset_index(drop=True)


def split_data(df):
    """80/20 split that keeps the same fraud share in both parts (stratify)."""
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=SEED)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def prepare(path=DATA_FILE, n_legit=N_LEGIT, fraud_rate=FRAUD_RATE):
    """Full step 1. Saves train/test CSVs and returns (train, test, info)."""
    raw = load_data(path)
    clean, report = clean_data(raw)
    study = make_study_set(clean, n_legit, fraud_rate)
    train, test = split_data(study)
    ARTIFACTS.mkdir(exist_ok=True)
    train.round(4).to_csv(ARTIFACTS / "train.csv", index=False)
    test.round(4).to_csv(ARTIFACTS / "test.csv", index=False)
    info = {"cleaning": report, "raw_rows": len(raw), "study_rows": len(study),
            "fraud_rate": float(study[TARGET].mean()), "train_rows": len(train), "test_rows": len(test)}
    print(f"Data ready: {len(raw):,} raw rows -> {len(study):,} study rows "
          f"({info['fraud_rate']:.2%} fraud) | train {len(train):,} / test {len(test):,}")
    return train, test, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_FILE))
    ap.add_argument("--n_legit", type=int, default=N_LEGIT)
    ap.add_argument("--fraud_rate", type=float, default=FRAUD_RATE)
    a = ap.parse_args()
    prepare(a.data, a.n_legit, a.fraud_rate)
