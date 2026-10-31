"""Shared constants + preprocessing. Imported by train.py AND app.py
(the saved model references log_amount, so the app must be able to import it)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

V_COLS = [f"V{i}" for i in range(1, 29)]   # anonymised (PCA) features
FEATURES = V_COLS + ["Amount"]
TARGET = "Class"                            # 1 = fraud, 0 = legitimate


def log_amount(X):
    """Feature transformation: log(1 + Amount) to tame the skewed amount."""
    return np.log1p(np.clip(X, 0, None))


def make_prep():
    """Unit I/II preprocessing: impute -> transform -> scale, as one ColumnTransformer."""
    amount = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(log_amount, feature_names_out="one-to-one")),
        ("sc", StandardScaler()),
    ])
    v_feats = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    return ColumnTransformer([("amt", amount, ["Amount"]), ("v", v_feats, V_COLS)])


def load_data(path):
    """Load the Kaggle CSV, drop the id column and duplicates, validate the schema."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ("id", "Id", "ID") if c in df.columns]).drop_duplicates()
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")
    return df[FEATURES + [TARGET]].reset_index(drop=True)
