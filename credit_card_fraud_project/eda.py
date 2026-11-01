"""STEP 2 - Exploratory data analysis (Unit II): statistics, correlation, outliers, charts.

Run on its own (after data_prep.py):   python eda.py
It prints a summary and saves PNG charts to reports/. EDA uses the TRAINING set only,
so no decision is influenced by the test set.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

import plots
from config import ARTIFACTS, FEATURES, REPORTS, SEED, TARGET


def iqr_outlier_mask(X, factor=1.5):
    """IQR rule: a value is an outlier if it lies 1.5 x IQR below Q1 or above Q3."""
    q1, q3 = X.quantile(.25), X.quantile(.75)
    iqr = q3 - q1
    return (X < q1 - factor * iqr) | (X > q3 + factor * iqr)


def compute_eda(df):
    """Return a JSON-friendly dict with all EDA statistics."""
    X, y = df[FEATURES], df[TARGET]

    corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET)
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)

    c = X.corr().abs()                                   # near-duplicate features (multicollinearity)
    upper = c.where(np.triu(np.ones(c.shape, dtype=bool), k=1)).stack()
    redundant = [{"a": a, "b": b, "abs_corr": round(float(v), 3)}
                 for (a, b), v in upper[upper >= 0.8].sort_values(ascending=False).head(10).items()]

    mask = iqr_outlier_mask(X)
    per_row = mask.sum(axis=1)
    iso = IsolationForest(n_estimators=200, contamination=float(y.mean()), random_state=SEED, n_jobs=-1)
    flag = (iso.fit_predict(X.fillna(X.median())) == -1).astype(int)

    return {
        "rows": len(df),
        "class_counts": {"Legitimate": int((y == 0).sum()), "Fraud": int((y == 1).sum())},
        "fraud_rate": float(y.mean()),
        "missing_values": int(X.isna().sum().sum()),
        "amount_skewness": float(df["Amount"].skew()),
        "corr_with_class": corr.round(4).to_dict(),
        "redundant_pairs": redundant,
        "iqr_outliers_per_feature": mask.sum().sort_values(ascending=False).to_dict(),
        "mean_outlier_features_per_row": {"Legitimate": float(per_row[y == 0].mean()),
                                          "Fraud": float(per_row[y == 1].mean())},
        "isolation_forest": {"flagged": int(flag.sum()),
                             "precision": float(precision_score(y, flag, zero_division=0)),
                             "recall": float(recall_score(y, flag, zero_division=0))},
        "note": "Outliers are reported, not removed: in fraud detection they often ARE the fraud.",
    }


def save_eda_charts(df, eda):
    REPORTS.mkdir(exist_ok=True)
    corr = pd.Series(eda["corr_with_class"])
    figs = {"eda_class_distribution.png": plots.class_distribution(eda["class_counts"]),
            "eda_amount_by_class.png": plots.amount_by_class(df),
            "eda_target_correlation.png": plots.target_correlation(corr),
            "eda_heatmap.png": plots.heatmap(df, list(corr.index[:10]) + [TARGET]),
            "eda_iqr_outliers.png": plots.iqr_outliers(eda["iqr_outliers_per_feature"])}
    for name, fig in figs.items():
        fig.savefig(REPORTS / name, dpi=150, bbox_inches="tight")
    print(f"  EDA charts saved to {REPORTS}")


def print_summary(eda):
    print(f"  rows {eda['rows']:,} | fraud {eda['fraud_rate']:.2%} | missing values {eda['missing_values']} "
          f"| Amount skewness {eda['amount_skewness']:.2f}")
    print(f"  top correlated features: {list(eda['corr_with_class'])[:5]}")
    print(f"  mean outlier features per row: {eda['mean_outlier_features_per_row']}")
    iso = eda["isolation_forest"]
    print(f"  Isolation Forest vs real fraud: precision {iso['precision']:.2%}, recall {iso['recall']:.2%}")


if __name__ == "__main__":
    path = ARTIFACTS / "train.csv"
    if not path.exists():
        raise SystemExit("Run first:  python data_prep.py --data data/creditcard_2023.csv")
    train = pd.read_csv(path)
    result = compute_eda(train)
    print_summary(result)
    save_eda_charts(train, result)
