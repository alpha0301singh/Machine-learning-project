"""Explainability and decision analysis: WHY is a transaction flagged, and WHERE to set the cutoff?"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config import FEATURES, SEED


def permutation_importances(model, X, y):
    """Shuffle one feature at a time on the TEST set; the bigger the drop in PR-AUC,
    the more the model relies on that feature."""
    r = permutation_importance(model, X, y, scoring="average_precision", n_repeats=5,
                               random_state=SEED, n_jobs=-1)
    order = np.argsort(-r.importances_mean)
    return {FEATURES[i]: {"mean": float(r.importances_mean[i]), "std": float(r.importances_std[i])}
            for i in order}


def occlusion_explain(model, row, medians):
    """Explain ONE transaction: reset each feature to its typical (median) value and see how
    much the fraud probability changes. Positive = that feature pushed the score towards fraud."""
    base = float(model.predict_proba(row[FEATURES])[0, 1])
    batch = pd.concat([row[FEATURES]] * len(FEATURES), ignore_index=True)
    for i, f in enumerate(FEATURES):
        batch.loc[i, f] = medians[f]
    p = model.predict_proba(batch[FEATURES])[:, 1]
    return base, pd.Series(base - p, index=FEATURES).sort_values(key=abs, ascending=False)


def cost_curve(y, score, cost_missed_fraud, cost_false_alarm):
    """Total cost at every threshold. Returns (thresholds, costs, best_threshold)."""
    y, score = np.asarray(y), np.asarray(score)
    grid = np.linspace(0.01, 0.99, 99)
    fn = np.array([((score < t) & (y == 1)).sum() for t in grid])
    fp = np.array([((score >= t) & (y == 0)).sum() for t in grid])
    cost = cost_missed_fraud * fn + cost_false_alarm * fp
    return grid, cost, float(grid[int(np.argmin(cost))])


def metrics_at(y, score, threshold):
    y, score = np.asarray(y), np.asarray(score)
    flagged = score >= threshold
    tp = int((flagged & (y == 1)).sum())
    return {"recall": tp / max(int(y.sum()), 1), "precision": tp / max(int(flagged.sum()), 1),
            "flagged": int(flagged.sum())}
