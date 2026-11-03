
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance

from config import FEATURES, SEED


# STEP 1: PERMUTATION FEATURE IMPORTANCE

def permutation_importances(model, X, y):

    result = permutation_importance(
        model,
        X,
        y,
        scoring="average_precision",
        n_repeats=5,
        random_state=SEED
    )

    importance = {}

    for i, feature in enumerate(FEATURES):

        importance[feature] = {
            "mean": float(result.importances_mean[i]),
            "std": float(result.importances_std[i])
        }

    return dict(
        sorted(
            importance.items(),
            key=lambda item: item[1]["mean"],
            reverse=True
        )
    )


# STEP 2: EXPLAIN ONE TRANSACTION

def occlusion_explain(model, row, medians):

    if len(row) != 1:
        raise ValueError("Provide exactly one transaction")

    missing = [
        feature for feature in FEATURES
        if feature not in row.columns
    ]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    original_probability = model.predict_proba(
        row[FEATURES]
    )[0, 1]

    explanation = {}

    for feature in FEATURES:

        changed_row = row[FEATURES].copy()

        changed_row.loc[:, feature] = medians[feature]

        new_probability = model.predict_proba(
            changed_row
        )[0, 1]

        explanation[feature] = float(
            original_probability - new_probability
        )

    explanation = pd.Series(explanation)

    explanation = explanation.reindex(
        explanation.abs().sort_values(
            ascending=False
        ).index
    )

    return float(original_probability), explanation


# STEP 3: COST CURVE

def cost_curve(
    y,
    scores,
    cost_missed_fraud,
    cost_false_alarm
):

    y = np.asarray(y)
    scores = np.asarray(scores)

    thresholds = np.linspace(0.01, 0.99, 99)

    costs = []

    for threshold in thresholds:

        missed_fraud = (
            (scores < threshold) & (y == 1)
        ).sum()

        false_alarms = (
            (scores >= threshold) & (y == 0)
        ).sum()

        total_cost = (
            cost_missed_fraud * missed_fraud
            + cost_false_alarm * false_alarms
        )

        costs.append(total_cost)

    best_index = np.argmin(costs)

    return (
        thresholds,
        np.array(costs),
        float(thresholds[best_index])
    )


# STEP 4: METRICS AT A SELECTED THRESHOLD

def metrics_at(y, scores, threshold):

    y = np.asarray(y)
    scores = np.asarray(scores)

    predictions = scores >= threshold

    true_positive = (
        (predictions == True) & (y == 1)
    ).sum()

    actual_fraud = (y == 1).sum()

    flagged = predictions.sum()

    recall = true_positive / max(actual_fraud, 1)

    precision = true_positive / max(flagged, 1)

    return {
        "recall": float(recall),
        "precision": float(precision),
        "flagged": int(flagged)
    }
