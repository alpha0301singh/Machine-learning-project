
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance

from config import FEATURES, SEED


# Step 1: Find Important Features

def permutation_importances(model, X, y):

    # Calculate importance of each feature
    result = permutation_importance(
        model,
        X,
        y,
        scoring="average_precision",
        n_repeats=5,
        random_state=SEED
    )

    # Store feature importance
    importance = {}

    for i in range(len(FEATURES)):

        feature = FEATURES[i]

        importance[feature] = {
            "mean": float(result.importances_mean[i]),
            "std": float(result.importances_std[i])
        }

    # Sort features by importance
    importance = dict(
        sorted(
            importance.items(),
            key=lambda x: x[1]["mean"],
            reverse=True
        )
    )

    return importance


# Step 2: Explain One Transaction

def occlusion_explain(model, row, medians):

    # Calculate original fraud probability
    original_probability = model.predict_proba(
        row[FEATURES]
    )[0, 1]

    # Create copies of the transaction
    new_data = pd.concat(
        [row[FEATURES]] * len(FEATURES),
        ignore_index=True
    )

    # Replace one feature at a time with its median value
    for i in range(len(FEATURES)):

        feature = FEATURES[i]

        new_data.loc[i, feature] = medians[feature]

    # Calculate new fraud probabilities
    new_probability = model.predict_proba(
        new_data[FEATURES]
    )[:, 1]

    # Calculate the effect of each feature
    explanation = original_probability - new_probability

    # Store feature explanations
    explanation = pd.Series(
        explanation,
        index=FEATURES
    )

    # Sort features by their impact
    explanation = explanation.reindex(
        explanation.abs().sort_values(ascending=False).index
    )

    return float(original_probability), explanation


# Step 3: Calculate Cost at Different Thresholds

def cost_curve(y, score, cost_missed_fraud, cost_false_alarm):

    y = np.asarray(y)
    score = np.asarray(score)

    # Test different threshold values
    thresholds = np.linspace(0.01, 0.99, 99)

    costs = []

    for threshold in thresholds:

        # Count missed fraud transactions
        missed_fraud = (
            (score < threshold) & (y == 1)
        ).sum()

        # Count false alarms
        false_alarms = (
            (score >= threshold) & (y == 0)
        ).sum()

        # Calculate total cost
        total_cost = (
            cost_missed_fraud * missed_fraud
            +
            cost_false_alarm * false_alarms
        )

        costs.append(total_cost)

    # Find threshold with minimum cost
    best_index = np.argmin(costs)

    best_threshold = thresholds[best_index]

    return thresholds, np.array(costs), float(best_threshold)


# Step 4: Calculate Performance at a Threshold

def metrics_at(y, score, threshold):

    y = np.asarray(y)
    score = np.asarray(score)

    # Predict fraud when probability is greater than threshold
    predictions = score >= threshold

    # Count correctly detected fraud transactions
    true_positive = (
        (predictions == True) & (y == 1)
    ).sum()

    # Count all fraud transactions
    actual_fraud = (y == 1).sum()

    # Count all flagged transactions
    flagged_transactions = predictions.sum()

    # Calculate recall
    recall = true_positive / max(actual_fraud, 1)

    # Calculate precision
    precision = true_positive / max(flagged_transactions, 1)

    results = {

        "recall": float(recall),

        "precision": float(precision),

        "flagged": int(flagged_transactions)

    }

    return results
