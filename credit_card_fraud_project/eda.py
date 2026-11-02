
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

import plots

from config import ARTIFACTS, FEATURES, REPORTS, SEED, TARGET


# Step 1: Detect Outliers Using IQR

def detect_outliers(data):

    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = (data < lower_limit) | (data > upper_limit)

    return outliers


# Step 2: Perform Exploratory Data Analysis

def perform_eda(data):

    X = data[FEATURES]
    y = data[TARGET]

    # Count normal and fraud transactions
    normal_count = (y == 0).sum()
    fraud_count = (y == 1).sum()

    # Calculate fraud percentage
    fraud_rate = y.mean()

    # Calculate missing values
    missing_values = X.isnull().sum().sum()

    # Calculate Amount skewness
    amount_skewness = data["Amount"].skew()

    # Calculate correlation with fraud
    correlation = data[FEATURES + [TARGET]].corr()

    correlation = correlation[TARGET].drop(TARGET)

    correlation = correlation.reindex(
        correlation.abs().sort_values(ascending=False).index
    )

    # Find highly correlated features
    feature_correlation = X.corr().abs()

    redundant_pairs = []

    for i in range(len(FEATURES)):

        for j in range(i + 1, len(FEATURES)):

            value = feature_correlation.iloc[i, j]

            if value >= 0.8:

                redundant_pairs.append({
                    "a": FEATURES[i],
                    "b": FEATURES[j],
                    "abs_corr": round(float(value), 3)
                })

    # Sort correlated feature pairs
    redundant_pairs = sorted(
        redundant_pairs,
        key=lambda x: x["abs_corr"],
        reverse=True
    )

    redundant_pairs = redundant_pairs[:10]

    # Detect outliers
    outliers = detect_outliers(X)

    # Count outliers in each row
    row_outliers = outliers.sum(axis=1)

    # Calculate average outliers for normal transactions
    normal_outliers = row_outliers[y == 0].mean()

    # Calculate average outliers for fraud transactions
    fraud_outliers = row_outliers[y == 1].mean()

    # Apply Isolation Forest
    X_filled = X.fillna(X.median())

    model = IsolationForest(
        n_estimators=200,
        contamination=float(fraud_rate),
        random_state=SEED
    )

    model.fit(X_filled)

    predictions = model.predict(X_filled)

    # Isolation Forest gives -1 for outliers and 1 for normal values
    predictions = (predictions == -1).astype(int)

    # Calculate precision and recall
    precision = precision_score(
        y, predictions, zero_division=0
    )

    recall = recall_score(
        y, predictions, zero_division=0
    )

    # Store all results
    results = {

        "rows": len(data),

        "class_counts": {
            "Legitimate": int(normal_count),
            "Fraud": int(fraud_count)
        },

        "fraud_rate": float(fraud_rate),

        "missing_values": int(missing_values),

        "amount_skewness": float(amount_skewness),

        "corr_with_class": correlation.round(4).to_dict(),

        "redundant_pairs": redundant_pairs,

        "iqr_outliers_per_feature":
            outliers.sum().sort_values(ascending=False).to_dict(),

        "mean_outlier_features_per_row": {
            "Legitimate": float(normal_outliers),
            "Fraud": float(fraud_outliers)
        },

        "isolation_forest": {
            "flagged": int(predictions.sum()),
            "precision": float(precision),
            "recall": float(recall)
        },

        "note": "Outliers are detected but not removed because unusual transactions can be fraudulent."
    }

    return results


# Step 3: Save Graphs

def save_charts(data, results):

    REPORTS.mkdir(parents=True, exist_ok=True)

    correlation = pd.Series(results["corr_with_class"])

    # Generate graphs
    fig1 = plots.class_distribution(
        results["class_counts"]
    )

    fig2 = plots.amount_by_class(data)

    fig3 = plots.target_correlation(correlation)

    fig4 = plots.heatmap(
        data,
        list(correlation.index[:10]) + [TARGET]
    )

    fig5 = plots.iqr_outliers(
        results["iqr_outliers_per_feature"]
    )

    # Save graphs
    fig1.savefig(
        REPORTS / "eda_class_distribution.png",
        dpi=150,
        bbox_inches="tight"
    )

    fig2.savefig(
        REPORTS / "eda_amount_by_class.png",
        dpi=150,
        bbox_inches="tight"
    )

    fig3.savefig(
        REPORTS / "eda_target_correlation.png",
        dpi=150,
        bbox_inches="tight"
    )

    fig4.savefig(
        REPORTS / "eda_heatmap.png",
        dpi=150,
        bbox_inches="tight"
    )

    fig5.savefig(
        REPORTS / "eda_iqr_outliers.png",
        dpi=150,
        bbox_inches="tight"
    )

    print("EDA graphs saved successfully!")


# Step 4: Display EDA Results

def print_results(results):

    print("\n========== EDA RESULTS ==========")

    print("Total Transactions:", results["rows"])

    print("Fraud Percentage:",
          round(results["fraud_rate"] * 100, 2), "%")

    print("Missing Values:",
          results["missing_values"])

    print("Amount Skewness:",
          round(results["amount_skewness"], 2))

    print("\nClass Distribution:")
    print(results["class_counts"])

    print("\nTop Correlated Features:")
    print(list(results["corr_with_class"])[:5])

    print("\nOutliers Per Transaction:")
    print(results["mean_outlier_features_per_row"])

    print("\nIsolation Forest Results:")

    isolation = results["isolation_forest"]

    print("Flagged Transactions:", isolation["flagged"])

    print("Precision:",
          round(isolation["precision"] * 100, 2), "%")

    print("Recall:",
          round(isolation["recall"] * 100, 2), "%")

    print("\n=================================")


# Step 5: Main Function

def main():

    # Read training dataset
    file_path = ARTIFACTS / "train.csv"

    if not file_path.exists():

        print("Training dataset not found!")

        print("First run: python data_prep.py")

        return

    data = pd.read_csv(file_path)

    # Perform EDA
    results = perform_eda(data)

    # Display results
    print_results(results)

    # Save graphs
    save_charts(data, results)


# Run Program

if __name__ == "__main__":

    main()
