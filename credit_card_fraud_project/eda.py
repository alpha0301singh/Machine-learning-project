
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score

from config import FEATURES, TARGET, SEED, REPORTS
import plots


# STEP 1: DETECT OUTLIERS USING IQR

def iqr_outlier_mask(data, factor=1.5):

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    return (data < lower) | (data > upper)


# STEP 2: PERFORM EDA

def compute_eda(data):

    X = data[FEATURES]
    y = data[TARGET]

    # Class distribution
    class_counts = {
        "Legitimate": int((y == 0).sum()),
        "Fraud": int((y == 1).sum())
    }

    # Missing values
    missing_values = data.isnull().sum().to_dict()

    # Amount skewness
    amount_skewness = float(data["Amount"].skew())

    # Correlation with target
    correlations = data[FEATURES + [TARGET]].corr()[TARGET]
    correlations = correlations.drop(TARGET)

    # Strongly correlated feature pairs
    correlation_matrix = data[FEATURES].corr()

    redundant_pairs = []

    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):

            value = correlation_matrix.iloc[i, j]

            if abs(value) >= 0.8:
                redundant_pairs.append({
                    "feature_1": FEATURES[i],
                    "feature_2": FEATURES[j],
                    "correlation": float(value)
                })

    # IQR outlier detection
    outlier_mask = iqr_outlier_mask(X)

    outlier_counts = outlier_mask.sum().to_dict()

    mean_outliers = float(outlier_mask.sum(axis=1).mean())

    # Isolation Forest
    filled_data = X.fillna(X.median())

    contamination = float(y.mean())

    isolation_forest = IsolationForest(
        contamination=contamination,
        random_state=SEED
    )

    predictions = isolation_forest.fit_predict(filled_data)

    # Isolation Forest: -1 means anomaly, 1 means normal
    anomaly_flags = predictions == -1

    isolation_results = {
        "flagged": int(anomaly_flags.sum()),
        "precision": float(
            precision_score(y, anomaly_flags, zero_division=0)
        ),
        "recall": float(
            recall_score(y, anomaly_flags, zero_division=0)
        )
    }

    results = {
        "class_counts": class_counts,
        "missing_values": missing_values,
        "amount_skewness": amount_skewness,
        "corr_with_class": correlations.to_dict(),
        "redundant_pairs": redundant_pairs,
        "isolation_forest": isolation_results,
        "iqr_outliers_per_feature": outlier_counts,
        "mean_outlier_features_per_row": mean_outliers,
        "note": (
            "Outlier detection identifies unusual observations. "
            "An unusual transaction is not automatically fraudulent."
        )
    }

    return results


# STEP 3: PRINT EDA SUMMARY

def print_summary(results):

    print("\nEXPLORATORY DATA ANALYSIS")

    print("\nClass Distribution:")
    print(results["class_counts"])

    print("\nMissing Values:")
    print(results["missing_values"])

    print("\nAmount Skewness:")
    print(results["amount_skewness"])

    print("\nFeature Correlation with Fraud:")
    print(results["corr_with_class"])

    print("\nIsolation Forest Results:")
    print(results["isolation_forest"])

    print("\nIQR Outliers:")
    print(results["iqr_outliers_per_feature"])


# STEP 4: SAVE EDA CHARTS

def save_eda_charts(data, results):

    REPORTS.mkdir(parents=True, exist_ok=True)

    save_chart = lambda fig, name: fig.savefig(
        REPORTS / name,
        dpi=150,
        bbox_inches="tight"
    )

    save_chart(
        plots.class_distribution(results["class_counts"]),
        "class_distribution.png"
    )

    save_chart(
        plots.amount_by_class(data),
        "amount_distribution.png"
    )

    save_chart(
        plots.target_correlation(
            pd.Series(results["corr_with_class"])
        ),
        "target_correlation.png"
    )

    # Use a smaller set of features to keep the heatmap readable
    heatmap_features = FEATURES[:10]

    save_chart(
        plots.heatmap(data, heatmap_features),
        "correlation_heatmap.png"
    )

    save_chart(
        plots.iqr_outliers(results["iqr_outliers_per_feature"]),
        "iqr_outliers.png"
    )


if __name__ == "__main__":

    from config import ARTIFACTS

    train_data = pd.read_csv(ARTIFACTS / "train.csv")

    results = compute_eda(train_data)

    print_summary(results)

    save_eda_charts(train_data, results)
