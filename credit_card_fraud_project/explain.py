
import joblib
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from config import (
    MODEL_FILE,
    REPORTS,
    SEED
)

from data_prep import prepare_data


def explain_model():

    print("Loading trained model...")

    model = joblib.load(MODEL_FILE)

    X_train, X_test, y_train, y_test = prepare_data()

    # Use a smaller stratified sample for explanation
    sample_size = min(
        5000,
        len(X_test)
    )

    X_sample, _, y_sample, _ = train_test_split(
        X_test,
        y_test,
        train_size=sample_size,
        stratify=y_test,
        random_state=SEED
    )

    print("Calculating permutation importance...")

    results = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="average_precision",
        n_repeats=3,
        random_state=SEED,
        n_jobs=1
    )

    importance_df = pd.DataFrame({

        "feature": X_sample.columns,

        "importance_mean": results.importances_mean,

        "importance_std": results.importances_std

    })

    importance_df = importance_df.sort_values(
        by="importance_mean",
        ascending=False
    )

    importance_df.to_csv(
        REPORTS / "feature_importance.csv",
        index=False
    )

    print("\nFeature importance:")

    print(importance_df)

    print("\nResults saved in:", REPORTS)


if __name__ == "__main__":

    explain_model()
