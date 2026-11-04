
import json
import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.utils import resample

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)

from sklearn.model_selection import train_test_split

from config import (
    MODEL_FILE,
    METRICS_FILE,
    COMPARISON_FILE,
    PREDICTIONS_FILE,
    SEED,
    MAX_DISTANCE_MODEL_SAMPLES_PER_CLASS,
    EVALUATION_SAMPLE_SIZE
)

from data_prep import prepare_data
from models import get_models, build_pipeline


DISTANCE_BASED_MODELS = [
    "KNN",
    "SVM",
    "Naive Bayes"
]


def make_balanced_sample(X, y):

    data = X.copy()

    data["_target"] = y.to_numpy()

    class_zero = data[data["_target"] == 0]

    class_one = data[data["_target"] == 1]

    sample_size = min(
        MAX_DISTANCE_MODEL_SAMPLES_PER_CLASS,
        len(class_zero),
        len(class_one)
    )

    if sample_size == 0:

        raise ValueError(
            "Both classes must have training samples."
        )

    zero_sample = resample(
        class_zero,
        replace=False,
        n_samples=sample_size,
        random_state=SEED
    )

    one_sample = resample(
        class_one,
        replace=False,
        n_samples=sample_size,
        random_state=SEED
    )

    balanced_data = pd.concat([
        zero_sample,
        one_sample
    ])

    balanced_data = balanced_data.sample(
        frac=1,
        random_state=SEED
    )

    y_sample = balanced_data.pop("_target")

    X_sample = balanced_data

    return X_sample, y_sample


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    results = {

        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y_test,
            probabilities
        )

    }

    return results, predictions, probabilities


def main():

    print("\nLoading dataset...")

    X_train, X_test, y_train, y_test = prepare_data()

    # Use the same stratified evaluation sample for all models
    evaluation_size = min(
        EVALUATION_SAMPLE_SIZE,
        len(X_test) - 1
    )

    X_eval, _, y_eval, _ = train_test_split(
        X_test,
        y_test,
        train_size=evaluation_size,
        stratify=y_test,
        random_state=SEED
    )

    print("\nEvaluation sample:", X_eval.shape)

    models = get_models()

    comparison_results = {}

    trained_models = {}

    for model_name, classifier in models.items():

        print("\n" + "=" * 55)

        print("Training:", model_name)

        print("=" * 55)

        # Use balanced sample for expensive algorithms
        if model_name in DISTANCE_BASED_MODELS:

            X_fit, y_fit = make_balanced_sample(
                X_train,
                y_train
            )

            print(
                "Training sample:",
                X_fit.shape
            )

        else:

            X_fit = X_train

            y_fit = y_train

            print(
                "Using complete training data:",
                X_fit.shape
            )

        pipeline = build_pipeline(
            clone(classifier)
        )

        pipeline.fit(
            X_fit,
            y_fit
        )

        results, predictions, probabilities = evaluate_model(
            pipeline,
            X_eval,
            y_eval
        )

        comparison_results[model_name] = results

        trained_models[model_name] = pipeline

        print("\nEvaluation metrics:")

        for metric, value in results.items():

            print(
                f"{metric}: {value:.4f}"
            )

        print("\nClassification report:")

        print(
            classification_report(
                y_eval,
                predictions,
                zero_division=0
            )
        )

    # Select model using PR-AUC
    best_model_name = max(
        comparison_results,
        key=lambda name: comparison_results[name]["pr_auc"]
    )

    best_model = trained_models[best_model_name]

    print("\n" + "=" * 55)

    print("BEST MODEL:", best_model_name)

    print("=" * 55)

    # Save selected model and its preprocessing pipeline
    joblib.dump(
        best_model,
        MODEL_FILE
    )

    print("\nModel saved:", MODEL_FILE)

    # Save comparison results
    comparison_df = pd.DataFrame(
        comparison_results
    ).T

    comparison_df.index.name = "Model"

    comparison_df.to_csv(
        COMPARISON_FILE
    )

    # Save metrics
    with open(METRICS_FILE, "w") as file:

        json.dump(
            {
                "best_model": best_model_name,

                "evaluation_sample_size": len(X_eval),

                "metrics": comparison_results

            },

            file,

            indent=4
        )

    # Save predictions
    best_predictions = best_model.predict(X_eval)

    best_probabilities = best_model.predict_proba(
        X_eval
    )[:, 1]

    prediction_df = pd.DataFrame({

        "actual": y_eval.to_numpy(),

        "predicted": best_predictions,

        "fraud_probability": best_probabilities

    })

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    print("\nTraining completed.")

    print("Best model:", best_model_name)

    print("Model comparison:", COMPARISON_FILE)

    print("Metrics:", METRICS_FILE)

    print("Predictions:", PREDICTIONS_FILE)


if __name__ == "__main__":

    main()
