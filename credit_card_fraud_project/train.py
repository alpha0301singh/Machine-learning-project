import json
import joblib
import gc

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.base import clone
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

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

from models import (
    get_final_models,
    build_pipeline
)


# ----------------------------------------
# MODELS THAT USE BALANCED TRAINING DATA
# ----------------------------------------

DISTANCE_BASED_MODELS = [
    "KNN",
    "SVM",
    "Naive Bayes"
]


# ----------------------------------------
# CREATE BALANCED TRAINING SAMPLE
# ----------------------------------------

def make_balanced_sample(X, y):

    data = X.copy()

    data["_target"] = np.asarray(y)

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

    return balanced_data, y_sample


# ----------------------------------------
# CREATE EVALUATION SAMPLE
# ----------------------------------------

def make_evaluation_sample(X, y):

    sample_size = min(
        EVALUATION_SAMPLE_SIZE,
        len(X)
    )

    if sample_size == len(X):

        return X.copy(), y.copy()

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=sample_size,
        stratify=y,
        random_state=SEED
    )

    return X_sample, y_sample


# ----------------------------------------
# GET TRAINING DATA
# ----------------------------------------

def get_training_data(model_name, X, y):

    if model_name in DISTANCE_BASED_MODELS:

        return make_balanced_sample(X, y)

    return X, y


# ----------------------------------------
# EVALUATE MODEL
# ----------------------------------------

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    results = {

        "accuracy": accuracy_score(
            y,
            predictions
        ),

        "precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),

        "f1_score": f1_score(
            y,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y,
            probabilities
        )
    }

    return results, predictions, probabilities


# ----------------------------------------
# MAIN FUNCTION
# ----------------------------------------

def main():

    print("\n" + "=" * 65)

    print("CREDIT CARD FRAUD DETECTION - FINAL MODEL TRAINING")

    print("=" * 65)

    # --------------------------------
    # STEP 1: LOAD DATA
    # --------------------------------

    print("\nLoading dataset...")

    X_train, X_test, y_train, y_test = prepare_data()

    print("\nTraining data:", X_train.shape)

    print("Testing data:", X_test.shape)

    # --------------------------------
    # STEP 2: CREATE TEST EVALUATION DATA
    # --------------------------------

    print("\nPreparing test evaluation data...")

    X_test_eval, y_test_eval = make_evaluation_sample(
        X_test,
        y_test
    )

    print("Test evaluation sample:", X_test_eval.shape)

    print("\nFraud cases in evaluation data:")

    print(pd.Series(y_test_eval).value_counts())

    # --------------------------------
    # STEP 3: LOAD FINAL MODELS
    # --------------------------------

    models = get_final_models()

    final_test_results = {}

    best_model_pipeline = None

    best_predictions = None

    best_probabilities = None

    # Selected from the completed validation experiments.
    best_model_name = "Random Forest"

    # --------------------------------
    # STEP 4: TRAIN AND EVALUATE ALL MODELS
    # --------------------------------

    for model_name, (settings, classifier) in models.items():

        print("\n" + "=" * 65)

        print("MODEL:", model_name)

        print("SETTINGS:", settings)

        print("=" * 65)

        X_fit, y_fit = get_training_data(
            model_name,
            X_train,
            y_train
        )

        print("\nTraining sample:", X_fit.shape)

        pipeline = build_pipeline(
            clone(classifier)
        )

        print("Training model...")

        pipeline.fit(
            X_fit,
            y_fit
        )

        print("Evaluating model...")

        results, predictions, probabilities = evaluate_model(
            pipeline,
            X_test_eval,
            y_test_eval
        )

        final_test_results[model_name] = results

        print("\nTest results:")

        for metric, value in results.items():

            print(f"{metric}: {value:.4f}")

        # Keep the selected Random Forest model.

        if model_name == best_model_name:

            best_model_pipeline = pipeline

            best_predictions = predictions

            best_probabilities = probabilities

        del X_fit
        del y_fit

        if model_name != best_model_name:

            del pipeline

        gc.collect()

    # --------------------------------
    # STEP 5: SAVE BEST MODEL
    # --------------------------------

    if best_model_pipeline is None:

        raise RuntimeError(
            "Selected best model was not trained."
        )

    Path(MODEL_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_model_pipeline,
        MODEL_FILE
    )

    print("\nBest model saved:", MODEL_FILE)

    # --------------------------------
    # STEP 6: SAVE MODEL COMPARISON
    # --------------------------------

    comparison_df = pd.DataFrame(
        final_test_results
    ).T

    comparison_df.index.name = "Model"

    Path(COMPARISON_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison_df.to_csv(
        COMPARISON_FILE
    )

    # --------------------------------
    # STEP 7: SAVE METRICS
    # --------------------------------

    metrics_data = {

        "best_model": best_model_name,

        "best_model_settings": models[best_model_name][0],

        "selection_basis": (
            "Previously completed validation experiments"
        ),

        "previous_best_validation_pr_auc": 0.0232,

        "evaluation_sample_size": len(X_test_eval),

        "best_settings": {
            name: settings
            for name, (settings, classifier) in models.items()
        },

        "test_metrics": final_test_results,

        "metrics": final_test_results
    }

    Path(METRICS_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(METRICS_FILE, "w") as file:

        json.dump(
            metrics_data,
            file,
            indent=4
        )

    # --------------------------------
    # STEP 8: SAVE BEST MODEL PREDICTIONS
    # --------------------------------

    prediction_df = pd.DataFrame({

        "actual": np.asarray(y_test_eval),

        "predicted": best_predictions,

        "fraud_probability": best_probabilities

    })

    Path(PREDICTIONS_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    # --------------------------------
    # STEP 9: FINAL SUMMARY
    # --------------------------------

    print("\n" + "=" * 65)

    print("FINAL MODEL TRAINING COMPLETED")

    print("=" * 65)

    print("\nSelected best model:", best_model_name)

    print("Selected settings:", models[best_model_name][0])

    print("\nFinal model comparison:")

    print(comparison_df.round(4))

    print("\nGenerated files:")

    print("1. Model:", MODEL_FILE)

    print("2. Metrics:", METRICS_FILE)

    print("3. Comparison:", COMPARISON_FILE)

    print("4. Predictions:", PREDICTIONS_FILE)


# ----------------------------------------
# PROGRAM ENTRY POINT
# ----------------------------------------

if __name__ == "__main__":

    main()