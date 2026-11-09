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
    get_model_experiments,
    build_pipeline
)


DISTANCE_BASED_MODELS = [
    "KNN",
    "SVM",
    "Naive Bayes"
]


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


def get_training_data(model_name, X, y):

    if model_name in DISTANCE_BASED_MODELS:

        return make_balanced_sample(X, y)

    return X, y


def main():

    print("\n" + "=" * 65)

    print("FRAUD DETECTION - SECOND HYPERPARAMETER EXPERIMENT")

    print("=" * 65)

    # --------------------------------
    # STEP 1: LOAD DATA
    # --------------------------------

    print("\nLoading dataset...")

    X_train, X_test, y_train, y_test = prepare_data()

    print("Training data:", X_train.shape)

    print("Testing data:", X_test.shape)

    # --------------------------------
    # STEP 2: VALIDATION SPLIT
    # --------------------------------

    print("\nCreating validation data...")

    X_fit_base, X_validation, y_fit_base, y_validation = (
        train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            stratify=y_train,
            random_state=SEED
        )
    )

    X_validation_eval, y_validation_eval = make_evaluation_sample(
        X_validation,
        y_validation
    )

    print("Training portion:", X_fit_base.shape)

    print("Validation sample:", X_validation_eval.shape)

    # --------------------------------
    # STEP 3: TEST DATA
    # --------------------------------

    X_test_eval, y_test_eval = make_evaluation_sample(
        X_test,
        y_test
    )

    print("Final test sample:", X_test_eval.shape)

    # --------------------------------
    # STEP 4: LOAD EXPERIMENTS
    # --------------------------------

    experiments = get_model_experiments()

    all_results = []

    best_settings = {}

    validation_metrics = {}

    # --------------------------------
    # STEP 5: EXPERIMENT WITH PARAMETERS
    # --------------------------------

    total_experiments = sum(
        len(candidates)
        for candidates in experiments.values()
    )

    experiment_number = 0

    print("\nTotal experiments:", total_experiments)

    for model_name, candidates in experiments.items():

        print("\n" + "=" * 65)

        print("MODEL:", model_name)

        print("=" * 65)

        model_results = []

        for setting_name, classifier in candidates:

            experiment_number += 1

            print("\nExperiment:", experiment_number, "/", total_experiments)

            print("Model:", model_name)

            print("Settings:", setting_name)

            X_fit, y_fit = get_training_data(
                model_name,
                X_fit_base,
                y_fit_base
            )

            print("Training sample:", X_fit.shape)

            pipeline = build_pipeline(
                clone(classifier)
            )

            pipeline.fit(
                X_fit,
                y_fit
            )

            results, _, _ = evaluate_model(
                pipeline,
                X_validation_eval,
                y_validation_eval
            )

            row = {

                "Model": model_name,

                "Settings": setting_name,

                **results

            }

            model_results.append(row)

            all_results.append(row)

            print("\nValidation metrics:")

            for metric, value in results.items():

                print(f"{metric}: {value:.4f}")

            del pipeline
            del X_fit
            del y_fit

            gc.collect()

        # Choose settings using validation PR-AUC
        best_result = max(
            model_results,
            key=lambda item: item["pr_auc"]
        )

        best_settings[model_name] = best_result["Settings"]

        validation_metrics[model_name] = {
            metric: best_result[metric]
            for metric in [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "pr_auc"
            ]
        }

        print("\nBest configuration:", best_settings[model_name])

        print(
            "Best validation PR-AUC:",
            round(best_result["pr_auc"], 4)
        )

    # --------------------------------
    # STEP 6: SAVE EXPERIMENT RESULTS
    # --------------------------------

    experiment_df = pd.DataFrame(all_results)

    experiment_file = Path(COMPARISON_FILE).with_name(
        "hyperparameter_experiments.csv"
    )

    experiment_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    experiment_df.to_csv(
        experiment_file,
        index=False
    )

    print("\nAll experiment results saved:", experiment_file)

    # --------------------------------
    # STEP 7: SELECT BEST MODEL
    # --------------------------------

    best_model_name = max(
        validation_metrics,
        key=lambda name: validation_metrics[name]["pr_auc"]
    )

    print("\n" + "=" * 65)

    print("BEST MODEL SELECTED")

    print("=" * 65)

    print("Best model:", best_model_name)

    print(
        "Validation PR-AUC:",
        round(
            validation_metrics[best_model_name]["pr_auc"],
            4
        )
    )

    # --------------------------------
    # STEP 8: FINAL TRAINING
    # --------------------------------

    print("\nRetraining selected configurations...")

    final_test_results = {}

    best_model_pipeline = None

    best_predictions = None

    best_probabilities = None

    for model_name, candidates in experiments.items():

        print("\nTraining final model:", model_name)

        selected_setting = best_settings[model_name]

        selected_classifier = None

        for setting_name, classifier in candidates:

            if setting_name == selected_setting:

                selected_classifier = classifier

                break

        if selected_classifier is None:

            raise ValueError(
                "Selected configuration not found."
            )

        X_fit, y_fit = get_training_data(
            model_name,
            X_train,
            y_train
        )

        pipeline = build_pipeline(
            clone(selected_classifier)
        )

        pipeline.fit(
            X_fit,
            y_fit
        )

        results, predictions, probabilities = evaluate_model(
            pipeline,
            X_test_eval,
            y_test_eval
        )

        final_test_results[model_name] = results

        print("\nFinal test metrics:")

        for metric, value in results.items():

            print(f"{metric}: {value:.4f}")

        if model_name == best_model_name:

            best_model_pipeline = pipeline

            best_predictions = predictions

            best_probabilities = probabilities

        else:

            del pipeline

        del X_fit
        del y_fit

        gc.collect()

    # --------------------------------
    # STEP 9: SAVE BEST MODEL
    # --------------------------------

    if best_model_pipeline is None:

        raise RuntimeError("Best model was not trained.")

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
    # STEP 10: SAVE COMPARISON
    # --------------------------------

    comparison_df = pd.DataFrame(
        final_test_results
    ).T

    comparison_df.index.name = "Model"

    comparison_df.to_csv(
        COMPARISON_FILE
    )

    # --------------------------------
    # STEP 11: SAVE METRICS
    # --------------------------------

    metrics_data = {

        "best_model": best_model_name,

        "best_model_settings": best_settings[best_model_name],

        "selection_metric": "validation_pr_auc",

        "validation_sample_size": len(X_validation_eval),

        "evaluation_sample_size": len(X_test_eval),

        "best_settings": best_settings,

        "validation_metrics": validation_metrics,

        "test_metrics": final_test_results,

        "metrics": final_test_results

    }

    with open(METRICS_FILE, "w") as file:

        json.dump(
            metrics_data,
            file,
            indent=4
        )

    # --------------------------------
    # STEP 12: SAVE PREDICTIONS
    # --------------------------------

    prediction_df = pd.DataFrame({

        "actual": np.asarray(y_test_eval),

        "predicted": best_predictions,

        "fraud_probability": best_probabilities

    })

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    # --------------------------------
    # STEP 13: FINAL SUMMARY
    # --------------------------------

    print("\n" + "=" * 65)

    print("SECOND EXPERIMENT COMPLETED")

    print("=" * 65)

    print("\nBest model:", best_model_name)

    print("Best settings:", best_settings[best_model_name])

    print("\nFinal test comparison:")

    print(comparison_df.round(4))

    print("\nGenerated files:")

    print("1.", experiment_file)

    print("2.", COMPARISON_FILE)

    print("3.", METRICS_FILE)

    print("4.", MODEL_FILE)

    print("5.", PREDICTIONS_FILE)


if _name_ == "_main_":

    main()