
"""Main training file for Credit Card Fraud Detection.

Run:
    python train.py
    python train.py --data data/creditcard_2023.csv
    python train.py --demo
"""

import argparse
import json
import time
import platform
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.pipeline import Pipeline

import eda
import explain
import model as models
import plots

from config import (
    ARTIFACTS,
    REPORTS,
    DATA_FILE,
    FEATURES,
    TARGET,
    V_COLS,
    SEED,
    N_LEGIT,
    FRAUD_RATE,
    CV_FOLDS
)

from data_prep import prepare


# ------------------------- SAVE CHARTS -------------------------

def save_chart(fig, filename):
    REPORTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        REPORTS / filename,
        dpi=150,
        bbox_inches="tight"
    )


# ------------------------- IMBALANCE STUDY -------------------------

def imbalance_study(X_train, y_train, X_test, y_test):

    def train_and_evaluate(class_weight, X, y):

        pipeline = Pipeline([
            ("prep", models.build_preprocessor()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight=class_weight
            ))
        ])

        pipeline.fit(X, y)

        predictions = pipeline.predict(X_test)
        scores = pipeline.predict_proba(X_test)[:, 1]

        return {
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "pr_auc": average_precision_score(y_test, scores)
        }

    # Select all fraud transactions
    fraud_rows = y_train[y_train == 1].index

    # Select three legitimate transactions for every fraud transaction
    legitimate_rows = y_train[y_train == 0].sample(
        n=min(3 * len(fraud_rows), (y_train == 0).sum()),
        random_state=SEED
    ).index

    selected_rows = fraud_rows.union(legitimate_rows)

    results = {
        "No handling": train_and_evaluate(
            None, X_train, y_train
        ),

        "Class weights": train_and_evaluate(
            "balanced", X_train, y_train
        ),

        "Under-sampling (1:3)": train_and_evaluate(
            None,
            X_train.loc[selected_rows],
            y_train.loc[selected_rows]
        )
    }

    return results


# ------------------------- REGRESSION STUDY -------------------------

def regression_study(train, test):

    results = {}

    # Predict Amount and Class separately
    for target in ["Amount", "Class"]:

        if target == "Amount":
            features = V_COLS
        else:
            features = FEATURES

        X_train = train[features].copy()
        X_test = test[features].copy()

        y_train = train[target]
        y_test = test[target]

        # Fill missing values using training medians
        medians = X_train.median()

        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)

        # Find the strongest single feature
        best_feature = X_train.corrwith(y_train).abs().idxmax()

        target_results = {}

        regression_models = models.get_regressors(
            features,
            best_feature
        )

        for name, pipeline, selected_features in regression_models:

            pipeline.fit(
                X_train[selected_features],
                y_train
            )

            predictions = pipeline.predict(
                X_test[selected_features]
            )

            train_predictions = pipeline.predict(
                X_train[selected_features]
            )

            metrics = models.regression_metrics(
                y_test,
                predictions,
                y_train,
                train_predictions
            )

            # Read the final estimator in the pipeline
            estimator = pipeline.steps[-1][1]

            if hasattr(estimator, "alpha_"):
                metrics["alpha"] = float(estimator.alpha_)

            if hasattr(estimator, "coef_"):
                metrics["nonzero_coefs"] = int(
                    (np.abs(estimator.coef_) > 1e-8).sum()
                )

            target_results[name] = metrics

        results[target] = target_results

        # Save regression chart
        chart_data = pd.DataFrame(target_results).T

        save_chart(
            plots.regression_comparison(chart_data, target),
            f"regression_{target}.png"
        )

    return results


# ------------------------- CLASSIFICATION STUDY -------------------------

def classification_study(X_train, y_train, X_test, y_test):

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    results = {}
    trained_models = {}

    # Train and compare all classifiers
    for name, classifier in models.get_classifiers().items():

        start_time = time.time()

        pipeline = models.build_pipeline(classifier)

        # Cross-validation using PR-AUC
        cv_score = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="average_precision"
        ).mean()

        # Train on complete training data
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        scores = models.get_scores(pipeline, X_test)

        metrics = models.classification_metrics(
            y_test,
            predictions,
            scores
        )

        metrics["cv_pr_auc"] = float(cv_score)

        results[name] = metrics
        trained_models[name] = pipeline

        print(
            f"{name}: "
            f"CV PR-AUC = {cv_score:.3f}, "
            f"Test PR-AUC = {metrics['pr_auc']:.3f}, "
            f"Time = {time.time() - start_time:.1f}s"
        )

    # Tune Random Forest
    print("\nTuning Random Forest...")

    grid_search = GridSearchCV(
        models.tuned_rf_pipeline(),
        models.RF_GRID,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_

    predictions = best_rf.predict(X_test)
    scores = models.get_scores(best_rf, X_test)

    rf_metrics = models.classification_metrics(
        y_test,
        predictions,
        scores
    )

    rf_metrics["cv_pr_auc"] = float(grid_search.best_score_)

    results["Random Forest (tuned)"] = rf_metrics
    trained_models["Random Forest (tuned)"] = best_rf

    tuning_results = {
        "best_params": grid_search.best_params_,
        "best_cv_pr_auc": float(grid_search.best_score_)
    }

    # Select the best model using cross-validation, not test performance
    eligible_models = {
        name: metrics["cv_pr_auc"]
        for name, metrics in results.items()
        if hasattr(trained_models[name], "predict_proba")
    }

    best_model_name = max(
        eligible_models,
        key=eligible_models.get
    )

    return results, trained_models, tuning_results, best_model_name


# ------------------------- MAIN FUNCTION -------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection Training"
    )

    parser.add_argument(
        "--data",
        default=str(DATA_FILE),
        help="Path to the dataset"
    )

    parser.add_argument(
        "--n_legit",
        type=int,
        default=N_LEGIT
    )

    parser.add_argument(
        "--fraud_rate",
        type=float,
        default=FRAUD_RATE
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use synthetic demo data"
    )

    args = parser.parse_args()

    start_time = time.time()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Use synthetic data if demo mode is selected
    if args.demo:
        from make_demo_data import make
        args.data = str(make())
        data_source = "synthetic demo data"
    else:
        data_source = str(args.data)

    # STEP 1: DATA PREPARATION
    print("\nSTEP 1: Data Preparation")

    train, test, info = prepare(
        args.data,
        args.n_legit,
        args.fraud_rate
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # STEP 2: EXPLORATORY DATA ANALYSIS
    print("\nSTEP 2: Exploratory Data Analysis")

    eda_results = eda.compute_eda(train)

    eda.print_summary(eda_results)
    eda.save_eda_charts(train, eda_results)

    # STEP 3: IMBALANCE STUDY
    print("\nSTEP 3: Imbalance Study")

    imbalance_results = imbalance_study(
        X_train,
        y_train,
        X_test,
        y_test
    )

    imbalance_table = pd.DataFrame(imbalance_results).T

    print(imbalance_table.round(3))

    save_chart(
        plots.imbalance_comparison(imbalance_table),
        "imbalance.png"
    )

    # STEP 4: REGRESSION
    print("\nSTEP 4: Regression Study")

    regression_results = regression_study(train, test)

    for target, values in regression_results.items():
        best_r2 = max(
            result["R2"] for result in values.values()
        )

        print(f"{target}: Best test R2 = {best_r2:.3f}")

    # STEP 5: CLASSIFICATION
    print("\nSTEP 5: Classification Study")

    classification_results, trained_models, tuning_results, best_name = (
        classification_study(
            X_train,
            y_train,
            X_test,
            y_test
        )
    )

    print("\nSelected model:", best_name)

    # Save classification charts
    classification_table = pd.DataFrame(
        classification_results
    ).T

    save_chart(
        plots.model_comparison(classification_table),
        "model_comparison.png"
    )

    save_chart(
        plots.roc_curves(classification_results),
        "roc_curves.png"
    )

    save_chart(
        plots.pr_curves(classification_results),
        "pr_curves.png"
    )

    save_chart(
        plots.confusion(
            classification_results[best_name]["confusion"],
            f"Confusion Matrix: {best_name}"
        ),
        "confusion_best.png"
    )

    # STEP 6: EXPLAINABILITY
    print("\nSTEP 6: Explainability")

    best_model = trained_models[best_name]

    importance = explain.permutation_importances(
        best_model,
        X_test,
        y_test
    )

    save_chart(
        plots.permutation_importance(importance),
        "permutation_importance.png"
    )

    # Save the trained model
    joblib.dump(
        best_model,
        ARTIFACTS / "best_model.joblib",
        compress=3
    )

    # Save actual labels and predicted probabilities
    test_scores = pd.DataFrame({
        "y_true": y_test.values,
        "score": best_model.predict_proba(X_test)[:, 1]
    })

    test_scores.to_csv(
        ARTIFACTS / "test_scores.csv",
        index=False
    )

    # Prepare information for Streamlit
    metadata = {
        "data_source": data_source,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "cv_folds": CV_FOLDS,
        **{
            key: value
            for key, value in info.items()
            if key != "cleaning"
        }
    }

    output = {
        "meta": metadata,
        "cleaning": info["cleaning"],
        "eda": eda_results,
        "imbalance": imbalance_results,
        "regression": regression_results,
        "classification": classification_results,
        "tuning": tuning_results,
        "best_model": best_name,
        "permutation": importance,
        "feature_stats": {
            "median": X_train.median().to_dict(),
            "q01": X_train.quantile(0.01).to_dict(),
            "q99": X_train.quantile(0.99).to_dict()
        }
    }

    # Save results as JSON
    with open(ARTIFACTS / "results.json", "w") as file:
        json.dump(
            output,
            file,
            indent=4,
            default=lambda value: (
                value.item()
                if hasattr(value, "item")
                else str(value)
            )
        )

    print("\nTraining completed successfully.")
    print(f"Time taken: {time.time() - start_time:.1f} seconds")
    print("Run the application using: streamlit run app.py")


if __name__ == "__main__":
    main()
