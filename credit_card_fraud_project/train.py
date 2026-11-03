
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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

import data_prep
import eda
import explain
import models
import plots

from config import (
    ARTIFACTS, REPORTS, DATA_FILE,
    FEATURES, TARGET, V_COLS,
    SEED, N_LEGIT, FRAUD_RATE, CV_FOLDS
)


# ---------------- SAVE CHARTS ----------------

def save_chart(fig, filename):

    REPORTS.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        REPORTS / filename,
        dpi=150,
        bbox_inches="tight"
    )


# ---------------- IMBALANCE STUDY ----------------

def imbalance_study(X_train, y_train, X_test, y_test):

    results = {}

    # Create a 1:3 under-sampled training set
    fraud_indexes = y_train[y_train == 1].index

    normal_indexes = y_train[y_train == 0].sample(
        n=min(
            3 * len(fraud_indexes),
            (y_train == 0).sum()
        ),
        random_state=SEED
    ).index

    selected_indexes = fraud_indexes.union(normal_indexes)

    experiments = {
        "No handling": (X_train, y_train, None),
        "Class weights": (X_train, y_train, "balanced"),
        "Under-sampling (1:3)": (
            X_train.loc[selected_indexes],
            y_train.loc[selected_indexes],
            None
        )
    }

    for name, (X, y, weight) in experiments.items():

        pipeline = Pipeline([
            ("prep", models.build_preprocessor()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight=weight
            ))
        ])

        pipeline.fit(X, y)

        predictions = pipeline.predict(X_test)
        scores = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "pr_auc": float(
                average_precision_score(y_test, scores)
            )
        }

    return results


# ---------------- REGRESSION STUDY ----------------

def regression_study(train, test):

    results = {}

    for target in ["Amount", "Class"]:

        if target == "Amount":
            features = V_COLS
        else:
            features = FEATURES

        X_train = train[features].copy()
        X_test = test[features].copy()

        y_train = train[target]
        y_test = test[target]

        # Select a single feature for simple regression
        correlations = X_train.corrwith(y_train).abs()

        best_single = correlations.idxmax()

        regression_models = models.get_regressors(
            features,
            best_single
        )

        target_results = {}

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

            target_results[name] = metrics

        results[target] = target_results

        table = pd.DataFrame(target_results).T

        save_chart(
            plots.regression_comparison(table, target),
            f"regression_{target}.png"
        )

    return results


# ---------------- CLASSIFICATION STUDY ----------------

def classification_study(X_train, y_train, X_test, y_test):

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    results = {}
    trained_models = {}

    # Train all classifiers
    for name, classifier in models.get_classifiers().items():

        print(f"\nTraining {name}...")

        pipeline = models.build_pipeline(classifier)

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="average_precision"
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        scores = models.get_scores(
            pipeline,
            X_test
        )

        metrics = models.classification_metrics(
            y_test,
            predictions,
            scores
        )

        metrics["cv_pr_auc"] = float(cv_scores.mean())

        results[name] = metrics
        trained_models[name] = pipeline

        print("CV PR-AUC:", round(cv_scores.mean(), 4))
        print("Test PR-AUC:", round(metrics["pr_auc"], 4))

    # Tune Random Forest
    print("\nTuning Random Forest...")

    grid = GridSearchCV(
        models.tuned_rf_pipeline(),
        models.RF_GRID,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_

    predictions = best_rf.predict(X_test)
    scores = models.get_scores(best_rf, X_test)

    metrics = models.classification_metrics(
        y_test,
        predictions,
        scores
    )

    metrics["cv_pr_auc"] = float(grid.best_score_)

    results["Random Forest (tuned)"] = metrics

    trained_models["Random Forest (tuned)"] = best_rf

    tuning = {
        "best_params": grid.best_params_,
        "best_cv_pr_auc": float(grid.best_score_)
    }

    # Select winner using cross-validation PR-AUC
    best_name = max(
        results,
        key=lambda name: results[name]["cv_pr_auc"]
    )

    return results, trained_models, tuning, best_name


# ---------------- MAIN PROGRAM ----------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=str(DATA_FILE)
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
        action="store_true"
    )

    args = parser.parse_args()

    start_time = time.time()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Demo dataset option
    if args.demo:
        from make_demo_data import make
        args.data = str(make())
        data_source = "synthetic demo data"
    else:
        data_source = str(args.data)

    # STEP 1: DATA PREPARATION
    print("\nSTEP 1: DATA PREPARATION")

    train, test, info = data_prep.prepare(
        args.data,
        args.n_legit,
        args.fraud_rate
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # STEP 2: EDA
    print("\nSTEP 2: EXPLORATORY DATA ANALYSIS")

    eda_results = eda.compute_eda(train)

    eda.print_summary(eda_results)
    eda.save_eda_charts(train, eda_results)

    # STEP 3: IMBALANCE HANDLING
    print("\nSTEP 3: IMBALANCE STUDY")

    imbalance_results = imbalance_study(
        X_train,
        y_train,
        X_test,
        y_test
    )

    imbalance_table = pd.DataFrame(
        imbalance_results
    ).T

    save_chart(
        plots.imbalance_comparison(imbalance_table),
        "imbalance.png"
    )

    # STEP 4: REGRESSION
    print("\nSTEP 4: REGRESSION")

    regression_results = regression_study(
        train,
        test
    )

    # STEP 5: CLASSIFICATION
    print("\nSTEP 5: CLASSIFICATION")

    classification_results, trained_models, tuning, best_name = (
        classification_study(
            X_train,
            y_train,
            X_test,
            y_test
        )
    )

    print("\nBest model:", best_name)

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
            f"Best Model: {best_name}"
        ),
        "confusion_best.png"
    )

    # STEP 6: EXPLAINABILITY
    print("\nSTEP 6: EXPLAINABILITY")

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

    # Save model
    joblib.dump(
        best_model,
        ARTIFACTS / "best_model.joblib",
        compress=3
    )

    # Save test predictions
    test_scores = pd.DataFrame({
        "y_true": y_test.values,
        "score": best_model.predict_proba(X_test)[:, 1]
    })

    test_scores.to_csv(
        ARTIFACTS / "test_scores.csv",
        index=False
    )

    # Save project results
    output = {
        "meta": {
            "data_source": data_source,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "seed": SEED,
            "cv_folds": CV_FOLDS,
            **{
                key: value
                for key, value in info.items()
                if key != "cleaning"
            }
        },

        "cleaning": info["cleaning"],
        "eda": eda_results,
        "imbalance": imbalance_results,
        "regression": regression_results,
        "classification": classification_results,
        "tuning": tuning,
        "best_model": best_name,
        "permutation": importance,

        "feature_stats": {
            "median": X_train.median().to_dict(),
            "q01": X_train.quantile(0.01).to_dict(),
            "q99": X_train.quantile(0.99).to_dict()
        }
    }

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

    print("\nTraining completed")
    print("Selected model:", best_name)
    print("Time taken:", round(time.time() - start_time, 2), "seconds")
    print("Artifacts saved in:", ARTIFACTS)
    print("\nRun the app using: streamlit run app.py")


if __name__ == "__main__":
    main()
