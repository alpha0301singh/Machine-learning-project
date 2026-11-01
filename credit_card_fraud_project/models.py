"""STEP 3 - All the models (Units III and IV) and the metric functions.

This file only DEFINES things; train.py runs them. Every classifier sits in the same
pipeline: impute -> log/scale -> select features -> model, so the comparison is fair and
nothing from the test set leaks into training.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (ElasticNetCV, LassoCV, LinearRegression, LogisticRegression, RidgeCV)
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_recall_curve,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import SEED, SELECT_K, V_COLS


# ------------------------------------------------------------------ preprocessing
def build_preprocessor():
    """Median-fill missing values, log-transform Amount (it is very skewed), scale everything."""
    amount = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                       ("sc", StandardScaler())])
    v = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    return ColumnTransformer([("amt", amount, ["Amount"]), ("v", v, V_COLS)])


def build_pipeline(model, k=SELECT_K):
    return Pipeline([("prep", build_preprocessor()), ("select", SelectKBest(f_classif, k=k)), ("clf", model)])


# ------------------------------------------------------------ classification (Unit IV)
def get_classifiers():
    """class_weight='balanced' = 'a missed fraud counts more than a false alarm'."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5,
                                                class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=2,
                                                class_weight="balanced_subsample", n_jobs=-1,
                                                random_state=SEED),
        "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
        "Naive Bayes": GaussianNB(),
    }


RF_GRID = {"clf__max_depth": [8, 12], "clf__min_samples_leaf": [1, 3], "select__k": [SELECT_K, "all"]}


def tuned_rf_pipeline():
    """Un-tuned Random Forest pipeline: the starting point of the grid search."""
    return build_pipeline(RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample",
                                                 n_jobs=-1, random_state=SEED))


def rebuild_best(name, params=None):
    """Re-create the winning (unfitted) pipeline from its name + tuned parameters.
    The app uses this to retrain if the saved model file cannot be loaded."""
    if name.endswith("(tuned)"):
        return tuned_rf_pipeline().set_params(**(params or {}))
    return build_pipeline(get_classifiers()[name])


# ----------------------------------------------------------------- regression (Unit III)
def get_regressors(features, best_single):
    scaled = lambda m: Pipeline([("sc", StandardScaler()), ("m", m)])  # noqa: E731
    return [
        (f"Simple Linear [{best_single}]", scaled(LinearRegression()), [best_single]),
        ("Multiple Linear", scaled(LinearRegression()), features),
        ("Polynomial (deg 2)", Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                                         ("sc", StandardScaler()), ("m", LinearRegression())]), features),
        ("Ridge (CV alpha)", scaled(RidgeCV(alphas=np.logspace(-2, 3, 20))), features),
        ("Lasso (CV alpha)", scaled(LassoCV(cv=3, n_alphas=30, max_iter=5000, random_state=SEED)), features),
        ("Elastic Net (CV)", scaled(ElasticNetCV(l1_ratio=[.2, .5, .8], cv=3, n_alphas=30,
                                                 max_iter=5000, random_state=SEED)), features),
    ]


# ------------------------------------------------------------------------- metrics
def get_scores(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)


def _thin(a, b, n=200):
    idx = np.linspace(0, len(a) - 1, min(n, len(a))).astype(int)
    return a[idx].round(4).tolist(), b[idx].round(4).tolist()


def classification_metrics(y, pred, score):
    fpr, tpr, _ = roc_curve(y, score)
    prec, rec, _ = precision_recall_curve(y, score)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score),
            "confusion": confusion_matrix(y, pred).tolist(),          # [[TN, FP], [FN, TP]]
            "roc": dict(zip(("fpr", "tpr"), _thin(fpr, tpr))),
            "pr": dict(zip(("precision", "recall"), _thin(prec, rec)))}


def regression_metrics(y_test, pred, y_train, pred_train):
    mse = mean_squared_error(y_test, pred)
    return {"MAE": mean_absolute_error(y_test, pred), "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "R2": r2_score(y_test, pred), "R2_train": r2_score(y_train, pred_train)}
