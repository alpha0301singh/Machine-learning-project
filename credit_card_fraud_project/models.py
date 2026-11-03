
import numpy as np

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    PolynomialFeatures
)
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    RidgeCV,
    LassoCV,
    ElasticNetCV
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import SEED, SELECT_K, RF_ESTIMATORS


# ---------------- PREPROCESSING ----------------

def build_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            make_column_selector(dtype_include=np.number)
        ),
        (
            "categorical",
            categorical_pipeline,
            make_column_selector(dtype_exclude=np.number)
        )
    ])

    return preprocessor


# ---------------- CLASSIFICATION PIPELINE ----------------

def build_pipeline(model, k=SELECT_K):

    return Pipeline([
        ("prep", build_preprocessor()),
        ("select", SelectKBest(score_func=f_classif, k=k)),
        ("clf", model)
    ])


# ---------------- CLASSIFICATION MODELS ----------------

def get_classifiers():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            n_jobs=-1
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=SEED
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=RF_ESTIMATORS,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED
        ),

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=SEED
        ),

        "Naive Bayes": GaussianNB()
    }


# ---------------- RANDOM FOREST TUNING ----------------

RF_GRID = {
    "clf__max_depth": [8, 12],
    "clf__min_samples_leaf": [1, 3],
    "select__k": [SELECT_K, "all"]
}


def tuned_rf_pipeline():

    return build_pipeline(
        RandomForestClassifier(
            n_estimators=RF_ESTIMATORS,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED
        )
    )


def rebuild_best(name, params=None):

    if name == "Random Forest (tuned)":
        model = tuned_rf_pipeline()

        if params:
            model.set_params(**params)

        return model

    classifiers = get_classifiers()

    if name not in classifiers:
        raise ValueError(f"Unknown model: {name}")

    return build_pipeline(classifiers[name])


# ---------------- REGRESSION MODELS ----------------

def get_regressors(features, best_single):

    models = []

    # Simple Linear Regression
    models.append((
        "Simple Linear Regression",
        Pipeline([
            ("prep", build_preprocessor()),
            ("m", LinearRegression())
        ]),
        [best_single]
    ))

    # Multiple Linear Regression
    models.append((
        "Multiple Linear Regression",
        Pipeline([
            ("prep", build_preprocessor()),
            ("m", LinearRegression())
        ]),
        features
    ))

    # Polynomial Regression
    models.append((
        "Polynomial Regression",
        Pipeline([
            ("prep", build_preprocessor()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("m", LinearRegression())
        ]),
        [best_single]
    ))

    # Ridge Regression
    models.append((
        "Ridge Regression",
        Pipeline([
            ("prep", build_preprocessor()),
            ("m", RidgeCV(alphas=[0.1, 1.0, 10.0]))
        ]),
        features
    ))

    # Lasso Regression
    models.append((
        "Lasso Regression",
        Pipeline([
            ("prep", build_preprocessor()),
            ("m", LassoCV(cv=3, max_iter=10000, random_state=SEED))
        ]),
        features
    ))

    # Elastic Net Regression
    models.append((
        "Elastic Net",
        Pipeline([
            ("prep", build_preprocessor()),
            ("m", ElasticNetCV(cv=3, max_iter=10000, random_state=SEED))
        ]),
        features
    ))

    return models


# ---------------- PREDICTION SCORES ----------------

def get_scores(model, X):

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    return model.decision_function(X)


# ---------------- CLASSIFICATION METRICS ----------------

def thin_data(x, y, maximum=200):

    if len(x) <= maximum:
        return x.tolist(), y.tolist()

    indexes = np.linspace(
        0,
        len(x) - 1,
        maximum
    ).astype(int)

    return x[indexes].tolist(), y[indexes].tolist()


def classification_metrics(y, predictions, scores):

    y = np.asarray(y)
    predictions = np.asarray(predictions)
    scores = np.asarray(scores)

    fpr, tpr, _ = roc_curve(y, scores)
    precision, recall, _ = precision_recall_curve(y, scores)

    fpr, tpr = thin_data(fpr, tpr)
    precision, recall = thin_data(precision, recall)

    cm = confusion_matrix(y, predictions, labels=[0, 1])

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "confusion": cm.tolist(),
        "roc": {"fpr": fpr, "tpr": tpr},
        "pr": {"precision": precision, "recall": recall}
    }


# ---------------- REGRESSION METRICS ----------------

def regression_metrics(y_test, predictions, y_train, train_predictions):

    mse = mean_squared_error(y_test, predictions)

    return {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_test, predictions)),
        "R2_train": float(r2_score(y_train, train_predictions))
    }
