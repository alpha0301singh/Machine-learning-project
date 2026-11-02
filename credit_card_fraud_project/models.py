
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    RidgeCV,
    LassoCV,
    ElasticNetCV
)

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

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    FunctionTransformer,
    PolynomialFeatures
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import SEED, SELECT_K, V_COLS


# Step 1: Prepare Data

def build_preprocessor():

    # Handle missing values and scale Amount
    amount_pipeline = Pipeline([

        ("imputer", SimpleImputer(strategy="median")),

        ("log", FunctionTransformer(np.log1p)),

        ("scaler", StandardScaler())

    ])

    # Handle missing values and scale V1 to V28
    feature_pipeline = Pipeline([

        ("imputer", SimpleImputer(strategy="median")),

        ("scaler", StandardScaler())

    ])

    # Apply different preprocessing to different columns
    preprocessor = ColumnTransformer([

        ("amount", amount_pipeline, ["Amount"]),

        ("features", feature_pipeline, V_COLS)

    ])

    return preprocessor


# Step 2: Create Machine Learning Pipeline

def build_pipeline(model, k=SELECT_K):

    pipeline = Pipeline([

        ("preprocessing", build_preprocessor()),

        ("feature_selection", SelectKBest(score_func=f_classif, k=k)),

        ("model", model)

    ])

    return pipeline


# Step 3: Create Classification Models

def get_classifiers():

    models = {

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
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED
        ),

        "SVM (RBF)": SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=SEED
        ),

        "Naive Bayes": GaussianNB()

    }

    return models


# Step 4: Random Forest Parameter Settings

RF_GRID = {

    "model__max_depth": [8, 12],

    "model__min_samples_leaf": [1, 3],

    "feature_selection__k": [SELECT_K, "all"]

}


# Step 5: Create Random Forest Model for Tuning

def tuned_rf_pipeline():

    model = RandomForestClassifier(

        n_estimators=150,

        class_weight="balanced_subsample",

        n_jobs=-1,

        random_state=SEED

    )

    return build_pipeline(model)


# Step 6: Rebuild Best Model

def rebuild_best(name, params=None):

    # Rebuild tuned Random Forest
    if name.endswith("(tuned)"):

        model = tuned_rf_pipeline()

        model.set_params(**(params or {}))

        return model

    # Rebuild another classification model
    models = get_classifiers()

    return build_pipeline(models[name])


# Step 7: Create Regression Models

def get_regressors(features, best_single):

    models = []

    # Simple Linear Regression
    simple_model = Pipeline([

        ("scaler", StandardScaler()),

        ("model", LinearRegression())

    ])

    models.append((
        f"Simple Linear [{best_single}]",
        simple_model,
        [best_single]
    ))

    # Multiple Linear Regression
    multiple_model = Pipeline([

        ("scaler", StandardScaler()),

        ("model", LinearRegression())

    ])

    models.append((
        "Multiple Linear",
        multiple_model,
        features
    ))

    # Polynomial Regression
    polynomial_model = Pipeline([

        ("polynomial", PolynomialFeatures(
            degree=2,
            include_bias=False
        )),

        ("scaler", StandardScaler()),

        ("model", LinearRegression())

    ])

    models.append((
        "Polynomial (deg 2)",
        polynomial_model,
        features
    ))

    # Ridge Regression
    ridge_model = Pipeline([

        ("scaler", StandardScaler()),

        ("model", RidgeCV(
            alphas=np.logspace(-2, 3, 20)
        ))

    ])

    models.append((
        "Ridge (CV alpha)",
        ridge_model,
        features
    ))

    # Lasso Regression
    lasso_model = Pipeline([

        ("scaler", StandardScaler()),

        ("model", LassoCV(
            cv=3,
            n_alphas=30,
            max_iter=5000,
            random_state=SEED
        ))

    ])

    models.append((
        "Lasso (CV alpha)",
        lasso_model,
        features
    ))

    # Elastic Net Regression
    elastic_model = Pipeline([

        ("scaler", StandardScaler()),

        ("model", ElasticNetCV(
            l1_ratio=[0.2, 0.5, 0.8],
            cv=3,
            n_alphas=30,
            max_iter=5000,
            random_state=SEED
        ))

    ])

    models.append((
        "Elastic Net (CV)",
        elastic_model,
        features
    ))

    return models


# Step 8: Get Model Predictions

def get_scores(model, X):

    # Get fraud probabilities if available
    if hasattr(model, "predict_proba"):

        scores = model.predict_proba(X)[:, 1]

    else:

        # Get decision scores for models without predict_proba
        scores = model.decision_function(X)

    return scores


# Step 9: Reduce Graph Data Points

def reduce_points(x_values, y_values, n=200):

    # Select a maximum of 200 points
    indexes = np.linspace(
        0,
        len(x_values) - 1,
        min(n, len(x_values))
    ).astype(int)

    x_values = x_values[indexes].round(4).tolist()

    y_values = y_values[indexes].round(4).tolist()

    return x_values, y_values


# Step 10: Calculate Classification Performance

def classification_metrics(y, predictions, scores):

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y, scores)

    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y, scores)

    # Calculate confusion matrix
    matrix = confusion_matrix(y, predictions)

    results = {

        "accuracy": accuracy_score(y, predictions),

        "precision": precision_score(
            y, predictions, zero_division=0
        ),

        "recall": recall_score(
            y, predictions, zero_division=0
        ),

        "f1": f1_score(
            y, predictions, zero_division=0
        ),

        "roc_auc": roc_auc_score(y, scores),

        "pr_auc": average_precision_score(y, scores),

        "confusion": matrix.tolist(),

        "roc": {

            "fpr": reduce_points(fpr, tpr)[0],

            "tpr": reduce_points(fpr, tpr)[1]

        },

        "pr": {

            "precision": reduce_points(precision, recall)[0],

            "recall": reduce_points(precision, recall)[1]

        }

    }

    return results


# Step 11: Calculate Regression Performance

def regression_metrics(y_test, predictions, y_train, train_predictions):

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, predictions)

    results = {

        "MAE": mean_absolute_error(y_test, predictions),

        "MSE": mse,

        "RMSE": float(np.sqrt(mse)),

        "R2": r2_score(y_test, predictions),

        "R2_train": r2_score(y_train, train_predictions)

    }

    return results
