
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    SEED
)


def build_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=20,
                sparse_output=False,
                dtype=np.float32
            )
        )
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        sparse_threshold=0.0
    )

    return preprocessor


def get_model_experiments():

    experiments = {

        # 1. Logistic Regression
        "Logistic Regression": [

            (
                "C=0.1, balanced",
                LogisticRegression(
                    C=0.1,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=SEED
                )
            ),

            (
                "C=0.5, balanced",
                LogisticRegression(
                    C=0.5,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=SEED
                )
            ),

            (
                "C=1.0, balanced",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=SEED
                )
            ),

            (
                "C=2.0, balanced",
                LogisticRegression(
                    C=2.0,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=SEED
                )
            ),

            (
                "C=0.5, weight 1:3",
                LogisticRegression(
                    C=0.5,
                    max_iter=2000,
                    class_weight={0: 1, 1: 3},
                    solver="lbfgs",
                    random_state=SEED
                )
            ),

            (
                "C=0.5, weight 1:5",
                LogisticRegression(
                    C=0.5,
                    max_iter=2000,
                    class_weight={0: 1, 1: 5},
                    solver="lbfgs",
                    random_state=SEED
                )
            )
        ],

        # 2. KNN
        "KNN": [

            (
                "K=5, uniform",
                KNeighborsClassifier(
                    n_neighbors=5,
                    weights="uniform",
                    p=2,
                    n_jobs=-1
                )
            ),

            (
                "K=5, distance",
                KNeighborsClassifier(
                    n_neighbors=5,
                    weights="distance",
                    p=2,
                    n_jobs=-1
                )
            ),

            (
                "K=11, distance",
                KNeighborsClassifier(
                    n_neighbors=11,
                    weights="distance",
                    p=2,
                    n_jobs=-1
                )
            ),

            (
                "K=21, distance",
                KNeighborsClassifier(
                    n_neighbors=21,
                    weights="distance",
                    p=2,
                    n_jobs=-1
                )
            ),

            (
                "K=11, Manhattan",
                KNeighborsClassifier(
                    n_neighbors=11,
                    weights="distance",
                    p=1,
                    n_jobs=-1
                )
            ),

            (
                "K=21, Manhattan",
                KNeighborsClassifier(
                    n_neighbors=21,
                    weights="distance",
                    p=1,
                    n_jobs=-1
                )
            )
        ],

        # 3. Decision Tree
        "Decision Tree": [

            (
                "Depth=5, leaf=10",
                DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=SEED
                )
            ),

            (
                "Depth=10, leaf=10",
                DecisionTreeClassifier(
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=SEED
                )
            ),

            (
                "Depth=15, leaf=10",
                DecisionTreeClassifier(
                    max_depth=15,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=SEED
                )
            ),

            (
                "Depth=20, leaf=20",
                DecisionTreeClassifier(
                    max_depth=20,
                    min_samples_split=40,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=SEED
                )
            ),

            (
                "Depth=10, leaf=20",
                DecisionTreeClassifier(
                    max_depth=10,
                    min_samples_split=40,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=SEED
                )
            )
        ],

        # 4. Random Forest
        "Random Forest": [

            (
                "100 trees, depth=10",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=SEED
                )
            ),

            (
                "200 trees, depth=10",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=SEED
                )
            ),

            (
                "100 trees, depth=15",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=SEED
                )
            ),

            (
                "200 trees, depth=15",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=SEED
                )
            )
        ],

        # 5. SVM
        "SVM": [

            (
                "C=0.1, RBF",
                SVC(
                    C=0.1,
                    kernel="rbf",
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=SEED
                )
            ),

            (
                "C=1.0, RBF",
                SVC(
                    C=1.0,
                    kernel="rbf",
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=SEED
                )
            ),

            (
                "C=5.0, RBF",
                SVC(
                    C=5.0,
                    kernel="rbf",
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=SEED
                )
            )
        ],

        # 6. Naive Bayes
        "Naive Bayes": [

            (
                "Smoothing=1e-10",
                GaussianNB(var_smoothing=1e-10)
            ),

            (
                "Smoothing=1e-9",
                GaussianNB(var_smoothing=1e-9)
            ),

            (
                "Smoothing=1e-8",
                GaussianNB(var_smoothing=1e-8)
            ),

            (
                "Smoothing=1e-7",
                GaussianNB(var_smoothing=1e-7)
            )
        ]
    }

    return experiments


# Keep this function for compatibility with existing imports.
def get_models():

    experiments = get_model_experiments()

    models = {}

    for model_name, candidates in experiments.items():

        models[model_name] = candidates[0][1]

    return models


def build_pipeline(model):

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", model)
    ])

    return pipeline
