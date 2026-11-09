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

        # --------------------------------
        # 1. LOGISTIC REGRESSION
        # --------------------------------

        "Logistic Regression": [

            ("C=0.01", LogisticRegression(
                C=0.01, max_iter=2000,
                class_weight="balanced",
                random_state=SEED
            )),

            ("C=0.05", LogisticRegression(
                C=0.05, max_iter=2000,
                class_weight="balanced",
                random_state=SEED
            )),

            ("C=0.1", LogisticRegression(
                C=0.1, max_iter=2000,
                class_weight="balanced",
                random_state=SEED
            )),

            ("C=0.5", LogisticRegression(
                C=0.5, max_iter=2000,
                class_weight="balanced",
                random_state=SEED
            )),

            ("C=1.0", LogisticRegression(
                C=1.0, max_iter=2000,
                class_weight="balanced",
                random_state=SEED
            )),

            ("C=2.0", LogisticRegression(
                C=2.0, max_iter=2000,
                class_weight="balanced",
                random_state=SEED
            )),

            ("C=5.0", LogisticRegression(
                C=5.0, max_iter=2000,
                class_weight="balanced",
                random_state=SEED
            )),

            ("C=0.5, weight 1:5", LogisticRegression(
                C=0.5, max_iter=2000,
                class_weight={0: 1, 1: 5},
                random_state=SEED
            )),

            ("C=0.5, weight 1:10", LogisticRegression(
                C=0.5, max_iter=2000,
                class_weight={0: 1, 1: 10},
                random_state=SEED
            ))
        ],

        # --------------------------------
        # 2. KNN
        # --------------------------------

        "KNN": [

            ("K=3, distance", KNeighborsClassifier(
                n_neighbors=3,
                weights="distance",
                p=2,
                n_jobs=-1
            )),

            ("K=5, distance", KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                p=2,
                n_jobs=-1
            )),

            ("K=9, distance", KNeighborsClassifier(
                n_neighbors=9,
                weights="distance",
                p=2,
                n_jobs=-1
            )),

            ("K=15, distance", KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                p=2,
                n_jobs=-1
            )),

            ("K=21, distance", KNeighborsClassifier(
                n_neighbors=21,
                weights="distance",
                p=2,
                n_jobs=-1
            )),

            ("K=31, distance", KNeighborsClassifier(
                n_neighbors=31,
                weights="distance",
                p=2,
                n_jobs=-1
            )),

            ("K=11, Manhattan", KNeighborsClassifier(
                n_neighbors=11,
                weights="distance",
                p=1,
                n_jobs=-1
            ))
        ],

        # --------------------------------
        # 3. DECISION TREE
        # --------------------------------

        "Decision Tree": [

            ("Depth=5, leaf=5", DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=SEED
            )),

            ("Depth=8, leaf=5", DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=SEED
            )),

            ("Depth=10, leaf=10", DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=SEED
            )),

            ("Depth=12, leaf=10", DecisionTreeClassifier(
                max_depth=12,
                min_samples_split=30,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=SEED
            )),

            ("Depth=15, leaf=10", DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=30,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=SEED
            )),

            ("Depth=15, leaf=20", DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=40,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=SEED
            )),

            ("Depth=20, leaf=20", DecisionTreeClassifier(
                max_depth=20,
                min_samples_split=40,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=SEED
            ))
        ],

        # --------------------------------
        # 4. RANDOM FOREST
        # --------------------------------

        "Random Forest": [

            ("100 trees, depth=10", RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED
            )),

            ("200 trees, depth=10", RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED
            )),

            ("300 trees, depth=10", RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED
            )),

            ("100 trees, depth=15", RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED
            )),

            ("200 trees, depth=15", RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED
            )),

            ("200 trees, depth=20", RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED
            )),

            ("200 trees, depth=15, leaf=10", RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features=None,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED
            ))
        ],

        # --------------------------------
        # 5. SVM
        # --------------------------------

        "SVM": [

            ("RBF, C=0.1", SVC(
                C=0.1,
                kernel="rbf",
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=SEED
            )),

            ("RBF, C=0.5", SVC(
                C=0.5,
                kernel="rbf",
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=SEED
            )),

            ("RBF, C=1.0", SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=SEED
            )),

            ("RBF, C=5.0", SVC(
                C=5.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=SEED
            )),

            ("Linear, C=1.0", SVC(
                C=1.0,
                kernel="linear",
                probability=True,
                class_weight="balanced",
                random_state=SEED
            ))
        ],

        # --------------------------------
        # 6. NAIVE BAYES
        # --------------------------------

        "Naive Bayes": [

            ("Smoothing=1e-11", GaussianNB(
                var_smoothing=1e-11
            )),

            ("Smoothing=1e-10", GaussianNB(
                var_smoothing=1e-10
            )),

            ("Smoothing=1e-9", GaussianNB(
                var_smoothing=1e-9
            )),

            ("Smoothing=1e-8", GaussianNB(
                var_smoothing=1e-8
            )),

            ("Smoothing=1e-7", GaussianNB(
                var_smoothing=1e-7
            ))
        ]
    }

    return experiments


def build_pipeline(model):

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", model)
    ])

    return pipeline