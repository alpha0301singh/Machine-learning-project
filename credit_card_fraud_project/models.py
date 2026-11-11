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


# ----------------------------------------
# PREPROCESSING
# ----------------------------------------

def build_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),

        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=20,
            sparse_output=False,
            dtype=np.float32
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        sparse_threshold=0.0
    )

    return preprocessor


# ----------------------------------------
# FINAL SELECTED MODELS
# ----------------------------------------

def get_final_models():

    models = {

        "Logistic Regression": (
            "C=0.01",

            LogisticRegression(
                C=0.01,
                class_weight="balanced",
                max_iter=1000,
                random_state=SEED
            )
        ),

        "KNN": (
            "K=31, distance",

            KNeighborsClassifier(
                n_neighbors=31,
                weights="distance",
                metric="minkowski",
                p=2
            )
        ),

        "Decision Tree": (
            "Depth=12, leaf=10",

            DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=SEED
            )
        ),

        "Random Forest": (
            "100 trees, depth=10",

            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1
            )
        ),

        "SVM": (
            "RBF, C=0.1",

            SVC(
                kernel="rbf",
                C=0.1,
                class_weight="balanced",
                probability=True,
                random_state=SEED
            )
        ),

        "Naive Bayes": (
            "Smoothing=1e-8",

            GaussianNB(
                var_smoothing=1e-8
            )
        )
    }

    return models


# ----------------------------------------
# COMPLETE MACHINE LEARNING PIPELINE
# ----------------------------------------

def build_pipeline(model):

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", model)
    ])

    return pipeline