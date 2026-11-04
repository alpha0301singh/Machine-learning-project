
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
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
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
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ],
        sparse_threshold=0.0
    )

    return preprocessor



def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            C=0.5,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=11,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=SEED
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED
        ),

        "SVM": SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=SEED
        ),

        "Naive Bayes": GaussianNB(
            var_smoothing=1e-8
        )
    }

    return models



def build_pipeline(model):

    pipeline = Pipeline([
        (
            "preprocessor",
            build_preprocessor()
        ),
        (
            "classifier",
            model
        )
    ])

    return pipeline
