
from pathlib import Path

# Project directories
ROOT = Path(__file__).resolve().parent

DATA_FOLDER = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

DATA_FOLDER.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


# Dataset path
DATA_FILE = DATA_FOLDER / "indian_banking_transactions.csv"


# Target column
TARGET = "is_fraud"


# Numerical features
NUMERIC_FEATURES = [
    "transaction_amount",
    "credit_score",
    "has_loan",
    "emi_amount",
    "transaction_hour",
    "transaction_month",
    "transaction_day_of_week"
]


# Categorical features
CATEGORICAL_FEATURES = [
    "account_type",
    "transaction_type",
    "transaction_direction",
    "merchant_category",
    "state",
    "loan_type",
    "channel",
    "kyc_status"
]


# All input features
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# Training settings
SEED = 42
TEST_SIZE = 0.20

# Smaller samples for computationally expensive algorithms
MAX_DISTANCE_MODEL_SAMPLES_PER_CLASS = 2000

# Common stratified evaluation sample for all six models
EVALUATION_SAMPLE_SIZE = 50000


# Saved files
MODEL_FILE = ARTIFACTS / "best_model.joblib"

METRICS_FILE = ARTIFACTS / "results.json"

COMPARISON_FILE = REPORTS / "model_comparison.csv"

PREDICTIONS_FILE = REPORTS / "test_predictions.csv"
