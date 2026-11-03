
from pathlib import Path

# Main project folder
ROOT = Path(__file__).resolve().parent

# Project folders
DATA_FOLDER = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

# Dataset
DATA_FILE = DATA_FOLDER / "creditcard_2023.csv"

# Dataset columns
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = V_COLS + ["Amount"]
TARGET = "Class"

# Model settings
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
SELECT_K = 20
RF_ESTIMATORS = 150


# Dataset balancing settings
N_LEGIT = 20000
FRAUD_RATE = 0.02

# Random Forest tuning
RF_ESTIMATORS = 150
