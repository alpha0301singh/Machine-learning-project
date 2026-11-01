"""All settings in one place. Edit here, every other file picks it up."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "creditcard_2023.csv"   # the Kaggle dataset (see README)
ARTIFACTS = ROOT / "artifacts"                       # everything the Streamlit app reads
REPORTS = ROOT / "reports"                           # PNG charts for your report

V_COLS = [f"V{i}" for i in range(1, 29)]             # anonymised features
FEATURES = V_COLS + ["Amount"]
TARGET = "Class"                                     # 1 = fraud, 0 = legitimate

SEED = 42
N_LEGIT = 20000      # legitimate rows kept (keeps training fast)
FRAUD_RATE = 0.02    # fraud share after re-balancing (real fraud is rare)
TEST_SIZE = 0.2
CV_FOLDS = 3
SELECT_K = 20        # features kept by feature selection
