
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from config import (
    DATA_FILE, ARTIFACTS, FEATURES, TARGET,
    N_LEGIT, FRAUD_RATE, SEED, TEST_SIZE
)


# STEP 1: LOAD DATA

def load_data(path=DATA_FILE):

    data = pd.read_csv(path)

    print("Dataset loaded successfully")
    print("Total rows:", len(data))

    return data


# STEP 2: CLEAN DATA

def clean_data(data):

    original_rows = len(data)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    data = data[required_columns].copy()

    # Convert columns into numeric values
    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Handle infinite values
    data = data.replace([np.inf, -np.inf], np.nan)

    # Negative amounts are treated as missing
    data.loc[data["Amount"] < 0, "Amount"] = np.nan

    # Keep valid target labels
    data = data[data[TARGET].isin([0, 1])].copy()

    data[TARGET] = data[TARGET].astype(int)

    # Remove duplicate records
    data = data.drop_duplicates().reset_index(drop=True)

    cleaning_info = {
        "original_rows": int(original_rows),
        "remaining_rows": int(len(data)),
        "removed_rows": int(original_rows - len(data)),
        "missing_values": int(data.isnull().sum().sum())
    }

    print("Data cleaning completed")
    print("Remaining rows:", len(data))
    print("Missing values:", cleaning_info["missing_values"])

    return data, cleaning_info


# STEP 3: HANDLE CLASS IMBALANCE

def balance_data(data, n_legit=N_LEGIT, fraud_rate=FRAUD_RATE):

    if n_legit <= 0:
        raise ValueError("n_legit must be positive")

    if not 0 < fraud_rate < 1:
        raise ValueError("fraud_rate must be between 0 and 1")

    normal = data[data[TARGET] == 0]
    fraud = data[data[TARGET] == 1]

    if len(normal) == 0 or len(fraud) == 0:
        raise ValueError("Both classes must be present")

    normal = normal.sample(
        n=min(n_legit, len(normal)),
        random_state=SEED
    )

    fraud_count = int(
        len(normal) * fraud_rate / (1 - fraud_rate)
    )

    fraud = fraud.sample(
        n=min(fraud_count, len(fraud)),
        random_state=SEED
    )

    balanced = pd.concat(
        [normal, fraud],
        ignore_index=True
    )

    balanced = balanced.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    print("Class balancing completed")
    print("Legitimate:", len(normal))
    print("Fraud:", len(fraud))
    print("Actual fraud rate:", round(balanced[TARGET].mean() * 100, 2), "%")

    return balanced


# STEP 4: SPLIT DATA

def split_data(data):

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )

    train = X_train.copy()
    train[TARGET] = y_train

    test = X_test.copy()
    test[TARGET] = y_test

    return train, test


# STEP 5: COMPLETE PREPARATION

def prepare(path=DATA_FILE, n_legit=N_LEGIT, fraud_rate=FRAUD_RATE):

    raw_data = load_data(path)

    cleaned_data, cleaning_info = clean_data(raw_data)

    balanced_data = balance_data(
        cleaned_data,
        n_legit,
        fraud_rate
    )

    train, test = split_data(balanced_data)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    train.to_csv(ARTIFACTS / "train.csv", index=False)
    test.to_csv(ARTIFACTS / "test.csv", index=False)

    info = {
        "raw_rows": int(len(raw_data)),
        "study_rows": int(len(balanced_data)),
        "fraud_rate": float(balanced_data[TARGET].mean()),
        "test_rows": int(len(test)),
        "cleaning": cleaning_info
    }

    print("\nData preparation completed")
    print("Training rows:", len(train))
    print("Testing rows:", len(test))

    return train, test, info


if __name__ == "__main__":
    prepare()
