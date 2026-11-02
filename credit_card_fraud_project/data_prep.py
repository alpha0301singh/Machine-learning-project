
"""Data loading, cleaning, balancing and splitting."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    DATA_FILE,
    ARTIFACTS,
    FEATURES,
    TARGET,
    N_LEGIT,
    FRAUD_RATE,
    SEED,
    TEST_SIZE
)


# STEP 1: LOAD DATA

def load_data(path=DATA_FILE):

    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found: {path}")

    print("Dataset loaded successfully")
    print("Total rows:", len(data))

    return data


# STEP 2: CLEAN DATA

def clean_data(data):

    original_rows = len(data)

    # Check required columns
    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep only required columns
    data = data[required_columns].copy()

    # Convert columns to numeric values
    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Replace infinite values with missing values
    data = data.replace([np.inf, -np.inf], np.nan)

    # Treat negative transaction amounts as missing
    data.loc[data["Amount"] < 0, "Amount"] = np.nan

    # Keep only valid fraud labels
    data = data[data[TARGET].isin([0, 1])].copy()

    # Convert fraud labels to integers
    data[TARGET] = data[TARGET].astype(int)

    # Remove duplicate rows
    data = data.drop_duplicates()

    data = data.reset_index(drop=True)

    # Cleaning information
    cleaning_info = {
        "original_rows": int(original_rows),
        "remaining_rows": int(len(data)),
        "removed_rows": int(original_rows - len(data)),
        "missing_values": int(data.isnull().sum().sum())
    }

    print("\nData cleaning completed")
    print("Remaining rows:", len(data))
    print("Missing values:", cleaning_info["missing_values"])

    return data, cleaning_info


# STEP 3: HANDLE CLASS IMBALANCE

def balance_data(data, n_legit=N_LEGIT, fraud_rate=FRAUD_RATE):

    if n_legit <= 0:
        raise ValueError("n_legit must be greater than zero")

    if not 0 < fraud_rate < 1:
        raise ValueError("fraud_rate must be between 0 and 1")

    # Separate legitimate and fraud transactions
    normal = data[data[TARGET] == 0]
    fraud = data[data[TARGET] == 1]

    if len(normal) == 0 or len(fraud) == 0:
        raise ValueError(
            "Dataset must contain both legitimate and fraud transactions"
        )

    # Select legitimate transactions
    normal = normal.sample(
        n=min(n_legit, len(normal)),
        random_state=SEED
    )

    # Calculate required fraud transactions
    fraud_count = int(
        len(normal) * fraud_rate / (1 - fraud_rate)
    )

    # Select available fraud transactions
    fraud = fraud.sample(
        n=min(fraud_count, len(fraud)),
        random_state=SEED
    )

    # Combine both classes
    balanced_data = pd.concat(
        [normal, fraud],
        ignore_index=True
    )

    # Shuffle transactions
    balanced_data = balanced_data.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    print("\nClass balancing completed")
    print("Legitimate transactions:", len(normal))
    print("Fraud transactions:", len(fraud))
    print(
        "Actual fraud percentage:",
        round(balanced_data[TARGET].mean() * 100, 2),
        "%"
    )

    return balanced_data


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

    # Combine features and target into DataFrames
    train = X_train.copy()
    train[TARGET] = y_train

    test = X_test.copy()
    test[TARGET] = y_test

    print("\nTraining and testing data created")
    print("Training rows:", len(train))
    print("Testing rows:", len(test))

    return train, test


# STEP 5: COMPLETE DATA PREPARATION

def prepare(path=DATA_FILE, n_legit=N_LEGIT, fraud_rate=FRAUD_RATE):

    # Load dataset
    raw_data = load_data(path)

    raw_rows = len(raw_data)

    # Clean dataset
    cleaned_data, cleaning_info = clean_data(raw_data)

    # Balance dataset
    balanced_data = balance_data(
        cleaned_data,
        n_legit,
        fraud_rate
    )

    # Split dataset
    train, test = split_data(balanced_data)

    # Create artifacts folder
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # Save training and testing datasets
    train.to_csv(
        ARTIFACTS / "train.csv",
        index=False
    )

    test.to_csv(
        ARTIFACTS / "test.csv",
        index=False
    )

    # Information used by train.py and app.py
    info = {
        "raw_rows": int(raw_rows),
        "study_rows": int(len(balanced_data)),
        "fraud_rate": float(balanced_data[TARGET].mean()),
        "test_rows": int(len(test)),
        "cleaning": cleaning_info
    }

    print("\nData preparation completed successfully")
    print("Training fraud percentage:",
          round(train[TARGET].mean() * 100, 2), "%")
    print("Testing fraud percentage:",
          round(test[TARGET].mean() * 100, 2), "%")

    return train, test, info


# RUN THIS FILE DIRECTLY

if __name__ == "__main__":
    prepare()
