
import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    DATA_FILE,
    FEATURES,
    TARGET,
    SEED,
    TEST_SIZE
)


def load_data():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE, low_memory=False)

    # Remove accidental spaces in column names
    df.columns = df.columns.str.strip()

    print("Dataset loaded successfully")
    print("Dataset shape:", df.shape)

    return df


def clean_data(df):

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows without target
    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    df = df.dropna(subset=[TARGET])

    # Validate target values
    if not set(df[TARGET].unique()).issubset({0, 1}):

        raise ValueError(
            "is_fraud must contain only 0 and 1."
        )

    df[TARGET] = df[TARGET].astype(int)

    # Convert numerical columns
    numeric_columns = [
        "transaction_amount",
        "credit_score",
        "has_loan",
        "emi_amount",
        "transaction_hour"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Convert date into datetime
    transaction_dates = pd.to_datetime(
        df["transaction_date"],
        errors="coerce"
    )

    # Extract month
    df["transaction_month"] = (
        transaction_dates.dt.month
    )

    # Extract day of week
    df["transaction_day_of_week"] = (
        transaction_dates.dt.dayofweek
    )

    print("Data cleaning completed")

    return df


def prepare_data():

    df = clean_data(load_data())

    # Verify all required features
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    X = df[FEATURES].copy()

    y = df[TARGET].copy()

    # Split before any sampling or balancing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )

    print("\nTraining data:", X_train.shape)

    print("Testing data:", X_test.shape)

    print("\nTraining class distribution:")

    print(y_train.value_counts())

    print("\nTesting class distribution:")

    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    prepare_data()
