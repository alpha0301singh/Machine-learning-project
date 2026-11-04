
import matplotlib.pyplot as plt

from config import REPORTS, TARGET
from data_prep import load_data, clean_data


def compute_eda():

    df = clean_data(load_data())

    print("\n========== DATASET SUMMARY ==========")

    print("Shape:", df.shape)

    print("\nMissing values:")

    print(df.isnull().sum())

    print("\nFraud distribution:")

    print(df[TARGET].value_counts())

    print("\nFraud percentage:")

    print(
        df[TARGET].value_counts(normalize=True) * 100
    )

    # 1. Fraud distribution
    plt.figure(figsize=(7, 5))

    df[TARGET].value_counts().sort_index().plot(
        kind="bar"
    )

    plt.title("Fraud vs Legitimate Transactions")

    plt.xlabel("Transaction Class")

    plt.ylabel("Number of Transactions")

    plt.xticks(
        [0, 1],
        ["Legitimate", "Fraud"],
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS / "fraud_distribution.png"
    )

    plt.close()

    # 2. Transaction amount distribution
    plt.figure(figsize=(8, 5))

    for fraud_class, group in df.groupby(TARGET):

        plt.hist(
            group["transaction_amount"].clip(upper=1000000),
            bins=50,
            alpha=0.5,
            label="Fraud" if fraud_class == 1 else "Legitimate"
        )

    plt.title("Transaction Amount Distribution")

    plt.xlabel("Transaction Amount (INR)")

    plt.ylabel("Number of Transactions")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORTS / "transaction_amount.png"
    )

    plt.close()

    # 3. Fraud rate by transaction hour
    fraud_by_hour = df.groupby(
        "transaction_hour"
    )[TARGET].mean() * 100

    plt.figure(figsize=(9, 5))

    fraud_by_hour.plot(
        marker="o"
    )

    plt.title("Fraud Percentage by Transaction Hour")

    plt.xlabel("Transaction Hour")

    plt.ylabel("Fraud Percentage")

    plt.xticks(range(0, 24))

    plt.tight_layout()

    plt.savefig(
        REPORTS / "fraud_by_hour.png"
    )

    plt.close()

    # 4. Fraud rate by transaction type
    fraud_by_type = (
        df.groupby("transaction_type")[TARGET]
        .mean()
        .sort_values(ascending=False) * 100
    )

    plt.figure(figsize=(10, 6))

    fraud_by_type.plot(
        kind="bar"
    )

    plt.title("Fraud Percentage by Transaction Type")

    plt.xlabel("Transaction Type")

    plt.ylabel("Fraud Percentage")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        REPORTS / "fraud_by_transaction_type.png"
    )

    plt.close()

    print("\nEDA completed.")

    print("Charts saved in:", REPORTS)

    return df


if __name__ == "__main__":

    compute_eda()
