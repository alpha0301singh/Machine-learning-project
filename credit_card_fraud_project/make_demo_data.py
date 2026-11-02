
import numpy as np
import pandas as pd
from pathlib import Path


# Step 1: Set Dataset Location

DEMO_FILE = Path(__file__).resolve().parent / "data" / "creditcard_2023_demo.csv"


# Step 2: Generate Demo Dataset

def make_demo_data(rows=30000, seed=0):

    # Set random seed
    np.random.seed(seed)

    # Generate fraud labels
    fraud_labels = np.random.randint(0, 2, rows)

    # Generate 28 transaction features
    features = np.random.normal(0, 1, (rows, 28))

    # Add some differences to fraud transactions
    fraud_features = [3, 9, 11, 13, 16]

    changes = [1.2, -1.4, -1.6, -1.8, -1.1]

    # Apply changes to fraud transactions
    for i in range(len(fraud_features)):

        feature_number = fraud_features[i]

        features[:, feature_number] += (
            fraud_labels * changes[i]
        )

    # Create feature column names
    column_names = []

    for i in range(1, 29):
        column_names.append("V" + str(i))

    # Create DataFrame
    data = pd.DataFrame(
        features,
        columns=column_names
    )

    # Add transaction ID
    data.insert(0, "id", range(rows))

    # Generate transaction amounts
    data["Amount"] = np.random.uniform(
        50, 24000, rows
    ).round(2)

    # Add fraud labels
    data["Class"] = fraud_labels

    # Create data folder if it does not exist
    DEMO_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save dataset
    data.to_csv(DEMO_FILE, index=False)

    print("\nDemo Dataset Created Successfully!")

    print("Total Transactions:", len(data))

    print("Normal Transactions:",
          (data["Class"] == 0).sum())

    print("Fraud Transactions:",
          (data["Class"] == 1).sum())

    print("Dataset Location:", DEMO_FILE)

    return data


# Step 3: Run Program

if __name__ == "__main__":

    make_demo_data()
