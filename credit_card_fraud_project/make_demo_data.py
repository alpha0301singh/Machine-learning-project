
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_FOLDER, V_COLS


DEMO_FILE = DATA_FOLDER / "creditcard_2023_demo.csv"


def make(path=DEMO_FILE, rows=30000, seed=42):

    rng = np.random.default_rng(seed)

    # Generate transaction labels
    labels = rng.choice(
        [0, 1],
        size=rows,
        p=[0.98, 0.02]
    )

    # Generate numerical features
    features = rng.normal(
        0,
        1,
        size=(rows, len(V_COLS))
    )

    # Add artificial differences to fraud transactions
    fraud_indexes = np.where(labels == 1)[0]

    selected_features = [3, 9, 11, 13, 16]

    shifts = [1.2, -1.4, -1.6, -1.8, -1.1]

    for feature, shift in zip(selected_features, shifts):

        features[fraud_indexes, feature] += shift

    # Generate transaction amounts
    amount = rng.lognormal(
        mean=4,
        sigma=1,
        size=rows
    )

    # Create dataset
    data = pd.DataFrame(
        features,
        columns=V_COLS
    )

    data["Amount"] = np.round(amount, 2)
    data["Class"] = labels

    # Add an ID column for demonstration
    data.insert(0, "id", range(1, rows + 1))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(path, index=False)

    print("Synthetic dataset created:", path)
    print("Total rows:", len(data))
    print("Fraud rows:", int(data["Class"].sum()))

    return path


if __name__ == "__main__":
    make()
