"""Create a SYNTHETIC stand-in with the same columns as creditcard_2023.csv
(id, V1..V28, Amount, Class) so you can test the whole project without downloading anything.
Results from this data are NOT real: use the Kaggle file for your report.

    python make_demo_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_FILE = Path(__file__).resolve().parent / "data" / "creditcard_2023_demo.csv"


def make(path=DEMO_FILE, rows=30000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, rows)                                  # the real 2023 file is ~50/50
    X = rng.normal(size=(rows, 28))
    shift = np.zeros(28)
    shift[[3, 9, 11, 13, 16]] = [1.2, -1.4, -1.6, -1.8, -1.1]     # fraud shifts a few components
    X += y[:, None] * shift
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "id", np.arange(rows))
    df["Amount"] = rng.uniform(50, 24000, rows).round(2)
    df["Class"] = y
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({rows:,} rows, SYNTHETIC)")
    return path


if __name__ == "__main__":
    make(rows=int(sys.argv[1]) if len(sys.argv) > 1 else 30000)
