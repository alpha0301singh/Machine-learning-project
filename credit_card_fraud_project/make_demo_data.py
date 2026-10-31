"""Create a SYNTHETIC stand-in with the same schema as creditcard_2023.csv
(id, V1..V28, Amount, Class) so you can smoke-test the pipeline and the app.
Use the real Kaggle file for your actual results.

    python make_demo_data.py --rows 60000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=60000)
ap.add_argument("--out", default="data/creditcard_2023_demo.csv")
a = ap.parse_args()

rng = np.random.default_rng(0)
y = rng.integers(0, 2, a.rows)                     # the real 2023 file is ~50/50
X = rng.normal(size=(a.rows, 28))
shift = np.zeros(28)
shift[[3, 9, 11, 13, 16]] = [1.2, -1.4, -1.6, -1.8, -1.1]   # fraud shifts a few components
X += y[:, None] * shift
df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
df.insert(0, "id", np.arange(a.rows))
df["Amount"] = rng.uniform(50, 24000, a.rows).round(2)
df["Class"] = y

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(f"Wrote {a.out} ({a.rows:,} rows)")
