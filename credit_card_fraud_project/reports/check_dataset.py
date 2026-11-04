
import pandas as pd
from config import DATA_FILE

print("Dataset path:")
print(DATA_FILE.resolve())

df = pd.read_csv(DATA_FILE, nrows=5)

print("\nDataset shape (first 5 rows loaded):")
print(df.shape)

print("\nActual column names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())
