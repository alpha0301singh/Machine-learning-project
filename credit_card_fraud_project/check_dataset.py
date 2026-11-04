
import pandas as pd
from config import DATA_FILE

# Load complete dataset
df = pd.read_csv(DATA_FILE)

print("\n========== DATASET INFORMATION ==========")

print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\n========== FRAUD DISTRIBUTION ==========")

print(df["is_fraud"].value_counts(dropna=False))

print("\nFraud percentage:")
print(
    df["is_fraud"].value_counts(normalize=True) * 100
)

print("\n========== NUMERICAL FEATURES ==========")

print(df.describe(include="number").T)

print("\n========== CATEGORICAL FEATURES ==========")

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    print(f"\n{column}")

    print(df[column].value_counts().head(10))
