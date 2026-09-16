import pandas as pd

# Load the raw dataset
df = pd.read_csv("data/bank-full.csv", sep=";")

print("===== PREPROCESSING =====")

# 1. Remove duration to avoid data leakage
df = df.drop(columns=["duration"])

# 2. Convert target variable from yes/no to 1/0
df["y"] = df["y"].map({"yes": 1, "no": 0})

# 3. Treat 'unknown' as a valid category
#    We keep these values because they may contain useful information.

# 4. Separate features and target
X = df.drop(columns=["y"])
y = df["y"]

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nPreprocessing completed successfully!")

# Save the processed data
processed_df = X.copy()
processed_df["y"] = y

processed_df.to_csv("data/bank_marketing_processed.csv", index=False)

print("\nProcessed dataset saved to:")
print("data/bank_marketing_processed.csv")