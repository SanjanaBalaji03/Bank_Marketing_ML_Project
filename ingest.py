import pandas as pd

# Load the raw dataset
df = pd.read_csv("data/bank-full.csv", sep=";")

# Display basic information
print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())