import pandas as pd

# Load the dataset
df = pd.read_csv("data/bank-full.csv", sep=";")

print("===== DATA UNDERSTANDING =====")

# Display all columns
print("\nColumns in the dataset:")
print(df.columns.tolist())

# Separate numerical and categorical columns
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

print("\nNumerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)

# Identify the target variable
target_column = "y"

print("\nTarget Variable:")
print(target_column)

# Display unique values of the target
print("\nTarget Values:")
print(df[target_column].value_counts())