import pandas as pd
from semantic_schema import semantic_schema

# Load the dataset
df = pd.read_csv("data/bank-full.csv", sep=";")

print("===== SEMANTIC SCHEMA VALIDATION =====")

# Check that all dataset columns are present in the schema
dataset_columns = set(df.columns)
schema_columns = set(semantic_schema.keys())

missing_from_schema = dataset_columns - schema_columns
extra_in_schema = schema_columns - dataset_columns

print("\nColumns missing from schema:")
print(missing_from_schema)

print("\nExtra columns in schema:")
print(extra_in_schema)

# Check target variable
target_columns = [
    column for column, details in semantic_schema.items()
    if details["role"] == "target"
]

print("\nTarget column defined in schema:")
print(target_columns)

# Final validation
if not missing_from_schema and not extra_in_schema and target_columns == ["y"]:
    print("\nSchema validation PASSED!")
else:
    print("\nSchema validation FAILED!")