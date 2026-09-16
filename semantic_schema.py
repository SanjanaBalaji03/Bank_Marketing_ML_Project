# Semantic schema for the Bank Marketing dataset

semantic_schema = {
    "age": {
        "type": "numerical",
        "role": "feature",
        "description": "Age of the customer"
    },
    "job": {
        "type": "categorical",
        "role": "feature",
        "description": "Type of job"
    },
    "marital": {
        "type": "categorical",
        "role": "feature",
        "description": "Marital status"
    },
    "education": {
        "type": "categorical",
        "role": "feature",
        "description": "Education level"
    },
    "default": {
        "type": "categorical",
        "role": "feature",
        "description": "Whether the customer has credit in default"
    },
    "balance": {
        "type": "numerical",
        "role": "feature",
        "description": "Average yearly balance in euros"
    },
    "housing": {
        "type": "categorical",
        "role": "feature",
        "description": "Whether the customer has a housing loan"
    },
    "loan": {
        "type": "categorical",
        "role": "feature",
        "description": "Whether the customer has a personal loan"
    },
    "contact": {
        "type": "categorical",
        "role": "feature",
        "description": "Type of communication used to contact the customer"
    },
    "day": {
        "type": "numerical",
        "role": "feature",
        "description": "Day of the month when the customer was contacted"
    },
    "month": {
        "type": "categorical",
        "role": "feature",
        "description": "Month when the customer was contacted"
    },
    "duration": {
        "type": "numerical",
        "role": "feature",
        "description": "Duration of the last contact in seconds",
        "note": "Potential data leakage for pre-contact prediction"
    },
    "campaign": {
        "type": "numerical",
        "role": "feature",
        "description": "Number of contacts performed during the current campaign"
    },
    "pdays": {
        "type": "numerical",
        "role": "feature",
        "description": "Number of days since the customer was last contacted"
    },
    "previous": {
        "type": "numerical",
        "role": "feature",
        "description": "Number of contacts performed before the current campaign"
    },
    "poutcome": {
        "type": "categorical",
        "role": "feature",
        "description": "Outcome of the previous marketing campaign"
    },
    "y": {
        "type": "categorical",
        "role": "target",
        "description": "Whether the customer subscribed to a term deposit"
    }
}


# Display the schema
print("===== SEMANTIC SCHEMA =====")

for column, details in semantic_schema.items():
    print(f"\n{column}")
    print(f"  Type: {details['type']}")
    print(f"  Role: {details['role']}")
    print(f"  Description: {details['description']}")

    if "note" in details:
        print(f"  Note: {details['note']}")