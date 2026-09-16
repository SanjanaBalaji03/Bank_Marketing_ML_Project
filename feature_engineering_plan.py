import pandas as pd
import os

# ============================================================
# FEATURE ENGINEERING PLAN
# ============================================================

print("\n===== FEATURE ENGINEERING PLAN =====")

# ------------------------------------------------------------
# 1. Load processed dataset
# ------------------------------------------------------------

df = pd.read_csv("data/bank_marketing_processed.csv")

print("\nProcessed dataset loaded successfully!")
print("Original shape:", df.shape)

# ------------------------------------------------------------
# 2. Create working copy
# ------------------------------------------------------------

df_fe = df.copy()


# ------------------------------------------------------------
# 3. Age Group
# ------------------------------------------------------------

df_fe["age_group"] = pd.cut(
    df_fe["age"],
    bins=[0, 25, 35, 45, 55, 100],
    labels=[
        "Young",
        "Young_Adult",
        "Middle_Aged",
        "Mature",
        "Senior"
    ]
)

print("\n1. Age group created successfully.")


# ------------------------------------------------------------
# 4. Balance Group
# ------------------------------------------------------------

df_fe["balance_group"] = pd.cut(
    df_fe["balance"],
    bins=[-float("inf"), 0, 500, 1500, 5000, float("inf")],
    labels=[
        "Negative",
        "Low",
        "Medium",
        "High",
        "Very_High"
    ]
)

print("2. Balance group created successfully.")


# ------------------------------------------------------------
# 5. Campaign Contact Group
# ------------------------------------------------------------

df_fe["campaign_group"] = pd.cut(
    df_fe["campaign"],
    bins=[0, 1, 2, 3, float("inf")],
    labels=[
        "1_contact",
        "2_contacts",
        "3_contacts",
        "4_plus_contacts"
    ]
)

print("3. Campaign contact group created successfully.")


# ------------------------------------------------------------
# 6. Previous Contact Indicator
# ------------------------------------------------------------

df_fe["previous_contacted"] = (
    df_fe["pdays"] != -1
).astype(int)

print("4. Previous contact indicator created successfully.")


# ------------------------------------------------------------
# 7. Previous Contact Recency
# ------------------------------------------------------------

df_fe["pdays_group"] = pd.cut(
    df_fe["pdays"],
    bins=[-2, 0, 30, 90, 180, float("inf")],
    labels=[
        "Not_Previously_Contacted",
        "Very_Recent",
        "Recent",
        "Medium_Recency",
        "Long_Ago"
    ]
)

print("5. Previous contact recency group created successfully.")


# ------------------------------------------------------------
# 8. Previous Campaign Contacts Group
# ------------------------------------------------------------

df_fe["previous_contacts_group"] = pd.cut(
    df_fe["previous"],
    bins=[-1, 0, 1, 3, float("inf")],
    labels=[
        "None",
        "One",
        "Few",
        "Many"
    ]
)

print("6. Previous campaign contact group created successfully.")


# ------------------------------------------------------------
# 9. Contact Month
# ------------------------------------------------------------

df_fe["contact_month"] = df_fe["month"]

print("7. Contact month feature created successfully.")


# ------------------------------------------------------------
# 10. List engineered features
# ------------------------------------------------------------

engineered_features = [
    "age_group",
    "balance_group",
    "campaign_group",
    "previous_contacted",
    "pdays_group",
    "previous_contacts_group",
    "contact_month"
]

print("\n===== ENGINEERED FEATURES =====")

for feature in engineered_features:
    print(f"- {feature}")


# ------------------------------------------------------------
# 11. Display sample
# ------------------------------------------------------------

print("\n===== SAMPLE OF ENGINEERED DATA =====")

print(
    df_fe[
        [
            "age",
            "age_group",
            "balance",
            "balance_group",
            "campaign",
            "campaign_group",
            "previous_contacted",
            "pdays_group",
            "previous_contacts_group"
        ]
    ].head()
)


# ------------------------------------------------------------
# 12. Feature Engineering Summary
# ------------------------------------------------------------

print("\n===== FEATURE ENGINEERING SUMMARY =====")

print("Original features:", df.shape[1])
print("New engineered features:", len(engineered_features))
print("Total features after engineering:", df_fe.shape[1])


# ------------------------------------------------------------
# 13. Save feature-engineered dataset
# ------------------------------------------------------------

output_file = "data/bank_marketing_feature_engineered.csv"

df_fe.to_csv(
    output_file,
    index=False
)

print("\nFeature-engineered dataset saved to:")
print(output_file)


# ------------------------------------------------------------
# 14. Save feature engineering plan
# ------------------------------------------------------------

plan_folder = "eda_outputs"

os.makedirs(
    plan_folder,
    exist_ok=True
)

plan_file = os.path.join(
    plan_folder,
    "feature_engineering_plan.txt"
)

with open(
    plan_file,
    "w",
    encoding="utf-8"
) as file:

    file.write("BANK MARKETING - FEATURE ENGINEERING PLAN\n")
    file.write("=" * 50 + "\n\n")

    file.write("Purpose:\n")
    file.write(
        "Create meaningful derived variables that can help the "
        "machine learning model identify customer segments and "
        "marketing response patterns.\n\n"
    )

    file.write("ENGINEERED FEATURES\n")
    file.write("-" * 30 + "\n")

    file.write(
        "1. age_group - Groups customers by age.\n"
    )

    file.write(
        "2. balance_group - Groups customers according to account balance.\n"
    )

    file.write(
        "3. campaign_group - Groups customers based on the number "
        "of contacts during the current campaign.\n"
    )

    file.write(
        "4. previous_contacted - Indicates whether the customer "
        "was contacted in a previous campaign.\n"
    )

    file.write(
        "5. pdays_group - Groups customers according to the time "
        "since the previous contact.\n"
    )

    file.write(
        "6. previous_contacts_group - Groups customers according "
        "to the number of previous contacts.\n"
    )

    file.write(
        "7. contact_month - Keeps the campaign month as a "
        "marketing-related categorical feature.\n"
    )

    file.write("\nIMPORTANT MODELING NOTE\n")
    file.write("-" * 30 + "\n")

    file.write(
        "The duration variable was excluded during preprocessing "
        "because it is only known after the customer interaction "
        "and could cause data leakage in a pre-contact prediction model.\n"
    )

    file.write(
        "\nThe engineered features will be evaluated during "
        "model development based on their predictive usefulness.\n"
    )


# ------------------------------------------------------------
# 15. Completion
# ------------------------------------------------------------

print("\n===== FEATURE ENGINEERING PLAN COMPLETED SUCCESSFULLY =====")

print("\nFiles created:")
print("- data/bank_marketing_feature_engineered.csv")
print("- eda_outputs/feature_engineering_plan.txt")