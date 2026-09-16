import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# REVISED EDA - BANK MARKETING ML PROJECT
# ============================================================

print("===== REVISED EDA =====")

# ------------------------------------------------------------
# 1. Create folder for EDA outputs
# ------------------------------------------------------------

os.makedirs("eda_outputs", exist_ok=True)

# ------------------------------------------------------------
# 2. Load processed dataset
# ------------------------------------------------------------

df = pd.read_csv("data/bank_marketing_processed.csv")

print("\nDataset Shape:")
print(df.shape)

# ------------------------------------------------------------
# 3. Target Distribution
# ------------------------------------------------------------

print("\nTarget Distribution:")
print(df["y"].value_counts())

# Calculate percentages
target_percentage = df["y"].value_counts(normalize=True) * 100

print("\nTarget Distribution (%):")
print(target_percentage)

# Target distribution chart
plt.figure(figsize=(8, 5))

df["y"].value_counts().sort_index().plot(kind="bar")

plt.title("Target Distribution")
plt.xlabel("Subscription (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(
    "eda_outputs/target_distribution.png",
    bbox_inches="tight"
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 4. Numerical Columns
# ------------------------------------------------------------

numerical_columns = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous"
]

print("\nNumerical Columns:")
print(numerical_columns)

print("\nNumerical Summary:")
print(df[numerical_columns].describe())

# ------------------------------------------------------------
# 5. Age Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["age"], bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig(
    "eda_outputs/age_distribution.png",
    bbox_inches="tight"
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 6. Balance Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["balance"], bins=30)

plt.title("Balance Distribution")
plt.xlabel("Account Balance")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig(
    "eda_outputs/balance_distribution.png",
    bbox_inches="tight"
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 7. Campaign Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["campaign"], bins=20)

plt.title("Number of Contacts During Campaign")
plt.xlabel("Campaign Contacts")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig(
    "eda_outputs/campaign_distribution.png",
    bbox_inches="tight"
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 8. Previous Contacts Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["previous"], bins=20)

plt.title("Previous Campaign Contacts")
plt.xlabel("Previous Contacts")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig(
    "eda_outputs/previous_contacts_distribution.png",
    bbox_inches="tight"
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 9. Subscription by Job
# ------------------------------------------------------------

job_subscription = pd.crosstab(
    df["job"],
    df["y"],
    normalize="index"
) * 100

print("\nSubscription Rate by Job (%):")
print(job_subscription)

plt.figure(figsize=(10, 6))

job_subscription[1].sort_values().plot(kind="barh")

plt.title("Subscription Rate by Job")
plt.xlabel("Subscription Rate (%)")
plt.ylabel("Job")

plt.tight_layout()
plt.savefig(
    "eda_outputs/subscription_by_job.png",
    bbox_inches="tight"
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 10. Subscription by Education
# ------------------------------------------------------------

education_subscription = pd.crosstab(
    df["education"],
    df["y"],
    normalize="index"
) * 100

print("\nSubscription Rate by Education (%):")
print(education_subscription)

plt.figure(figsize=(8, 5))

education_subscription[1].sort_values().plot(kind="bar")

plt.title("Subscription Rate by Education")
plt.xlabel("Education")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(
    "eda_outputs/subscription_by_education.png",
    bbox_inches="tight"
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 11. Subscription by Marital Status
# ------------------------------------------------------------

marital_subscription = pd.crosstab(
    df["marital"],
    df["y"],
    normalize="index"
) * 100

print("\nSubscription Rate by Marital Status (%):")
print(marital_subscription)

plt.figure(figsize=(8, 5))

marital_subscription[1].sort_values().plot(kind="bar")

plt.title("Subscription Rate by Marital Status")
plt.xlabel("Marital Status")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(
    "eda_outputs/subscription_by_marital.png",
    bbox_inches="tight"
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 12. Subscription by Contact Method
# ------------------------------------------------------------

contact_subscription = pd.crosstab(
    df["contact"],
    df["y"],
    normalize="index"
) * 100

print("\nSubscription Rate by Contact Method (%):")
print(contact_subscription)

plt.figure(figsize=(8, 5))

contact_subscription[1].sort_values().plot(kind="bar")

plt.title("Subscription Rate by Contact Method")
plt.xlabel("Contact Method")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(
    "eda_outputs/subscription_by_contact.png",
    bbox_inches="tight"
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 13. Subscription by Month
# ------------------------------------------------------------

month_subscription = pd.crosstab(
    df["month"],
    df["y"],
    normalize="index"
) * 100

print("\nSubscription Rate by Month (%):")
print(month_subscription)

plt.figure(figsize=(10, 5))

month_subscription[1].plot(kind="bar")

plt.title("Subscription Rate by Month")
plt.xlabel("Month")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(
    "eda_outputs/subscription_by_month.png",
    bbox_inches="tight"
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 14. Housing Loan vs Subscription
# ------------------------------------------------------------

housing_subscription = pd.crosstab(
    df["housing"],
    df["y"],
    normalize="index"
) * 100

print("\nSubscription Rate by Housing Loan Status (%):")
print(housing_subscription)

plt.figure(figsize=(8, 5))

housing_subscription[1].sort_values().plot(kind="bar")

plt.title("Subscription Rate by Housing Loan Status")
plt.xlabel("Housing Loan")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(
    "eda_outputs/subscription_by_housing.png",
    bbox_inches="tight"
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 15. Personal Loan vs Subscription
# ------------------------------------------------------------

loan_subscription = pd.crosstab(
    df["loan"],
    df["y"],
    normalize="index"
) * 100

print("\nSubscription Rate by Personal Loan Status (%):")
print(loan_subscription)

plt.figure(figsize=(8, 5))

loan_subscription[1].sort_values().plot(kind="bar")

plt.title("Subscription Rate by Personal Loan Status")
plt.xlabel("Personal Loan")
plt.ylabel("Subscription Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(
    "eda_outputs/subscription_by_personal_loan.png",
    bbox_inches="tight"
)
plt.show()
plt.close()

# ------------------------------------------------------------
# 16. EDA Summary
# ------------------------------------------------------------

print("\n===== EDA SUMMARY =====")

print("\n1. Target imbalance:")
print(
    f"Customers who did not subscribe: "
    f"{df['y'].value_counts()[0]}"
)

print(
    f"Customers who subscribed: "
    f"{df['y'].value_counts()[1]}"
)

print(
    f"Overall subscription rate: "
    f"{df['y'].mean() * 100:.2f}%"
)

print("\n2. Average customer age:")
print(f"{df['age'].mean():.2f} years")

print("\n3. Average account balance:")
print(f"{df['balance'].mean():.2f}")

print("\n4. Average campaign contacts:")
print(f"{df['campaign'].mean():.2f}")

print("\n5. Average previous contacts:")
print(f"{df['previous'].mean():.2f}")

print("\n===== EDA COMPLETED SUCCESSFULLY =====")

print("\nCharts saved in:")
print("eda_outputs/")