import pandas as pd
import os


# ============================================================
# BANK MARKETING ML PROJECT - BUSINESS INSIGHTS
# ============================================================

print("\n===== BUSINESS INSIGHTS =====\n")


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

data_path = "data/bank_marketing_processed.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print(f"Total customers: {len(df)}")


# ------------------------------------------------------------
# 2. TARGET / SUBSCRIPTION RATE
# ------------------------------------------------------------

total_customers = len(df)
subscribed_customers = df["y"].sum()
non_subscribed_customers = total_customers - subscribed_customers

subscription_rate = (subscribed_customers / total_customers) * 100

print("\n===== CUSTOMER SUBSCRIPTION =====")

print(f"Total customers: {total_customers}")
print(f"Subscribed customers: {subscribed_customers}")
print(f"Non-subscribed customers: {non_subscribed_customers}")
print(f"Subscription rate: {subscription_rate:.2f}%")


# ------------------------------------------------------------
# 3. CUSTOMER PROFILE
# ------------------------------------------------------------

print("\n===== CUSTOMER PROFILE =====")

print(f"Average customer age: {df['age'].mean():.2f}")
print(f"Average account balance: {df['balance'].mean():.2f}")
print(f"Average campaign contacts: {df['campaign'].mean():.2f}")
print(f"Average previous contacts: {df['previous'].mean():.2f}")


# ------------------------------------------------------------
# 4. SUBSCRIPTION BY CONTACT METHOD
# ------------------------------------------------------------

print("\n===== SUBSCRIPTION BY CONTACT METHOD =====")

contact_analysis = (
    df.groupby("contact")["y"]
    .agg(["count", "sum", "mean"])
    .sort_values("mean", ascending=False)
)

contact_analysis["subscription_rate"] = contact_analysis["mean"] * 100

print(contact_analysis[["count", "sum", "subscription_rate"]])


# ------------------------------------------------------------
# 5. SUBSCRIPTION BY EDUCATION
# ------------------------------------------------------------

print("\n===== SUBSCRIPTION BY EDUCATION =====")

education_analysis = (
    df.groupby("education")["y"]
    .agg(["count", "sum", "mean"])
    .sort_values("mean", ascending=False)
)

education_analysis["subscription_rate"] = (
    education_analysis["mean"] * 100
)

print(education_analysis[["count", "sum", "subscription_rate"]])


# ------------------------------------------------------------
# 6. SUBSCRIPTION BY HOUSING LOAN
# ------------------------------------------------------------

print("\n===== SUBSCRIPTION BY HOUSING LOAN =====")

housing_analysis = (
    df.groupby("housing")["y"]
    .agg(["count", "sum", "mean"])
)

housing_analysis["subscription_rate"] = (
    housing_analysis["mean"] * 100
)

print(housing_analysis[["count", "sum", "subscription_rate"]])


# ------------------------------------------------------------
# 7. SUBSCRIPTION BY JOB
# ------------------------------------------------------------

print("\n===== SUBSCRIPTION BY JOB =====")

job_analysis = (
    df.groupby("job")["y"]
    .agg(["count", "sum", "mean"])
    .sort_values("mean", ascending=False)
)

job_analysis["subscription_rate"] = job_analysis["mean"] * 100

print(job_analysis[["count", "sum", "subscription_rate"]])


# ------------------------------------------------------------
# 8. SAVE BUSINESS ANALYSIS
# ------------------------------------------------------------

os.makedirs("model_outputs", exist_ok=True)

contact_analysis.to_csv(
    "model_outputs/subscription_by_contact.csv"
)

education_analysis.to_csv(
    "model_outputs/subscription_by_education.csv"
)

housing_analysis.to_csv(
    "model_outputs/subscription_by_housing.csv"
)

job_analysis.to_csv(
    "model_outputs/subscription_by_job.csv"
)


# ------------------------------------------------------------
# 9. BUSINESS RECOMMENDATIONS
# ------------------------------------------------------------

best_contact = contact_analysis.index[0]
best_education = education_analysis.index[0]
best_job = job_analysis.index[0]

print("\n===== BUSINESS RECOMMENDATIONS =====\n")

print(
    "1. Focus marketing campaigns on customer segments "
    "with higher historical subscription rates."
)

print(
    f"2. {best_contact} contact channel shows the highest "
    "historical subscription rate and should receive greater attention."
)

print(
    f"3. Customers with {best_education} education show the "
    "highest historical subscription rate among education groups."
)

print(
    f"4. The {best_job} customer segment shows the highest "
    "subscription rate among job categories."
)

print(
    "5. Use the Random Forest model to prioritize customers "
    "based on their predicted subscription probability."
)

print(
    "6. Avoid spending equal marketing resources on every customer. "
    "Use targeted campaigns for high-probability customers."
)


# ------------------------------------------------------------
# 10. SAVE TEXT SUMMARY
# ------------------------------------------------------------

summary_path = "model_outputs/business_insights_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("BANK MARKETING ML PROJECT - BUSINESS INSIGHTS\n")
    file.write("=" * 55 + "\n\n")

    file.write(f"Total customers: {total_customers}\n")
    file.write(f"Subscribed customers: {subscribed_customers}\n")
    file.write(
        f"Subscription rate: {subscription_rate:.2f}%\n\n"
    )

    file.write("CUSTOMER PROFILE\n")
    file.write("-" * 30 + "\n")

    file.write(
        f"Average age: {df['age'].mean():.2f}\n"
    )

    file.write(
        f"Average balance: {df['balance'].mean():.2f}\n"
    )

    file.write(
        f"Average campaign contacts: "
        f"{df['campaign'].mean():.2f}\n"
    )

    file.write(
        f"Average previous contacts: "
        f"{df['previous'].mean():.2f}\n\n"
    )

    file.write("BUSINESS RECOMMENDATIONS\n")
    file.write("-" * 30 + "\n")

    file.write(
        "1. Focus campaigns on customers with higher "
        "historical subscription rates.\n"
    )

    file.write(
        f"2. Prioritize the {best_contact} contact channel.\n"
    )

    file.write(
        f"3. Give greater attention to the {best_education} "
        "education segment.\n"
    )

    file.write(
        f"4. Consider the {best_job} segment for targeted campaigns.\n"
    )

    file.write(
        "5. Use Random Forest prediction probabilities "
        "for customer prioritization.\n"
    )

    file.write(
        "6. Allocate marketing resources based on predicted "
        "customer response.\n"
    )


print("\n===== BUSINESS INSIGHTS COMPLETED SUCCESSFULLY =====")

print("\nFiles created:")

print("- model_outputs/subscription_by_contact.csv")
print("- model_outputs/subscription_by_education.csv")
print("- model_outputs/subscription_by_housing.csv")
print("- model_outputs/subscription_by_job.csv")
print("- model_outputs/business_insights_summary.txt")