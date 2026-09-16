import pandas as pd
import os

# ============================================================
# EDA ANALYSIS SUMMARY & PRESENTATION
# ============================================================

print("\n===== EDA ANALYSIS SUMMARY =====")

# ------------------------------------------------------------
# 1. Load the processed dataset
# ------------------------------------------------------------

df = pd.read_csv("data/bank_marketing_processed.csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 2. Convert target variable into readable format
# ------------------------------------------------------------

# The target variable is y
# 0 = No subscription
# 1 = Yes subscription

if df["y"].dtype == "object":
    df["y_numeric"] = df["y"].map({"no": 0, "yes": 1})
else:
    df["y_numeric"] = df["y"]


# ------------------------------------------------------------
# 3. Overall subscription rate
# ------------------------------------------------------------

total_customers = len(df)
subscribers = df["y_numeric"].sum()

subscription_rate = (subscribers / total_customers) * 100

print("\n1. OVERALL SUBSCRIPTION")
print("--------------------------------")
print("Total customers:", total_customers)
print("Customers subscribed:", subscribers)
print(f"Overall subscription rate: {subscription_rate:.2f}%")


# ------------------------------------------------------------
# 4. Numerical variable analysis
# ------------------------------------------------------------

print("\n2. NUMERICAL VARIABLES")
print("--------------------------------")

numerical_columns = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous"
]

for column in numerical_columns:

    if column in df.columns:
        print(f"\n{column.upper()}")

        print(f"Average: {df[column].mean():.2f}")
        print(f"Minimum: {df[column].min():.2f}")
        print(f"Maximum: {df[column].max():.2f}")


# ------------------------------------------------------------
# 5. Subscription rate by important categorical variables
# ------------------------------------------------------------

categorical_columns = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome"
]

print("\n3. SUBSCRIPTION RATE BY CUSTOMER CHARACTERISTICS")
print("------------------------------------------------")

results = {}

for column in categorical_columns:

    if column in df.columns:

        rate = (
            df.groupby(column)["y_numeric"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )

        results[column] = rate

        print(f"\n--- {column.upper()} ---")

        for category, value in rate.items():
            print(f"{category}: {value:.2f}%")


# ------------------------------------------------------------
# 6. Identify best-performing categories
# ------------------------------------------------------------

print("\n4. TOP-PERFORMING CUSTOMER SEGMENTS")
print("------------------------------------")

top_segments = {}

for column, rate in results.items():

    if len(rate) > 0:

        best_category = rate.index[0]
        best_rate = rate.iloc[0]

        top_segments[column] = {
            "category": best_category,
            "rate": best_rate
        }

        print(
            f"{column.capitalize()}: "
            f"{best_category} ({best_rate:.2f}% subscription)"
        )


# ------------------------------------------------------------
# 7. Business insights
# ------------------------------------------------------------

print("\n5. KEY BUSINESS INSIGHTS")
print("-------------------------")

insights = []

# Overall subscription
insights.append(
    f"The overall subscription rate is {subscription_rate:.2f}%, "
    "showing that the marketing campaign has a relatively small "
    "share of customers converting."
)

# Job
if "job" in results:
    best_job = results["job"].index[0]
    best_job_rate = results["job"].iloc[0]

    insights.append(
        f"The {best_job} customer segment has the highest subscription "
        f"rate among job categories ({best_job_rate:.2f}%)."
    )

# Education
if "education" in results:
    best_education = results["education"].index[0]
    best_education_rate = results["education"].iloc[0]

    insights.append(
        f"The {best_education} education group shows the highest "
        f"subscription rate ({best_education_rate:.2f}%)."
    )

# Contact
if "contact" in results:
    best_contact = results["contact"].index[0]
    best_contact_rate = results["contact"].iloc[0]

    insights.append(
        f"The {best_contact} contact method has the highest "
        f"subscription rate ({best_contact_rate:.2f}%)."
    )

# Housing
if "housing" in results:
    best_housing = results["housing"].index[0]
    best_housing_rate = results["housing"].iloc[0]

    insights.append(
        f"Customers with housing status '{best_housing}' show the "
        f"higher subscription rate ({best_housing_rate:.2f}%)."
    )

# Personal loan
if "loan" in results:
    best_loan = results["loan"].index[0]
    best_loan_rate = results["loan"].iloc[0]

    insights.append(
        f"Customers with personal-loan status '{best_loan}' have the "
        f"highest subscription rate ({best_loan_rate:.2f}%)."
    )

# Previous campaign
if "poutcome" in results:
    best_poutcome = results["poutcome"].index[0]
    best_poutcome_rate = results["poutcome"].iloc[0]

    insights.append(
        f"Customers with previous campaign outcome '{best_poutcome}' "
        f"have the highest subscription rate ({best_poutcome_rate:.2f}%)."
    )


for i, insight in enumerate(insights, start=1):
    print(f"{i}. {insight}")


# ------------------------------------------------------------
# 8. Business recommendations
# ------------------------------------------------------------

print("\n6. BUSINESS RECOMMENDATIONS")
print("----------------------------")

recommendations = []

recommendations.append(
    "Focus future campaigns on customer segments with higher "
    "historical subscription rates."
)

if "contact" in results:
    recommendations.append(
        f"Give greater attention to the {results['contact'].index[0]} "
        "contact channel because it shows the strongest subscription rate."
    )

if "job" in results:
    recommendations.append(
        f"Prioritize the {results['job'].index[0]} customer segment "
        "when designing targeted marketing campaigns."
    )

if "education" in results:
    recommendations.append(
        f"Use customer education level as a segmentation variable, "
        f"with {results['education'].index[0]} showing the strongest "
        "historical response."
    )

recommendations.append(
    "Use previous campaign outcomes to identify customers who are "
    "more likely to respond positively."
)

recommendations.append(
    "Avoid treating all customers the same; use customer characteristics "
    "to create targeted marketing campaigns."
)

for i, recommendation in enumerate(recommendations, start=1):
    print(f"{i}. {recommendation}")


# ------------------------------------------------------------
# 9. Save presentation-ready summary
# ------------------------------------------------------------

output_folder = "eda_outputs"

os.makedirs(output_folder, exist_ok=True)

summary_file = os.path.join(
    output_folder,
    "eda_analysis_summary.txt"
)

with open(summary_file, "w", encoding="utf-8") as file:

    file.write("BANK MARKETING CAMPAIGN - EDA ANALYSIS SUMMARY\n")
    file.write("=" * 55 + "\n\n")

    file.write("1. OVERALL SUBSCRIPTION\n")
    file.write("-" * 30 + "\n")
    file.write(f"Total customers: {total_customers}\n")
    file.write(f"Customers subscribed: {subscribers}\n")
    file.write(
        f"Overall subscription rate: {subscription_rate:.2f}%\n\n"
    )

    file.write("2. TOP-PERFORMING CUSTOMER SEGMENTS\n")
    file.write("-" * 40 + "\n")

    for column, data in top_segments.items():
        file.write(
            f"{column.capitalize()}: "
            f"{data['category']} "
            f"({data['rate']:.2f}% subscription)\n"
        )

    file.write("\n3. KEY BUSINESS INSIGHTS\n")
    file.write("-" * 30 + "\n")

    for i, insight in enumerate(insights, start=1):
        file.write(f"{i}. {insight}\n")

    file.write("\n4. BUSINESS RECOMMENDATIONS\n")
    file.write("-" * 35 + "\n")

    for i, recommendation in enumerate(recommendations, start=1):
        file.write(f"{i}. {recommendation}\n")


# ------------------------------------------------------------
# 10. Completion message
# ------------------------------------------------------------

print("\n===== EDA SUMMARY COMPLETED SUCCESSFULLY =====")

print("\nSummary saved to:")
print("eda_outputs/eda_analysis_summary.txt")