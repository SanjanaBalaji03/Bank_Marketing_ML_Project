import pandas as pd
import joblib
import os


# ============================================================
# BANK MARKETING - CUSTOMER SUBSCRIPTION PREDICTION
# ============================================================

print("\n===== CUSTOMER SUBSCRIPTION PREDICTION =====\n")


# ------------------------------------------------------------
# 1. Load trained model
# ------------------------------------------------------------

model_path = "models/best_bank_marketing_model.pkl"

if not os.path.exists(model_path):
    print("ERROR: Trained model not found!")
    print(f"Expected location: {model_path}")
    exit()

model = joblib.load(model_path)

print("Trained model loaded successfully!")


# ------------------------------------------------------------
# 2. Get customer information
# ------------------------------------------------------------

print("\nEnter customer details below.")
print("")


age = int(input("Age: "))

job = input(
    "Job "
    "(admin., blue-collar, entrepreneur, housemaid, management, "
    "retired, self-employed, services, student, technician, "
    "unemployed, unknown): "
).strip()

marital = input(
    "Marital status (married, single, divorced): "
).strip()

education = input(
    "Education (primary, secondary, tertiary, unknown): "
).strip()

default = input(
    "Has credit in default? (yes/no): "
).strip()

balance = float(
    input("Average yearly account balance (€): ")
)

housing = input(
    "Has housing loan? (yes/no): "
).strip()

loan = input(
    "Has personal loan? (yes/no): "
).strip()

contact = input(
    "Contact type (cellular, telephone, unknown): "
).strip()

day = int(
    input("Day of month contacted (1-31): ")
)

month = input(
    "Month contacted "
    "(jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec): "
).strip()

campaign = int(
    input("Number of contacts during current campaign: ")
)

pdays = int(
    input("Number of days since previous contact (-1 if never contacted): ")
)

previous = int(
    input("Number of contacts before current campaign: ")
)

poutcome = input(
    "Previous campaign outcome "
    "(failure, success, other, unknown): "
).strip()


# ------------------------------------------------------------
# 3. Create customer DataFrame
# ------------------------------------------------------------

customer = pd.DataFrame({
    "age": [age],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "default": [default],
    "balance": [balance],
    "housing": [housing],
    "loan": [loan],
    "contact": [contact],
    "day": [day],
    "month": [month],
    "campaign": [campaign],
    "pdays": [pdays],
    "previous": [previous],
    "poutcome": [poutcome]
})


# ------------------------------------------------------------
# 4. Make prediction
# ------------------------------------------------------------

prediction = model.predict(customer)[0]

probabilities = model.predict_proba(customer)[0]

subscription_probability = probabilities[1] * 100


# ------------------------------------------------------------
# 5. Display prediction
# ------------------------------------------------------------

print("\n==========================================")
print("          CUSTOMER PREDICTION")
print("==========================================")

if prediction == 1:

    print("Prediction: YES - CUSTOMER MAY SUBSCRIBE")

else:

    print("Prediction: NO - CUSTOMER MAY NOT SUBSCRIBE")


print(f"Subscription Probability: {subscription_probability:.2f}%")
print(f"Non-Subscription Probability: {probabilities[0] * 100:.2f}%")


# ------------------------------------------------------------
# 6. Business recommendation
# ------------------------------------------------------------

print("\n===== BUSINESS RECOMMENDATION =====")

if subscription_probability >= 70:

    recommendation = "HIGH PRIORITY"
    print("HIGH PRIORITY CUSTOMER")
    print("Recommended action: Contact this customer with a targeted offer.")

elif subscription_probability >= 40:

    recommendation = "MEDIUM PRIORITY"
    print("MEDIUM PRIORITY CUSTOMER")
    print("Recommended action: Consider targeted communication.")

else:

    recommendation = "LOW PRIORITY"
    print("LOW PRIORITY CUSTOMER")
    print("Recommended action: Avoid spending excessive campaign resources.")


# ------------------------------------------------------------
# 7. Save prediction
# ------------------------------------------------------------

os.makedirs("model_outputs", exist_ok=True)

prediction_result = pd.DataFrame({
    "age": [age],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "balance": [balance],
    "housing": [housing],
    "loan": [loan],
    "contact": [contact],
    "campaign": [campaign],
    "pdays": [pdays],
    "previous": [previous],
    "poutcome": [poutcome],
    "subscription_probability": [subscription_probability],
    "prediction": ["Yes" if prediction == 1 else "No"],
    "priority": [recommendation]
})

output_path = "model_outputs/customer_prediction.csv"

prediction_result.to_csv(
    output_path,
    index=False
)


# ------------------------------------------------------------
# 8. Completion message
# ------------------------------------------------------------

print("\n==========================================")
print("Prediction completed successfully!")
print(f"Prediction saved to: {output_path}")
print("==========================================\n")
