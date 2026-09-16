import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt


# ============================================================
# MODEL INTERPRETATION
# ============================================================

print("\n===== MODEL INTERPRETATION =====\n")


# ------------------------------------------------------------
# 1. Load the trained model
# ------------------------------------------------------------

model_path = "models/best_bank_marketing_model.pkl"

model = joblib.load(model_path)

print("Best model loaded successfully!")
print("Model type:", type(model).__name__)


# ------------------------------------------------------------
# 2. Identify the Random Forest inside the Pipeline
# ------------------------------------------------------------

if hasattr(model, "named_steps"):

    print("\nModel pipeline steps:")
    print(list(model.named_steps.keys()))

    # Find the model step
    rf_model = None

    for name, step in model.named_steps.items():

        if hasattr(step, "feature_importances_"):
            rf_model = step
            model_step_name = name
            break

    if rf_model is None:
        raise ValueError(
            "Random Forest model could not be found inside the pipeline."
        )

else:

    rf_model = model
    model_step_name = "model"


# ------------------------------------------------------------
# 3. Get feature names
# ------------------------------------------------------------

feature_names = None


if hasattr(model, "named_steps"):

    # Look for preprocessing step
    for name, step in model.named_steps.items():

        if hasattr(step, "get_feature_names_out"):

            try:
                feature_names = step.get_feature_names_out()
                print(
                    "\nFeature names obtained from preprocessing step."
                )
                break

            except Exception:
                pass


# If feature names cannot be obtained automatically,
# use the original dataset columns.

if feature_names is None:

    X_test = pd.read_csv("data/X_test.csv")

    feature_names = X_test.columns.tolist()

    print(
        "\nOriginal feature names used."
    )


# ------------------------------------------------------------
# 4. Extract feature importance
# ------------------------------------------------------------

importance_values = rf_model.feature_importances_


# Make sure lengths match
if len(feature_names) != len(importance_values):

    print(
        "\nWarning: Feature name count does not match "
        "feature importance count."
    )

    # Try to use original feature names
    X_test = pd.read_csv("data/X_test.csv")

    if len(X_test.columns) == len(importance_values):

        feature_names = X_test.columns.tolist()

    else:

        feature_names = [
            f"Feature_{i+1}"
            for i in range(len(importance_values))
        ]


# ------------------------------------------------------------
# 5. Create feature importance dataframe
# ------------------------------------------------------------

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance_values

})


# Sort from highest to lowest
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


# ------------------------------------------------------------
# 6. Display feature importance
# ------------------------------------------------------------

print("\n===== FEATURE IMPORTANCE =====\n")

print(
    importance_df.to_string(index=False)
)


# ------------------------------------------------------------
# 7. Save feature importance CSV
# ------------------------------------------------------------

os.makedirs("model_outputs", exist_ok=True)

importance_df.to_csv(
    "model_outputs/feature_importance.csv",
    index=False
)

print(
    "\nFeature importance saved to:"
)

print(
    "model_outputs/feature_importance.csv"
)


# ------------------------------------------------------------
# 8. Display top 10 features
# ------------------------------------------------------------

top_n = min(10, len(importance_df))

top_features = importance_df.head(top_n)


print("\n===== TOP 10 IMPORTANT FEATURES =====\n")

for i, row in top_features.iterrows():

    print(
        f"{i + 1}. {row['Feature']} "
        f"-> {row['Importance']:.4f}"
    )


# ------------------------------------------------------------
# 9. Create feature importance chart
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Feature Importance")

plt.ylabel("Feature")

plt.title(
    "Random Forest - Top 10 Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "model_outputs/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 10. Create interpretation summary
# ------------------------------------------------------------

summary_text = """
MODEL INTERPRETATION SUMMARY
============================

Model:
Random Forest

Purpose:
Identify which customer characteristics are most important
for predicting term-deposit subscription.

Top Important Features:
"""

for i, row in top_features.iterrows():

    summary_text += (
        f"\n{i + 1}. {row['Feature']}: "
        f"{row['Importance']:.4f}"
    )


summary_text += """

Business Interpretation:
Feature importance indicates how strongly each feature
contributes to the Random Forest's predictions.

The most important features can be used by the bank to
better understand customer response patterns and design
more targeted marketing campaigns.

Feature importance does NOT mean that a feature directly
causes subscription. It indicates its contribution to
the model's predictions.
"""


with open(
    "model_outputs/model_interpretation_summary.txt",
    "w"
) as file:

    file.write(summary_text)


# ------------------------------------------------------------
# 11. Completion message
# ------------------------------------------------------------

print(
    "\n===== MODEL INTERPRETATION COMPLETED SUCCESSFULLY ====="
)

print("\nFiles created:")

print(
    "- model_outputs/feature_importance.csv"
)

print(
    "- model_outputs/feature_importance.png"
)

print(
    "- model_outputs/model_interpretation_summary.txt"
)