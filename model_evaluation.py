import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve
)

import matplotlib.pyplot as plt


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n===== MODEL EVALUATION =====\n")


# ------------------------------------------------------------
# 1. Load test data
# ------------------------------------------------------------

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

print("Test data loaded successfully!")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ------------------------------------------------------------
# 2. Load the best trained model
# ------------------------------------------------------------

model_path = "models/best_bank_marketing_model.pkl"

model = joblib.load(model_path)

print("\nBest model loaded successfully!")
print("Model:", type(model).__name__)


# ------------------------------------------------------------
# 3. Make predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test)

# Probability of class 1 (Subscription = Yes)
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)[:, 1]
else:
    y_prob = None


# ------------------------------------------------------------
# 4. Calculate evaluation metrics
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

if y_prob is not None:
    roc_auc = roc_auc_score(y_test, y_prob)
else:
    roc_auc = None


# ------------------------------------------------------------
# 5. Display results
# ------------------------------------------------------------

print("\n===== FINAL MODEL RESULTS =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

if roc_auc is not None:
    print(f"ROC-AUC  : {roc_auc:.4f}")


# ------------------------------------------------------------
# 6. Classification Report
# ------------------------------------------------------------

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Subscription", "Subscription"],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 7. Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\n===== CONFUSION MATRIX =====")
print(cm)


# Create output folder
os.makedirs("model_outputs", exist_ok=True)


# Save confusion matrix chart
plt.figure(figsize=(7, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Subscription", "Subscription"]
)

disp.plot()

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "model_outputs/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 8. ROC Curve
# ------------------------------------------------------------

if y_prob is not None:

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_prob
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"Random Forest (AUC = {roc_auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("Random Forest - ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "model_outputs/roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 9. Precision-Recall Curve
# ------------------------------------------------------------

if y_prob is not None:

    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_test,
            y_prob
        )
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        recall_values,
        precision_values
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        "Random Forest - Precision-Recall Curve"
    )

    plt.tight_layout()

    plt.savefig(
        "model_outputs/precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 10. Save evaluation summary
# ------------------------------------------------------------

summary = f"""
MODEL EVALUATION SUMMARY
========================

Selected Model:
Random Forest

Test Dataset:
{X_test.shape[0]} customers

Accuracy:
{accuracy:.4f}

Precision:
{precision:.4f}

Recall:
{recall:.4f}

F1 Score:
{f1:.4f}

ROC-AUC:
{roc_auc:.4f}

Confusion Matrix:
{cm}

Business Interpretation:
The model is evaluated based on its ability to identify
customers who are likely to subscribe to the term deposit.

Because the target variable is imbalanced, F1-score,
precision, recall and ROC-AUC are considered along with
accuracy.

The model can help the bank identify customers who are
more likely to respond positively to marketing campaigns.
"""

with open(
    "model_outputs/model_evaluation_summary.txt",
    "w"
) as file:

    file.write(summary)


# ------------------------------------------------------------
# 11. Completion message
# ------------------------------------------------------------

print("\n===== MODEL EVALUATION COMPLETED SUCCESSFULLY =====")

print("\nFiles created:")

print("- model_outputs/confusion_matrix.png")
print("- model_outputs/roc_curve.png")
print("- model_outputs/precision_recall_curve.png")
print("- model_outputs/model_evaluation_summary.txt")