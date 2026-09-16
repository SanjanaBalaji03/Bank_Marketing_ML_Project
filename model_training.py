import pandas as pd
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ============================================================
# MODEL TRAINING
# ============================================================

print("\n===== MODEL TRAINING =====")


# ------------------------------------------------------------
# 1. Load training and testing data
# ------------------------------------------------------------

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")

y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()

print("\nTraining data loaded successfully!")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ------------------------------------------------------------
# 2. Identify numerical and categorical columns
# ------------------------------------------------------------

numerical_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ------------------------------------------------------------
# 3. Preprocessing pipelines
# ------------------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ------------------------------------------------------------
# 4. Define models
# ------------------------------------------------------------

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ------------------------------------------------------------
# 5. Train Logistic Regression
# ------------------------------------------------------------

print("\n===== TRAINING LOGISTIC REGRESSION =====")

logistic_model.fit(
    X_train,
    y_train
)

print("Logistic Regression training completed!")


# ------------------------------------------------------------
# 6. Evaluate Logistic Regression
# ------------------------------------------------------------

logistic_predictions = logistic_model.predict(X_test)

logistic_probabilities = logistic_model.predict_proba(
    X_test
)[:, 1]


logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

logistic_precision = precision_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_roc_auc = roc_auc_score(
    y_test,
    logistic_probabilities
)


print("\nLogistic Regression Results")
print("----------------------------")
print(f"Accuracy:  {logistic_accuracy:.4f}")
print(f"Precision: {logistic_precision:.4f}")
print(f"Recall:    {logistic_recall:.4f}")
print(f"F1 Score:  {logistic_f1:.4f}")
print(f"ROC-AUC:   {logistic_roc_auc:.4f}")


# ------------------------------------------------------------
# 7. Train Random Forest
# ------------------------------------------------------------

print("\n===== TRAINING RANDOM FOREST =====")

random_forest_model.fit(
    X_train,
    y_train
)

print("Random Forest training completed!")


# ------------------------------------------------------------
# 8. Evaluate Random Forest
# ------------------------------------------------------------

rf_predictions = random_forest_model.predict(X_test)

rf_probabilities = random_forest_model.predict_proba(
    X_test
)[:, 1]


rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_precision = precision_score(
    y_test,
    rf_predictions,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_predictions,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_predictions,
    zero_division=0
)

rf_roc_auc = roc_auc_score(
    y_test,
    rf_probabilities
)


print("\nRandom Forest Results")
print("---------------------")
print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1 Score:  {rf_f1:.4f}")
print(f"ROC-AUC:   {rf_roc_auc:.4f}")


# ------------------------------------------------------------
# 9. Classification reports
# ------------------------------------------------------------

print("\n===== LOGISTIC REGRESSION CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        logistic_predictions,
        target_names=["No Subscription", "Subscription"],
        zero_division=0
    )
)


print("\n===== RANDOM FOREST CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=["No Subscription", "Subscription"],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 10. Compare models
# ------------------------------------------------------------

results = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],
        "Accuracy": [
            logistic_accuracy,
            rf_accuracy
        ],
        "Precision": [
            logistic_precision,
            rf_precision
        ],
        "Recall": [
            logistic_recall,
            rf_recall
        ],
        "F1_Score": [
            logistic_f1,
            rf_f1
        ],
        "ROC_AUC": [
            logistic_roc_auc,
            rf_roc_auc
        ]
    }
)


print("\n===== MODEL COMPARISON =====")
print(results.to_string(index=False))


# ------------------------------------------------------------
# 11. Select best model
# ------------------------------------------------------------

# F1 score is used as the main selection metric because
# the target variable is imbalanced.

best_model_name = results.loc[
    results["F1_Score"].idxmax(),
    "Model"
]


if best_model_name == "Logistic Regression":
    best_model = logistic_model
else:
    best_model = random_forest_model


print("\n===== BEST MODEL =====")
print("Selected model:", best_model_name)


# ------------------------------------------------------------
# 12. Create output folders
# ------------------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "model_outputs",
    exist_ok=True
)


# ------------------------------------------------------------
# 13. Save best model
# ------------------------------------------------------------

model_path = "models/best_bank_marketing_model.pkl"

joblib.dump(
    best_model,
    model_path
)

print("\nBest model saved to:")
print(model_path)


# ------------------------------------------------------------
# 14. Save model comparison
# ------------------------------------------------------------

results_path = "model_outputs/model_comparison.csv"

results.to_csv(
    results_path,
    index=False
)

print("Model comparison saved to:")
print(results_path)


# ------------------------------------------------------------
# 15. Save text summary
# ------------------------------------------------------------

summary_path = "model_outputs/model_training_summary.txt"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("BANK MARKETING MODEL TRAINING SUMMARY\n")
    file.write("=" * 50 + "\n\n")

    file.write("MODELS TRAINED\n")
    file.write("-" * 25 + "\n")
    file.write("1. Logistic Regression\n")
    file.write("2. Random Forest\n\n")

    file.write("MODEL PERFORMANCE\n")
    file.write("-" * 25 + "\n")

    for _, row in results.iterrows():

        file.write(f"\n{row['Model']}\n")
        file.write(f"Accuracy: {row['Accuracy']:.4f}\n")
        file.write(f"Precision: {row['Precision']:.4f}\n")
        file.write(f"Recall: {row['Recall']:.4f}\n")
        file.write(f"F1 Score: {row['F1_Score']:.4f}\n")
        file.write(f"ROC-AUC: {row['ROC_AUC']:.4f}\n")

    file.write("\nBEST MODEL\n")
    file.write("-" * 25 + "\n")
    file.write(f"Selected model: {best_model_name}\n")

    file.write(
        "\nModel selection was based primarily on F1 Score "
        "because the target variable is imbalanced.\n"
    )


# ------------------------------------------------------------
# 16. Completion
# ------------------------------------------------------------

print("\n===== MODEL TRAINING COMPLETED SUCCESSFULLY =====")

print("\nFiles created:")
print("- models/best_bank_marketing_model.pkl")
print("- model_outputs/model_comparison.csv")
print("- model_outputs/model_training_summary.txt")