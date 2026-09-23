# Bank Marketing ML Project

## 📌 Project Overview

This project uses Machine Learning to predict whether a bank customer is likely to subscribe to a term deposit after a marketing campaign.

The project follows an end-to-end Machine Learning workflow, starting from data understanding and preprocessing to model training, evaluation, prediction and business insights.

## 🎯 Business Problem

Banks spend significant time and resources contacting customers through marketing campaigns. However, not every customer is equally likely to subscribe.

The objective of this project is to identify customers who are more likely to subscribe to a term deposit so that marketing efforts can be better targeted.

## 📊 Dataset

The project uses the UCI Bank Marketing Dataset.

- Total records: 45,211
- Input features: 16
- Target variable: `y`
- Target: Whether the customer subscribed to a term deposit

The dataset contains customer demographic, account and campaign-related information.

## 🔄 Machine Learning Pipeline

The project follows this workflow:

**Data Ingestion → Data Understanding → Preprocessing → EDA → Feature Engineering → Train-Test Split → Model Training → Model Evaluation → Model Interpretation → Prediction → Business Insights → Dashboard**

## 🤖 Models Used

Two classification algorithms were compared:

### 1. Logistic Regression
Used as a baseline classification model.

### 2. Random Forest
An ensemble model consisting of multiple decision trees.

Random Forest was selected as the final model based on the overall evaluation results.

## 📈 Model Performance

### Logistic Regression

- Accuracy: 75.48%
- Precision: 26.62%
- Recall: 62.38%
- F1 Score: 37.32%
- ROC-AUC: 77.22%

### Random Forest

- Accuracy: 88.07%
- Precision: 48.83%
- Recall: 41.30%
- F1 Score: 44.75%
- ROC-AUC: 79.13%

## 💡 Business Insights

The analysis identified differences in subscription rates across customer groups.

Some important predictive variables identified through Random Forest feature importance include:

- Account balance
- Age
- Day of contact
- Campaign contacts
- Previous campaign information
- Contact type

These represent variables useful to the model's predictions and should not be interpreted as causal relationships.

## 📊 Streamlit Dashboard

An interactive Streamlit dashboard was developed with the following sections:

- Overview
- Customer Insights
- Model Performance
- Feature Importance
- Customer Prediction
- Business Recommendations

The dashboard allows users to explore customer patterns, evaluate model performance and generate predictions for individual customers.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Git & GitHub

## 🚀 Future Scope

The project can be extended with:

- Formal feature selection
- Separate validation set / cross-validation
- Feature store
- Data and model versioning
- Model registry
- ChatGPT API for automated business reporting
- Docker containerization
- Cloud deployment
- Model and data drift monitoring

## 👩‍💻 Project

**Bank Marketing Machine Learning Project**

Developed as part of an MBA Business Analytics project.