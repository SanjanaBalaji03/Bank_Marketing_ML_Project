import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing ML Dashboard",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Bank Marketing Campaign Prediction Dashboard")
st.markdown(
    "### Customer Subscription Analysis & Machine Learning Insights"
)

st.markdown("---")


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "data/bank_marketing_processed.csv"
MODEL_PATH = "models/best_bank_marketing_model.pkl"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    model = load_model()

except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Dashboard Menu")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Customer Insights",
        "Model Performance",
        "Feature Importance",
        "Customer Prediction",
        "Business Recommendations"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("📊 Project Overview")

    total_customers = len(df)

    subscribed = (
        df["y"].astype(str).str.lower().isin(["yes", "1", "true"]).sum()
    )

    subscription_rate = (subscribed / total_customers) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Subscribed Customers",
        f"{subscribed:,}"
    )

    col3.metric(
        "Subscription Rate",
        f"{subscription_rate:.2f}%"
    )

    col4.metric(
        "Average Age",
        f"{df['age'].mean():.1f}"
    )

    st.markdown("---")

    st.subheader("🎯 Project Objective")

    st.write(
        "The objective of this project is to predict whether a bank customer "
        "will subscribe to a term deposit after a marketing campaign."
    )

    st.subheader("🔄 Machine Learning Workflow")

    st.write(
        "Data Ingestion → Data Preprocessing → EDA → Feature Engineering "
        "→ Train/Test Split → Model Training → Model Evaluation "
        "→ Model Interpretation → Prediction → Business Insights"
    )


# ============================================================
# CUSTOMER INSIGHTS
# ============================================================

elif page == "Customer Insights":

    st.header("👥 Customer Insights")

    st.subheader("Subscription by Contact Method")

    contact_file = "model_outputs/subscription_by_contact.csv"

    if os.path.exists(contact_file):

        contact_df = pd.read_csv(contact_file)

        st.dataframe(
            contact_df,
            use_container_width=True
        )

        if "subscription_rate" in contact_df.columns:

            fig, ax = plt.subplots()

            ax.bar(
                contact_df["contact"],
                contact_df["subscription_rate"]
            )

            ax.set_xlabel("Contact Method")
            ax.set_ylabel("Subscription Rate (%)")
            ax.set_title("Subscription Rate by Contact Method")

            st.pyplot(fig)

    st.markdown("---")

    st.subheader("Subscription by Education")

    education_file = "model_outputs/subscription_by_education.csv"

    if os.path.exists(education_file):

        education_df = pd.read_csv(education_file)

        st.dataframe(
            education_df,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Subscription by Job")

    job_file = "model_outputs/subscription_by_job.csv"

    if os.path.exists(job_file):

        job_df = pd.read_csv(job_file)

        st.dataframe(
            job_df,
            use_container_width=True
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("🤖 Model Performance")

    comparison_file = "model_outputs/model_comparison.csv"

    if os.path.exists(comparison_file):

        comparison_df = pd.read_csv(comparison_file)

        st.subheader("Model Comparison")

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("🏆 Selected Model")

        st.success(
            "Random Forest was selected as the best-performing model."
        )

        cols = st.columns(5)

        metrics = {
            "Accuracy": 0.8807,
            "Precision": 0.4883,
            "Recall": 0.4130,
            "F1 Score": 0.4475,
            "ROC-AUC": 0.7913
        }

        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.4f}")

    st.markdown("---")

    st.subheader("📈 Confusion Matrix")

    confusion_path = "model_outputs/confusion_matrix.png"

    if os.path.exists(confusion_path):

        st.image(
            confusion_path,
            caption="Random Forest Confusion Matrix",
            use_container_width=True
        )

    st.subheader("📈 ROC Curve")

    roc_path = "model_outputs/roc_curve.png"

    if os.path.exists(roc_path):

        st.image(
            roc_path,
            caption="ROC Curve",
            use_container_width=True
        )

    st.subheader("📊 Precision-Recall Curve")

    pr_path = "model_outputs/precision_recall_curve.png"

    if os.path.exists(pr_path):

        st.image(
            pr_path,
            caption="Precision-Recall Curve",
            use_container_width=True
        )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "Feature Importance":

    st.header("🔍 Important Features")

    feature_file = "model_outputs/feature_importance.csv"

    if os.path.exists(feature_file):

        feature_df = pd.read_csv(feature_file)

        st.dataframe(
            feature_df.head(10),
            use_container_width=True
        )

        st.subheader("Top 10 Important Features")

        top_features = feature_df.head(10)

        fig, ax = plt.subplots()

        ax.barh(
            top_features["Feature"][::-1],
            top_features["Importance"][::-1]
        )

        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Top 10 Feature Importance")

        st.pyplot(fig)

        st.markdown("---")

        st.info(
            "The most influential features include account balance, age, "
            "day of contact, campaign contacts and previous campaign activity."
        )


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

elif page == "Customer Prediction":

    st.header("🎯 Customer Subscription Prediction")

    st.write(
        "Enter customer details to estimate the probability of subscribing."
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        job = st.selectbox(
            "Job",
            [
                "admin.",
                "blue-collar",
                "entrepreneur",
                "housemaid",
                "management",
                "retired",
                "self-employed",
                "services",
                "student",
                "technician",
                "unemployed",
                "unknown"
            ]
        )

        marital = st.selectbox(
            "Marital Status",
            ["married", "single", "divorced"]
        )

        education = st.selectbox(
            "Education",
            ["primary", "secondary", "tertiary", "unknown"]
        )

        default = st.selectbox(
            "Has Credit Default?",
            ["no", "yes"]
        )

        balance = st.number_input(
            "Account Balance",
            value=1500
        )

        housing = st.selectbox(
            "Has Housing Loan?",
            ["no", "yes"]
        )

        loan = st.selectbox(
            "Has Personal Loan?",
            ["no", "yes"]
        )

    with col2:

        contact = st.selectbox(
            "Contact Type",
            ["cellular", "telephone", "unknown"]
        )

        day = st.number_input(
            "Day of Month Contacted",
            min_value=1,
            max_value=31,
            value=15
        )

        month = st.selectbox(
            "Month",
            [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec"
            ]
        )

        campaign = st.number_input(
            "Number of Contacts During Current Campaign",
            min_value=1,
            value=1
        )

        pdays = st.number_input(
            "Days Since Previous Contact (-1 if never contacted)",
            value=-1
        )

        previous = st.number_input(
            "Number of Previous Contacts",
            min_value=0,
            value=0
        )

        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            ["failure", "other", "success", "unknown"]
        )

    if st.button("🔮 Predict Subscription"):

        input_data = pd.DataFrame({
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

        try:

            prediction = model.predict(input_data)[0]

            probability = model.predict_proba(input_data)[0][1]

            st.markdown("---")

            if prediction == 1 or str(prediction).lower() == "yes":

                st.success(
                    "🎉 Customer is likely to subscribe!"
                )

            else:

                st.warning(
                    "Customer is less likely to subscribe."
                )

            col1, col2 = st.columns(2)

            col1.metric(
                "Subscription Probability",
                f"{probability * 100:.2f}%"
            )

            col2.metric(
                "Non-Subscription Probability",
                f"{(1 - probability) * 100:.2f}%"
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# ============================================================
# BUSINESS RECOMMENDATIONS
# ============================================================

elif page == "Business Recommendations":

    st.header("💡 Business Recommendations")

    recommendations = [
        "Focus marketing campaigns on customer segments with higher historical subscription rates.",
        "Cellular contact has the highest historical subscription rate and should receive greater attention.",
        "Customers with tertiary education show a relatively higher subscription rate.",
        "Student customers show the highest subscription rate among job categories.",
        "Use the Random Forest model to prioritize customers based on predicted subscription probability.",
        "Avoid spending equal marketing resources on every customer. Use targeted campaigns for high-probability customers."
    ]

    for i, recommendation in enumerate(recommendations, 1):

        st.write(
            f"**{i}.** {recommendation}"
        )

    st.markdown("---")

    st.subheader("🎯 Overall Business Objective")

    st.success(
        "Use machine learning to identify customers who are more likely "
        "to subscribe, allowing the bank to allocate marketing resources "
        "more efficiently."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Bank Marketing ML Project | Business Analytics & Marketing"
)