
import json
import joblib

import pandas as pd
import streamlit as st

import plots
import explain

from config import ARTIFACTS, FEATURES, TARGET


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("Credit Card Fraud Detection")
st.write(
    "Machine Learning Project covering data preparation, "
    "EDA, regression and classification."
)


# ---------------- LOAD PROJECT FILES ----------------

@st.cache_data
def load_results():

    with open(ARTIFACTS / "results.json", "r") as file:
        return json.load(file)


@st.cache_data
def load_train():

    return pd.read_csv(ARTIFACTS / "train.csv")


@st.cache_data
def load_test_scores():

    return pd.read_csv(ARTIFACTS / "test_scores.csv")


@st.cache_resource
def load_model():

    return joblib.load(
        ARTIFACTS / "best_model.joblib"
    )


try:

    results = load_results()
    train_data = load_train()
    test_scores = load_test_scores()
    model = load_model()

except Exception as error:

    st.error("Project artifacts are not available.")
    st.write("First run: python train.py")
    st.exception(error)
    st.stop()


# ---------------- SIDEBAR ----------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Data Analysis",
        "Classification",
        "Regression",
        "Explainability",
        "Fraud Prediction",
        "Metrics Guide"
    ]
)


# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.header("Project Overview")

    meta = results["meta"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Original Dataset Rows",
        f"{meta['raw_rows']:,}"
    )

    col2.metric(
        "Study Dataset Rows",
        f"{meta['study_rows']:,}"
    )

    col3.metric(
        "Best Model",
        results["best_model"]
    )

    st.subheader("Project Information")

    st.write("Data source:", meta["data_source"])
    st.write("Training date:", meta["trained_at"])
    st.write("Python version:", meta["python_version"])
    st.write("Scikit-learn version:", meta["sklearn_version"])

    st.subheader("Data Cleaning")

    st.json(results["cleaning"])

    st.subheader("Dataset Class Distribution")

    st.pyplot(
        plots.class_distribution(
            results["eda"]["class_counts"]
        )
    )

    st.warning(
        "The study dataset has been sampled to change the fraud "
        "percentage. Its fraud prevalence is not the original "
        "real-world prevalence."
    )


# ---------------- DATA ANALYSIS ----------------

elif page == "Data Analysis":

    st.header("Exploratory Data Analysis")

    eda_results = results["eda"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        st.pyplot(
            plots.class_distribution(
                eda_results["class_counts"]
            )
        )

    with col2:
        st.subheader("Transaction Amount")
        st.pyplot(
            plots.amount_by_class(train_data)
        )

    st.subheader("Correlation with Fraud")

    st.pyplot(
        plots.target_correlation(
            eda_results["corr_with_class"]
        )
    )

    st.subheader("Feature Correlation Heatmap")

    st.pyplot(
        plots.heatmap(
            train_data,
            FEATURES[:10]
        )
    )

    st.subheader("IQR Outlier Detection")

    st.pyplot(
        plots.iqr_outliers(
            eda_results["iqr_outliers_per_feature"]
        )
    )

    st.subheader("Isolation Forest")

    st.json(eda_results["isolation_forest"])

    st.write(eda_results["note"])

    st.subheader("Missing Values")

    st.json(eda_results["missing_values"])

    st.subheader("Highly Correlated Feature Pairs")

    st.write(eda_results["redundant_pairs"])


# ---------------- CLASSIFICATION ----------------

elif page == "Classification":

    st.header("Classification Model Comparison")

    classification = results["classification"]

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "cv_pr_auc"
    ]

    table = pd.DataFrame.from_dict(
        classification,
        orient="index"
    )

    st.dataframe(
        table[metric_columns].round(4),
        use_container_width=True
    )

    st.subheader("Model Comparison")

    st.pyplot(
        plots.model_comparison(table)
    )

    st.subheader("ROC Curves")

    st.pyplot(
        plots.roc_curves(classification)
    )

    st.subheader("Precision-Recall Curves")

    st.pyplot(
        plots.pr_curves(classification)
    )

    best_name = results["best_model"]

    st.subheader("Best Model Confusion Matrix")

    st.pyplot(
        plots.confusion(
            classification[best_name]["confusion"],
            best_name
        )
    )

    st.subheader("Imbalance Handling Comparison")

    imbalance_table = pd.DataFrame(
        results["imbalance"]
    ).T

    st.dataframe(
        imbalance_table.round(4),
        use_container_width=True
    )

    st.pyplot(
        plots.imbalance_comparison(
            imbalance_table
        )
    )


# ---------------- REGRESSION ----------------

elif page == "Regression":

    st.header("Regression Experiments")

    regression = results["regression"]

    target = st.selectbox(
        "Select regression target",
        list(regression.keys())
    )

    table = pd.DataFrame(
        regression[target]
    ).T

    st.subheader(f"Results for {target}")

    st.dataframe(
        table.round(4),
        use_container_width=True
    )

    st.pyplot(
        plots.regression_comparison(
            table,
            target
        )
    )

    st.info(
        "Predicting the binary Class column using regression "
        "is included as a syllabus demonstration. It is not "
        "a replacement for the classification models."
    )


# ---------------- EXPLAINABILITY ----------------

elif page == "Explainability":

    st.header("Model Explainability")

    st.subheader("Permutation Feature Importance")

    st.pyplot(
        plots.permutation_importance(
            results["permutation"]
        )
    )

    st.subheader("Explain One Transaction")

    sample = train_data.sample(
        n=1,
        random_state=42
    )

    medians = results["feature_stats"]["median"]

    probability, contributions = explain.occlusion_explain(
        model,
        sample,
        medians
    )

    st.metric(
        "Original Fraud Probability",
        f"{probability:.4f}"
    )

    st.pyplot(
        plots.local_explanation(contributions)
    )

    st.caption(
        "This explanation measures how the model's prediction "
        "changes when one feature is replaced by its training "
        "median. It does not establish causation."
    )


# ---------------- FRAUD PREDICTION ----------------

elif page == "Fraud Prediction":

    st.header("Predict New Transactions")

    st.write(
        "Upload a CSV containing all required feature columns. "
        "The CSV does not need a Class column."
    )

    threshold = st.slider(
        "Fraud Probability Threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        new_data = pd.read_csv(uploaded_file)

        missing = [
            col for col in FEATURES
            if col not in new_data.columns
        ]

        if missing:

            st.error(
                f"Missing required features: {missing}"
            )

        else:

            new_data = new_data[FEATURES].copy()

            for col in FEATURES:
                new_data[col] = pd.to_numeric(
                    new_data[col],
                    errors="coerce"
                )

            st.write("Uploaded transactions:", len(new_data))

            if new_data.empty:
                st.warning("The uploaded CSV contains no transactions.")

            else:

                scores = model.predict_proba(new_data)[:, 1]

                predictions = (
                    scores >= threshold
                ).astype(int)

                output = pd.DataFrame({
                    "Fraud_Probability": scores,
                    "Prediction": [
                        "Fraud" if value == 1 else "Legitimate"
                        for value in predictions
                    ]
                })

                st.subheader("Prediction Results")

                st.dataframe(
                    output,
                    use_container_width=True
                )

                fraud_count = int(predictions.sum())

                col1, col2 = st.columns(2)

                col1.metric(
                    "Fraud Transactions",
                    fraud_count
                )

                col2.metric(
                    "Legitimate Transactions",
                    len(predictions) - fraud_count
                )

                st.download_button(
                    "Download Prediction Results",
                    output.to_csv(index=False),
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

    st.divider()

    st.subheader("Threshold Cost Analysis")

    cost_missed = st.number_input(
        "Cost of missing one fraud transaction",
        min_value=0.0,
        value=1000.0
    )

    cost_false = st.number_input(
        "Cost of one false alarm",
        min_value=0.0,
        value=100.0
    )

    thresholds, costs, optimal = explain.cost_curve(
        test_scores["y_true"],
        test_scores["score"],
        cost_missed,
        cost_false
    )

    st.metric(
        "Minimum Cost Threshold",
        f"{optimal:.2f}"
    )

    st.pyplot(
        plots.cost_curve(
            thresholds,
            costs,
            threshold,
            optimal
        )
    )

    st.warning(
        "The cost curve uses the rebalanced test dataset. "
        "Its minimum-cost threshold may not be suitable for "
        "real-world deployment."
    )


# ---------------- METRICS GUIDE ----------------

elif page == "Metrics Guide":

    st.header("Machine Learning Evaluation Metrics")

    st.subheader("Classification Metrics")

    st.markdown("""
    - **Accuracy:** Overall proportion of correct predictions.
    - **Precision:** Proportion of predicted fraud transactions
      that are actually fraud.
    - **Recall:** Proportion of actual fraud transactions detected.
    - **F1-score:** Harmonic mean of precision and recall.
    - **ROC-AUC:** Measures ranking performance across classification
      thresholds.
    - **PR-AUC:** Measures the precision-recall trade-off.
    """)

    st.subheader("Confusion Matrix")

    st.markdown("""
    | Actual / Predicted | Legitimate | Fraud |
    |---|---:|---:|
    | Legitimate | True Negative | False Positive |
    | Fraud | False Negative | True Positive |
    """)

    st.subheader("Regression Metrics")

    st.markdown("""
    - **MAE:** Mean Absolute Error.
    - **MSE:** Mean Squared Error.
    - **RMSE:** Root Mean Squared Error.
    - **R²:** Coefficient of Determination.
    """)

    st.subheader("Project Model")

    st.write("Selected model:", results["best_model"])
