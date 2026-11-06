import json
import joblib
import streamlit as st
import pandas as pd

from pathlib import Path
from datetime import date, datetime

from config import (
    REPORTS,
    MODEL_FILE,
    METRICS_FILE,
    COMPARISON_FILE,
    PREDICTIONS_FILE,
    FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Detection Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0b1220, #111c32);
    color: #f1f5f9;
}

h1 {
    color: #ffffff !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #60a5fa !important;
    font-weight: 700 !important;
}

p, label, li {
    color: #dbeafe;
}

section[data-testid="stSidebar"] {
    background-color: #101b30;
    border-right: 1px solid #263b5c;
}

/* Metric cards */

div[data-testid="stMetric"] {
    background-color: #172640;
    border: 1px solid #29466d;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    color: #38bdf8;
    font-weight: 800;
}

/* Buttons */

.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {

    background: linear-gradient(90deg, #2563eb, #0891b2) !important;
    color: #ffffff !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 700 !important;
    transition: all 0.25s ease-in-out !important;
}

.stButton > button p,
.stFormSubmitButton > button p,
.stDownloadButton > button p {
    color: #ffffff !important;
    font-weight: 700 !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {

    background: linear-gradient(90deg, #1d4ed8, #0e7490) !important;
    color: #ffffff !important;
    border: 1px solid #7dd3fc !important;
    transform: translateY(-2px);
}

/* Input fields */

.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stTimeInput input,
.stTextArea textarea {

    background-color: #172640;
    color: white;
    border-radius: 8px;
}

/* Dataframes */

div[data-testid="stDataFrame"] {
    border: 1px solid #29466d;
    border-radius: 10px;
}

/* Tabs */

button[data-baseweb="tab"] {
    color: #bfdbfe;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8;
}

hr {
    border-color: #29466d;
}

/* Result cards */

.info-card {
    background-color: #172640;
    border: 1px solid #29466d;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 12px;
}

.success-card {
    background-color: #064e3b;
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 20px;
}

.danger-card {
    background-color: #7f1d1d;
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 20px;
}

/* Comparison section */

.comparison-card {
    background-color: #172640;
    border: 1px solid #29466d;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. HEADER
# ============================================================

st.markdown("""
<style>
.main-header {
    background: linear-gradient(110deg, #172554, #164e63);
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #2563eb;
    margin-bottom: 25px;
}

.main-header h1 {
    color: #ffffff !important;
    font-size: 34px !important;
    font-weight: 800 !important;
    margin: 0 !important;
}
</style>

<div class="main-header">
    <h1>🛡️ Fraud Detection Intelligence</h1>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 4. LOAD PROJECT FILES
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_FILE.exists():
        return None

    return joblib.load(MODEL_FILE)


@st.cache_data
def load_metrics():

    if not METRICS_FILE.exists():
        return None

    with open(METRICS_FILE, "r") as file:
        return json.load(file)


@st.cache_data
def load_comparison():

    if not COMPARISON_FILE.exists():
        return None

    return pd.read_csv(COMPARISON_FILE, index_col=0)


@st.cache_data
def load_predictions():

    if not PREDICTIONS_FILE.exists():
        return None

    return pd.read_csv(PREDICTIONS_FILE)


@st.cache_data
def load_feature_importance():

    file_path = REPORTS / "feature_importance.csv"

    if not file_path.exists():
        return None

    return pd.read_csv(file_path)


model = None
metrics = None
comparison = None
predictions = None
feature_importance = None

try:

    model = load_model()
    metrics = load_metrics()
    comparison = load_comparison()
    predictions = load_predictions()
    feature_importance = load_feature_importance()

except Exception as error:

    st.error(f"Error loading project files: {error}")


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("""
<h2 style="color:#38bdf8;">
🛡️ Fraud Detection
</h2>

<p style="color:#94a3b8;">
Machine Learning Dashboard
</p>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Data Analysis",
        "Model Comparison",
        "Explainability",
        "Fraud Prediction",
        "Metrics Guide"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Project Information

**Project:** Fraud Detection

**Dataset:** Indian Banking Transactions

**Framework:** Streamlit

**ML Library:** Scikit-learn
""")

st.sidebar.markdown("---")

if model is None:
    st.sidebar.error("Saved model not found.")
else:
    st.sidebar.success("ML model loaded successfully.")


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def show_metric(label, value, help_text=None):

    st.metric(
        label=label,
        value=value,
        help=help_text
    )


def get_category_options(feature):

    try:

        encoder = (
            model
            .named_steps["preprocessor"]
            .named_transformers_["categorical"]
            .named_steps["encoder"]
        )

        index = CATEGORICAL_FEATURES.index(feature)

        categories = encoder.categories_[index]

        options = [
            str(value)
            for value in categories
            if pd.notna(value)
        ]

        return sorted(options)

    except Exception:

        return []


def predict_transaction(transaction_df, threshold=0.50):

    probability = model.predict_proba(transaction_df)[0][1]

    prediction = int(probability >= threshold)

    return probability, prediction


def show_image(image_name, caption):

    image_path = REPORTS / image_name

    if image_path.exists():

        st.image(
            str(image_path),
            caption=caption,
            use_container_width=True
        )

    else:

        st.info(f"{image_name} is not available. Generate it first.")


# ============================================================
# 7. INTERACTIVE MODEL COMPARISON TABLE
# ============================================================

def show_model_comparison_table(data, key_prefix="comparison"):

    if data is None or data.empty:

        st.warning("Model comparison data is not available.")
        return

    df = data.copy()

    # Normalize metric names from different source files.

    rename_columns = {
        "average_precision": "PR-AUC",
        "pr_auc": "PR-AUC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-Score",
        "roc_auc": "ROC-AUC"
    }

    df = df.rename(columns=rename_columns)

    # Make sure the model names are displayed as a column.

    if df.index.name is None:
        df.index.name = "Model"

    df.index = df.index.astype(str)
    df.index.name = "Model"

    metric_columns = [
        column
        for column in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "PR-AUC"
        ]
        if column in df.columns
    ]

    if not metric_columns:

        st.warning("No recognized performance metrics were found.")
        return

    # Convert numeric metric columns safely.

    for column in metric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Search and sorting controls.

    st.markdown("#### 🔎 Explore Model Results")

    search_col, sort_col, order_col = st.columns([2, 2, 1])

    with search_col:

        search_text = st.text_input(
            "Search model",
            placeholder="Enter model name...",
            key=f"{key_prefix}_search"
        )

    with sort_col:

        selected_metric = st.selectbox(
            "Rank models by",
            metric_columns,
            index=(
                metric_columns.index("PR-AUC")
                if "PR-AUC" in metric_columns
                else 0
            ),
            key=f"{key_prefix}_metric"
        )

    with order_col:

        sort_order = st.selectbox(
            "Order",
            ["Highest first", "Lowest first"],
            key=f"{key_prefix}_order"
        )

    filtered_df = df.copy()

    # Filter by model name.

    if search_text:

        filtered_df = filtered_df[
            filtered_df.index.str.contains(
                search_text,
                case=False,
                na=False
            )
        ]

    ascending = sort_order == "Lowest first"

    filtered_df = filtered_df.sort_values(
        by=selected_metric,
        ascending=ascending,
        na_position="last"
    )

    if filtered_df.empty:

        st.warning("No models match your search.")
        return

    st.caption(
        f"Showing {len(filtered_df)} of {len(df)} models. "
        "Click any column heading to sort the table."
    )

    # Display percentage values.

    display_df = filtered_df.copy() * 100

    # Colour the metric cells and highlight the best result.

    styled_df = (
        display_df.style
        .format("{:.2f}%")
        .background_gradient(
            cmap="RdYlGn",
            subset=metric_columns,
            axis=0
        )
    )

    # Highlight the highest value in every metric column.

    styled_df = styled_df.highlight_max(
        subset=metric_columns,
        color="#14532d"
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=320,
        key=f"{key_prefix}_table"
    )

    st.caption(
        "Green indicates relatively stronger values and red indicates "
        "relatively weaker values within each metric column. "
        "These colours do not indicate whether a model is suitable "
        "for real-world deployment."
    )

    # Dynamic comparison chart.

    st.markdown("#### 📊 Model Performance Chart")

    chart_data = filtered_df[[selected_metric]].sort_values(
        selected_metric,
        ascending=True
    )

    chart_data = chart_data.rename(
        columns={selected_metric: f"{selected_metric} (%)"}
    ) * 100

    st.bar_chart(
        chart_data,
        horizontal=True
    )


def show_metrics_comparison_from_json(all_results, key_prefix):

    if not all_results:

        st.warning("Model evaluation results are not available.")
        return

    results_df = pd.DataFrame.from_dict(
        all_results,
        orient="index"
    )

    results_df.index.name = "Model"

    show_model_comparison_table(
        results_df,
        key_prefix=key_prefix
    )


# ============================================================
# 8. OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.title("📊 Dashboard Overview")

    st.write(
        "Welcome to the Fraud Detection Intelligence Dashboard. "
        "Explore transaction patterns, model performance and "
        "individual transaction predictions."
    )

    if metrics is None:

        st.warning(
            "Model metrics are not available. Run train.py first."
        )

    else:

        best_model = metrics.get(
            "best_model",
            "Not available"
        )

        evaluation_size = metrics.get(
            "evaluation_sample_size",
            "Not available"
        )

        results = metrics.get(
            "metrics", {}
        ).get(
            best_model, {}
        )

        st.markdown("### 🏆 Model Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            show_metric("Best Model", best_model)

        with col2:
            show_metric("Evaluation Samples", evaluation_size)

        with col3:
            show_metric(
                "PR-AUC",
                f"{results.get('pr_auc', 0):.4f}"
            )

        st.markdown("---")

        st.markdown(
            "### 🤖 Performance of All Machine Learning Models"
        )

        all_results = metrics.get("metrics", {})

        show_metrics_comparison_from_json(
            all_results,
            "overview"
        )

        st.markdown("---")

        st.markdown("### 📈 Selected Model Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            show_metric(
                "Accuracy",
                f"{results.get('accuracy', 0):.2%}"
            )

        with col2:
            show_metric(
                "Precision",
                f"{results.get('precision', 0):.2%}"
            )

        with col3:
            show_metric(
                "Recall",
                f"{results.get('recall', 0):.2%}"
            )

        with col4:
            show_metric(
                "F1 Score",
                f"{results.get('f1_score', 0):.4f}"
            )

        st.markdown("---")

        st.markdown("### 📊 Additional Evaluation Metrics")

        col1, col2 = st.columns(2)

        with col1:
            show_metric(
                "ROC-AUC",
                f"{results.get('roc_auc', 0):.4f}"
            )

        with col2:
            show_metric(
                "PR-AUC",
                f"{results.get('pr_auc', 0):.4f}"
            )

        st.markdown("---")

        st.markdown("### ⚠️ Model Readiness")

        st.warning(
            "This is an experimental fraud detection model. "
            "Its current evaluation results are not strong enough "
            "for real banking deployment. Predictions should not "
            "be treated as confirmed fraud decisions."
        )

    st.markdown("---")

    st.markdown("### 📁 Available Project Reports")

    report_files = [
        "fraud_distribution.png",
        "transaction_amount.png",
        "fraud_by_hour.png",
        "fraud_by_transaction_type.png",
        "model_comparison.png",
        "confusion_matrix.png"
    ]

    available_reports = [
        name for name in report_files
        if (REPORTS / name).exists()
    ]

    if available_reports:

        for report in available_reports:
            st.write(f"✅ {report}")

    else:

        st.info("No report images found. Generate reports first.")


# ============================================================
# 9. DATA ANALYSIS PAGE
# ============================================================

elif page == "Data Analysis":

    st.title("📊 Transaction Data Analysis")

    st.write(
        "Explore transaction distribution and fraud-related "
        "patterns in the dataset."
    )

    st.markdown("### Fraud Distribution")

    show_image(
        "fraud_distribution.png",
        "Distribution of legitimate and fraudulent transactions"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Transaction Amount")

        show_image(
            "transaction_amount.png",
            "Transaction amount distribution"
        )

    with col2:

        st.markdown("### Fraud by Transaction Hour")

        show_image(
            "fraud_by_hour.png",
            "Fraud distribution by transaction hour"
        )

    st.markdown("---")

    st.markdown("### Fraud by Transaction Type")

    show_image(
        "fraud_by_transaction_type.png",
        "Fraud distribution across transaction types"
    )


# ============================================================
# 10. MODEL COMPARISON PAGE
# ============================================================

elif page == "Model Comparison":

    st.title("🤖 Machine Learning Model Comparison")

    st.write(
        "Explore and compare the performance of all trained "
        "machine-learning algorithms."
    )

    if comparison is None:

        st.warning(
            "Model comparison file not found. Run train.py first."
        )

    else:

        st.markdown("### 📋 Interactive Performance Comparison")

        show_model_comparison_table(
            comparison,
            "model_comparison"
        )

        st.markdown("---")

        st.markdown("### 📈 Saved Model Comparison Chart")

        show_image(
            "model_comparison.png",
            "Comparison of machine learning models"
        )

        st.markdown("### Confusion Matrix")

        show_image(
            "confusion_matrix.png",
            "Confusion matrix of the selected model"
        )


# ============================================================
# 11. EXPLAINABILITY PAGE
# ============================================================

elif page == "Explainability":

    st.title("🔍 Model Explainability")

    st.write(
        "Understand which input features influence the "
        "model's fraud detection performance."
    )

    if feature_importance is None:

        st.warning(
            "Feature importance report not found. "
            "Run explain.py first."
        )

    else:

        st.markdown("### Feature Importance")

        st.dataframe(
            feature_importance,
            use_container_width=True
        )

        if (
            "feature" in feature_importance.columns
            and "mean_importance" in feature_importance.columns
        ):

            chart_data = (
                feature_importance
                .set_index("feature")[["mean_importance"]]
                .sort_values(
                    "mean_importance",
                    ascending=True
                )
            )

            st.markdown("### Importance Chart")

            st.bar_chart(chart_data)

        elif (
            "feature" in feature_importance.columns
            and "importance" in feature_importance.columns
        ):

            chart_data = (
                feature_importance
                .set_index("feature")[["importance"]]
                .sort_values(
                    "importance",
                    ascending=True
                )
            )

            st.markdown("### Importance Chart")

            st.bar_chart(chart_data)

        st.info(
            "Feature importance describes how much model performance "
            "changes when a feature is shuffled. It does not prove "
            "that a feature causes fraud."
        )


# ============================================================
# 12. FRAUD PREDICTION PAGE
# ============================================================

elif page == "Fraud Prediction":

    st.title("🚨 Fraud Prediction Center")

    st.write(
        "Enter transaction details to obtain a prediction "
        "from the saved machine-learning model."
    )

    if model is None:

        st.error(
            "The trained model is unavailable. Run train.py first."
        )

    else:

        prediction_tab, batch_tab = st.tabs(
            [
                "📝 Single Transaction",
                "📂 Batch Prediction"
            ]
        )

        # ----------------------------------------------------
        # SINGLE TRANSACTION
        # ----------------------------------------------------

        with prediction_tab:

            st.markdown("### Enter Transaction Information")

            st.caption(
                "Enter transaction details below. The transaction "
                "date and time are used to calculate the model's "
                "date and hour features."
            )

            with st.form("transaction_form"):

                col1, col2 = st.columns(2)

                with col1:

                    transaction_amount = st.number_input(
                        "Transaction Amount",
                        min_value=0.0,
                        value=1000.0,
                        step=100.0
                    )

                    credit_score = st.number_input(
                        "Credit Score",
                        min_value=0.0,
                        max_value=1000.0,
                        value=700.0,
                        step=1.0
                    )

                    has_loan_option = st.selectbox(
                        "Does the customer have a loan?",
                        ["No", "Yes"]
                    )

                    emi_amount = st.number_input(
                        "EMI Amount",
                        min_value=0.0,
                        value=0.0,
                        step=100.0
                    )

                    transaction_date = st.date_input(
                        "Transaction Date",
                        value=date.today()
                    )

                with col2:

                    transaction_time = st.time_input(
                        "Transaction Time",
                        value=datetime.now().time().replace(
                            second=0,
                            microsecond=0
                        )
                    )

                    account_type_options = get_category_options(
                        "account_type"
                    )

                    transaction_type_options = get_category_options(
                        "transaction_type"
                    )

                    direction_options = get_category_options(
                        "transaction_direction"
                    )

                    merchant_options = get_category_options(
                        "merchant_category"
                    )

                    state_options = get_category_options("state")

                    channel_options = get_category_options("channel")

                    kyc_options = get_category_options("kyc_status")

                    account_type = st.selectbox(
                        "Account Type",
                        account_type_options
                    )

                    transaction_type = st.selectbox(
                        "Transaction Type",
                        transaction_type_options
                    )

                    transaction_direction = st.selectbox(
                        "Transaction Direction",
                        direction_options
                    )

                    merchant_category = st.selectbox(
                        "Merchant Category",
                        merchant_options
                    )

                col3, col4 = st.columns(2)

                with col3:

                    state = st.selectbox(
                        "State",
                        state_options
                    )

                    channel = st.selectbox(
                        "Transaction Channel",
                        channel_options
                    )

                with col4:

                    kyc_status = st.selectbox(
                        "KYC Status",
                        kyc_options
                    )

                    loan_type_options = get_category_options(
                        "loan_type"
                    )

                    loan_type = st.selectbox(
                        "Loan Type",
                        loan_type_options,
                        index=0,
                        disabled=(has_loan_option == "No")
                    )

                st.markdown("---")

                threshold = st.slider(
                    "Fraud Classification Threshold",
                    min_value=0.05,
                    max_value=0.95,
                    value=0.50,
                    step=0.05,
                    help=(
                        "A lower threshold flags more transactions "
                        "as potentially fraudulent. It may also "
                        "increase false alarms."
                    )
                )

                submit_prediction = st.form_submit_button(
                    "🔍 Analyze Transaction",
                    use_container_width=True,
                    type="primary"
                )

            if submit_prediction:

                if not all([
                    account_type_options,
                    transaction_type_options,
                    direction_options,
                    merchant_options,
                    state_options,
                    channel_options,
                    kyc_options
                ]):

                    st.error(
                        "Could not retrieve all trained categorical "
                        "values from the saved model."
                    )

                else:

                    transaction_hour = transaction_time.hour

                    transaction_month = transaction_date.month

                    transaction_day_of_week = (
                        transaction_date.weekday()
                    )

                    has_loan = (
                        1 if has_loan_option == "Yes" else 0
                    )

                    if has_loan == 0:
                        loan_type = None

                    transaction_data = {
                        "transaction_amount": transaction_amount,
                        "credit_score": credit_score,
                        "has_loan": has_loan,
                        "emi_amount": emi_amount,
                        "transaction_hour": transaction_hour,
                        "transaction_month": transaction_month,
                        "transaction_day_of_week": transaction_day_of_week,
                        "account_type": account_type,
                        "transaction_type": transaction_type,
                        "transaction_direction": transaction_direction,
                        "merchant_category": merchant_category,
                        "state": state,
                        "loan_type": loan_type,
                        "channel": channel,
                        "kyc_status": kyc_status
                    }

                    transaction_df = pd.DataFrame(
                        [transaction_data],
                        columns=FEATURES
                    )

                    try:

                        probability, prediction = predict_transaction(
                            transaction_df,
                            threshold
                        )

                        st.markdown("---")

                        st.markdown("### Prediction Result")

                        result_col1, result_col2 = st.columns(2)

                        with result_col1:

                            st.metric(
                                "Fraud Probability",
                                f"{probability:.2%}"
                            )

                        with result_col2:

                            st.metric(
                                "Classification Threshold",
                                f"{threshold:.0%}"
                            )

                        st.progress(
                            min(max(float(probability), 0.0), 1.0)
                        )

                        if prediction == 1:

                            st.markdown("""
                            <div class="danger-card">
                                <h2 style="color:#fecaca;">
                                    ⚠️ Potential Fraud Detected
                                </h2>
                                <p style="color:#fee2e2;">
                                    The model classified this transaction
                                    as potentially fraudulent.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                        else:

                            st.markdown("""
                            <div class="success-card">
                                <h2 style="color:#a7f3d0;">
                                    ✅ Classified as Legitimate
                                </h2>
                                <p style="color:#d1fae5;">
                                    The model classified this transaction
                                    as legitimate.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("---")

                        st.markdown("### Transaction Summary")

                        st.dataframe(
                            transaction_df,
                            use_container_width=True
                        )

                        st.caption(
                            "This is a model prediction, not a confirmed "
                            "fraud investigation result. The probability "
                            "is model-estimated and may not be calibrated."
                        )

                    except Exception as error:

                        st.error(f"Prediction failed: {error}")

        # ----------------------------------------------------
        # BATCH PREDICTION
        # ----------------------------------------------------

        with batch_tab:

            st.markdown("### Upload Transaction CSV")

            st.write(
                "Upload a CSV file containing the same features "
                "used during model training."
            )

            st.info(
                "Your CSV must contain all 15 columns from the "
                "model's FEATURES list, including transaction_month "
                "and transaction_day_of_week."
            )

            uploaded_file = st.file_uploader(
                "Choose transaction CSV",
                type=["csv"]
            )

            batch_threshold = st.slider(
                "Batch Prediction Threshold",
                min_value=0.05,
                max_value=0.95,
                value=0.50,
                step=0.05,
                key="batch_threshold"
            )

            if uploaded_file is not None:

                try:

                    batch_df = pd.read_csv(uploaded_file)

                    missing_columns = [
                        column
                        for column in FEATURES
                        if column not in batch_df.columns
                    ]

                    if missing_columns:

                        st.error(
                            "Missing required columns: "
                            + ", ".join(missing_columns)
                        )

                    else:

                        st.markdown("### Uploaded Data Preview")

                        st.dataframe(
                            batch_df.head(10),
                            use_container_width=True
                        )

                        if st.button(
                            "🚀 Predict All Transactions",
                            use_container_width=True,
                            type="primary"
                        ):

                            input_df = batch_df[FEATURES].copy()

                            for column in NUMERIC_FEATURES:

                                input_df[column] = pd.to_numeric(
                                    input_df[column],
                                    errors="coerce"
                                )

                            probabilities = model.predict_proba(
                                input_df
                            )[:, 1]

                            predictions_batch = (
                                probabilities >= batch_threshold
                            ).astype(int)

                            output_df = batch_df.copy()

                            output_df["fraud_probability"] = probabilities

                            output_df["predicted"] = predictions_batch

                            output_df["prediction_result"] = (
                                output_df["predicted"].map({
                                    0: "Legitimate",
                                    1: "Potential Fraud"
                                })
                            )

                            st.markdown("---")

                            st.markdown("### Batch Prediction Results")

                            col1, col2, col3 = st.columns(3)

                            with col1:

                                show_metric(
                                    "Total Transactions",
                                    len(output_df)
                                )

                            with col2:

                                show_metric(
                                    "Potential Fraud",
                                    int(predictions_batch.sum())
                                )

                            with col3:

                                show_metric(
                                    "Legitimate",
                                    int((predictions_batch == 0).sum())
                                )

                            st.dataframe(
                                output_df,
                                use_container_width=True
                            )

                            csv_data = output_df.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                label="⬇️ Download Prediction Results",
                                data=csv_data,
                                file_name="fraud_predictions.csv",
                                mime="text/csv",
                                use_container_width=True,
                                type="primary"
                            )

                except Exception as error:

                    st.error(
                        f"Could not process the uploaded CSV: {error}"
                    )


# ============================================================
# 13. METRICS GUIDE PAGE
# ============================================================

elif page == "Metrics Guide":

    st.title("📚 Understanding Model Evaluation Metrics")

    st.write(
        "Learn how the fraud detection model is evaluated."
    )

    st.markdown("### Accuracy")

    st.write(
        "The percentage of all transactions classified correctly. "
        "Accuracy can be misleading when fraud cases are rare."
    )

    st.markdown("### Precision")

    st.write(
        "Among the transactions predicted as fraud, the proportion "
        "that were actually fraud."
    )

    st.markdown("### Recall")

    st.write(
        "Among the actual fraudulent transactions, the proportion "
        "that the model successfully detected."
    )

    st.markdown("### F1 Score")

    st.write(
        "The harmonic mean of precision and recall."
    )

    st.markdown("### ROC-AUC")

    st.write(
        "Measures how well the model ranks positive cases above "
        "negative cases across different thresholds."
    )

    st.markdown("### PR-AUC / Average Precision")

    st.write(
        "Summarizes the precision-recall trade-off. It is especially "
        "useful for evaluating fraud detection when the dataset "
        "has a highly imbalanced target."
    )

    st.info(
        "For this project, PR-AUC is an important metric because "
        "fraudulent transactions represent a small proportion "
        "of the dataset."
    )


# ============================================================
# 14. FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:15px;">

    <p style="color:#64748b; font-size:13px;">
        🛡️ Fraud Detection Intelligence Dashboard
    </p>

    <p style="color:#64748b; font-size:12px;">
        Machine Learning Project | Built with Python and Streamlit
    </p>

</div>
""", unsafe_allow_html=True)