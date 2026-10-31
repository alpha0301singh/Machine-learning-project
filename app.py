"""Streamlit frontend.   streamlit run app.py
Reads only ./artifacts (produced by train.py) - the big raw CSV is NOT needed to deploy."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from common import FEATURES  # noqa: F401  (also lets joblib unpickle log_amount)

ART = Path(__file__).parent / "artifacts"
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")


@st.cache_data
def load_json(name):
    return json.loads((ART / name).read_text())


@st.cache_data
def load_csv(name):
    return pd.read_csv(ART / name)


@st.cache_resource
def load_model():
    return joblib.load(ART / "best_model.joblib")


if not (ART / "metrics.json").exists():
    st.error("No artifacts found. Run `python train.py --data data/creditcard_2023.csv` first, "
             "then commit the `artifacts/` folder.")
    st.stop()

M, EDA = load_json("metrics.json"), load_json("eda.json")
CLS, CFG = M["classification"], M["config"]
CLS_TABLE = pd.DataFrame(CLS).T[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "cv_pr_auc"]].astype(float)

page = st.sidebar.radio("Navigate", ["Overview", "EDA", "Classification", "Regression", "Live prediction"])
st.sidebar.caption("M.Tech ML project - academic demo, not a production fraud system.")


# --------------------------------------------------------------------- pages
def overview():
    st.title("💳 Credit Card Fraud Detection")
    st.write("End-to-end ML study on the **Credit Card Fraud Detection Dataset 2023** "
             "(anonymised European cardholder transactions): data preparation, EDA and feature "
             "engineering, regression, and classification, with a deployable fraud scorer.")
    c = st.columns(4)
    c[0].metric("Rows in study set", f"{CFG['study_rows']:,}")
    c[1].metric("Fraud share", f"{CFG['fraud_rate']:.2%}")
    c[2].metric("Deployed model", M["best_model"])
    c[3].metric("Its test PR-AUC", f"{CLS[M['best_model']]['pr_auc']:.3f}")
    st.subheader("Pipeline")
    st.markdown(
        "1. **Load & clean** - drop `id`, remove duplicates, validate schema\n"
        "2. **Re-balance** - the raw file is ~50/50, so fraud is down-sampled to a realistic rate\n"
        "3. **Preprocess** - median imputation, `log(1+Amount)`, standard scaling (one `ColumnTransformer`)\n"
        "4. **Feature selection** - `SelectKBest` (ANOVA F-score) inside the pipeline\n"
        "5. **Imbalance handling** - none vs class weights vs under-sampling vs SMOTE\n"
        "6. **Models** - LR, KNN, Decision Tree, Random Forest, SVM, Naive Bayes + tuned Random Forest\n"
        "7. **Selection** - best *cross-validated* PR-AUC (not the test set), then threshold tuning in the app")
    st.info("With ~2% fraud, plain accuracy is misleading (predicting 'legit' for everything gives ~98%). "
            "Judge models on **recall, precision, F1 and PR-AUC**.")


def eda():
    st.header("Exploratory data analysis")
    sample = load_csv("eda_sample.csv")
    sample["Label"] = sample["Class"].map({0: "Legitimate", 1: "Fraud"})
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Class distribution")
        st.bar_chart(pd.Series(EDA["class_counts"]))
        st.caption(f"Missing values in features: {EDA['missing_values']}")
    with c2:
        st.subheader("log(1 + Amount) by class")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(x=np.log1p(sample["Amount"]), hue=sample["Label"], bins=40, stat="density",
                     common_norm=False, ax=ax)
        ax.set_xlabel("log(1 + Amount)")
        st.pyplot(fig)
    st.subheader("Correlation with the fraud label")
    corr = pd.Series(EDA["corr_with_class"])
    st.bar_chart(corr.head(15))
    top = list(corr.head(10).index) + ["Class"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(sample[top].corr(), cmap="coolwarm", center=0, annot=True, fmt=".1f", ax=ax,
                annot_kws={"size": 7})
    st.pyplot(fig)
    st.subheader("Outlier detection")
    iso = EDA["isolation_forest"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Isolation Forest flagged", f"{iso['flagged']:,}")
    c2.metric("Precision vs fraud", f"{iso['precision']:.2%}")
    c3.metric("Recall vs fraud", f"{iso['recall']:.2%}")
    st.caption("IQR-rule outlier counts per feature (top 10):")
    st.bar_chart(pd.Series(EDA["iqr_outliers"]).head(10))


def classification():
    st.header("Classification results")
    st.caption(f"Held-out test set: {CFG['test_rows']:,} transactions. Best value per column is highlighted.")
    st.dataframe(CLS_TABLE.style.highlight_max(axis=0, color="#cfe8cf").format("{:.4f}"))
    t1, t2, t3, t4 = st.tabs(["ROC curves", "Confusion matrix", "Imbalance handling", "Feature importance"])
    with t1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for n, r in CLS.items():
            ax.plot(r["roc"]["fpr"], r["roc"]["tpr"], label=f"{n} ({r['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=.8)
        ax.set(xlabel="False positive rate", ylabel="True positive rate")
        ax.legend(fontsize=7)
        st.pyplot(fig)
    with t2:
        name = st.selectbox("Model", list(CLS), index=list(CLS).index(M["best_model"]))
        cm = np.array(CLS[name]["confusion"])
        fig, ax = plt.subplots(figsize=(4, 3.2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        ax.set(xlabel="Predicted", ylabel="Actual")
        st.pyplot(fig)
        st.caption("Missed fraud (bottom-left) is usually costlier than a false alarm (top-right).")
    with t3:
        imb = pd.DataFrame(M["imbalance"]).T
        st.dataframe(imb.style.highlight_max(axis=0, color="#cfe8cf").format("{:.3f}"))
        if not CFG["smote_available"]:
            st.caption("SMOTE row missing: install `imbalanced-learn` and re-run train.py.")
        st.write("Tuned Random Forest:", M["tuning"]["best_params"])
    with t4:
        imp = pd.Series(load_json("feature_importance.json")).head(15)
        st.bar_chart(imp)


def regression():
    st.header("Regression results")
    target = st.radio("Target", list(M["regression"]), horizontal=True,
                      format_func=lambda t: {"Amount": "Transaction amount", "Class": "Fraud risk score (0/1 target)"}[t])
    df = pd.DataFrame(M["regression"][target]).T
    st.dataframe(df.style.format("{:.4g}", na_rep="-"))
    st.bar_chart(df[["R2", "R2_train"]])
    if target == "Amount":
        st.info("R² near zero means the anonymised features carry almost no information about the amount. "
                "That is a finding, not a bug. Compare train vs test R² for the polynomial model to see "
                "overfitting, and how Ridge / Lasso / Elastic Net shrink it.")
    else:
        st.info("A linear model on a 0/1 target is a *linear probability model*: useful as a risk score and "
                "for Lasso-based feature selection, but logistic regression is the right classifier.")


def predict():
    st.header("Score a transaction")
    model, stats, imp = load_model(), load_json("feature_stats.json"), load_json("feature_importance.json")
    ts = load_csv("test_scores.csv")
    st.sidebar.subheader("Decision threshold")
    thr = st.sidebar.slider("Flag as fraud if probability ≥", 0.01, 0.99, 0.50, 0.01)

    mode = st.radio("Input", ["Manual", "Sample transaction", "Upload CSV"], horizontal=True)
    if mode == "Manual":
        top = [f for f in imp if f != "Amount"][:8]
        st.caption("The 8 most influential features are adjustable; the rest are held at their median.")
        vals = dict(stats["median"])
        cols = st.columns(4)
        for i, f in enumerate(top):
            lo, hi = float(stats["q01"][f]), float(stats["q99"][f])
            vals[f] = cols[i % 4].slider(f, lo, hi, float(np.clip(stats["median"][f], lo, hi)))
        vals["Amount"] = st.number_input("Amount", min_value=0.0, value=float(stats["median"]["Amount"]))
        show_one(model, pd.DataFrame([vals])[FEATURES], thr)
    elif mode == "Sample transaction":
        sample = load_csv("eda_sample.csv")
        kind = st.radio("Pick a", ["Fraud", "Legitimate"], horizontal=True)
        pool = sample[sample["Class"] == (1 if kind == "Fraud" else 0)].reset_index(drop=True)
        i = st.number_input("Row", 0, len(pool) - 1, 0)
        show_one(model, pool.loc[[i], FEATURES], thr, actual=kind)
    else:
        up = st.file_uploader("CSV with columns V1..V28 and Amount", type="csv")
        if up:
            df = pd.read_csv(up)
            miss = [c for c in FEATURES if c not in df.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                out = df.copy()
                out["fraud_probability"] = model.predict_proba(df[FEATURES])[:, 1]
                out["flagged"] = out["fraud_probability"] >= thr
                st.metric("Flagged transactions", f"{int(out['flagged'].sum())} of {len(out)}")
                st.dataframe(out.sort_values("fraud_probability", ascending=False).head(200))
                st.download_button("Download scored CSV", out.to_csv(index=False), "scored.csv")

    st.divider()
    st.subheader("Threshold & cost analysis (held-out test set)")
    c1, c2 = st.columns(2)
    c_fn = c1.number_input("Cost of a missed fraud", 1, 100000, 500)
    c_fp = c2.number_input("Cost of a false alarm", 1, 100000, 5)
    grid = np.linspace(0.01, 0.99, 99)
    y, s = ts["y_true"].values, ts["score"].values
    fn = np.array([((s < t) & (y == 1)).sum() for t in grid])
    fp = np.array([((s >= t) & (y == 0)).sum() for t in grid])
    cost = c_fn * fn + c_fp * fp
    tp = ((s >= thr) & (y == 1)).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Recall at threshold", f"{tp / max(y.sum(), 1):.1%}")
    c2.metric("Precision at threshold", f"{tp / max((s >= thr).sum(), 1):.1%}")
    c3.metric("Cost-optimal threshold", f"{grid[cost.argmin()]:.2f}")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(grid, cost)
    ax.axvline(thr, color="r", ls="--", label="selected")
    ax.set(xlabel="Threshold", ylabel="Total cost")
    ax.legend()
    st.pyplot(fig)


def show_one(model, row, thr, actual=None):
    p = float(model.predict_proba(row)[0, 1])
    c1, c2 = st.columns(2)
    c1.metric("Fraud probability", f"{p:.1%}")
    (c2.error if p >= thr else c2.success)("🚨 FLAGGED as fraud" if p >= thr else "✅ Looks legitimate")
    st.progress(min(max(p, 0.0), 1.0))
    if actual:
        st.caption(f"Ground truth for this sample: {actual}")


{"Overview": overview, "EDA": eda, "Classification": classification,
 "Regression": regression, "Live prediction": predict}[page]()
