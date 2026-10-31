"""STEP 5 - Streamlit web app.      streamlit run app.py

It only reads the small files in artifacts/ (made by train.py), so the big CSV is NOT needed
to run or deploy it.
"""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import explain
import models
import plots
from config import ARTIFACTS, FEATURES, TARGET

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

RESULTS_FILE = ARTIFACTS / "results.json"
if not RESULTS_FILE.exists():
    st.error("No results found. Run  `python train.py --data data/creditcard_2023.csv`  first "
             "(or `python train.py --demo` for a quick test), then start the app again.")
    st.stop()


@st.cache_data
def load_results():
    return json.loads(RESULTS_FILE.read_text())


@st.cache_data
def load_csv(name):
    return pd.read_csv(ARTIFACTS / name)


R = load_results()
META, CLS, BEST = R["meta"], R["classification"], R["best_model"]
CLS_TABLE = pd.DataFrame(CLS).T[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "cv_pr_auc"]].astype(float)


@st.cache_resource
def load_model():
    """Load the saved model. If that fails (for example a different scikit-learn version on
    the server), quietly retrain the same model from artifacts/train.csv instead."""
    try:
        return joblib.load(ARTIFACTS / "best_model.joblib"), False
    except Exception:
        train = load_csv("train.csv")
        pipe = models.rebuild_best(BEST, R["tuning"]["best_params"])
        return pipe.fit(train[FEATURES], train[TARGET]), True


# ------------------------------------------------------------------------- pages
def overview():
    st.title("💳 Credit Card Fraud Detection")
    st.write("A machine-learning study on the **Credit Card Fraud Detection Dataset 2023** "
             "(anonymised European cardholder transactions): data preparation, EDA, regression, "
             "classification, explainability, and a live fraud scorer.")
    c = st.columns(4)
    c[0].metric("Raw rows", f"{META['raw_rows']:,}")
    c[1].metric("Study set", f"{META['study_rows']:,}", f"{META['fraud_rate']:.1%} fraud")
    c[2].metric("Deployed model", BEST)
    c[3].metric("Its test PR-AUC", f"{CLS[BEST]['pr_auc']:.3f}")
    st.subheader("Pipeline")
    st.markdown(
        "1. **data_prep.py**: clean, re-balance to a realistic fraud rate, stratified train/test split\n"
        "2. **eda.py**: statistics, correlation, outliers (IQR + Isolation Forest), charts\n"
        "3. **models.py**: preprocessing pipeline, 6 classifiers, 6 regression models\n"
        "4. **train.py**: cross-validation, tuning, model selection, explainability, saves artifacts\n"
        "5. **app.py**: this web app")
    st.info("Fraud is rare (about 2%), so plain accuracy is misleading: a model that says 'legitimate' every "
            "time is ~98% accurate. Judge models on **recall, precision, F1 and PR-AUC**.")
    st.subheader("Data cleaning report")
    st.json(R["cleaning"])


def eda_page():
    st.header("Exploratory data analysis")
    E, train = R["eda"], load_csv("train.csv")
    c1, c2 = st.columns(2)
    c1.pyplot(plots.class_distribution(E["class_counts"]))
    c2.metric("Missing values in features", E["missing_values"])
    c2.metric("Amount skewness (before log)", f"{E['amount_skewness']:.2f}")
    st.pyplot(plots.amount_by_class(train))
    st.subheader("Correlation with the fraud label")
    corr = pd.Series(E["corr_with_class"])
    st.pyplot(plots.target_correlation(corr))
    st.pyplot(plots.heatmap(train, list(corr.index[:10]) + [TARGET]))
    st.caption("Features strongly correlated with the label are useful predictors; features strongly "
               "correlated with each other are redundant (multicollinearity hurts linear models).")
    if E["redundant_pairs"]:
        st.write("Near-duplicate feature pairs (|corr| >= 0.8):", E["redundant_pairs"])
    st.subheader("Outlier detection")
    iso = E["isolation_forest"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Isolation Forest flagged", f"{iso['flagged']:,}")
    c2.metric("Precision vs real fraud", f"{iso['precision']:.1%}")
    c3.metric("Recall vs real fraud", f"{iso['recall']:.1%}")
    st.pyplot(plots.iqr_outliers(E["iqr_outliers_per_feature"]))
    st.write("Average number of outlier features per transaction:", E["mean_outlier_features_per_row"])
    st.caption(E["note"])


def classification_page():
    st.header("Classification results")
    st.caption(f"Held-out test set: {META['test_rows']:,} transactions. cv_pr_auc = {META['cv_folds']}-fold "
               "cross-validated PR-AUC on the training set; it is what picked the deployed model.")
    st.dataframe(CLS_TABLE.round(4))
    st.pyplot(plots.model_comparison(CLS_TABLE))
    t1, t2, t3, t4 = st.tabs(["ROC curves", "Precision-Recall", "Confusion matrix", "Imbalance handling"])
    with t1:
        st.pyplot(plots.roc_curves(CLS))
    with t2:
        st.pyplot(plots.pr_curves(CLS))
        st.caption("With rare fraud, the PR curve is more honest than ROC. Top-right is best.")
    with t3:
        name = st.selectbox("Model", list(CLS), index=list(CLS).index(BEST))
        st.pyplot(plots.confusion(CLS[name]["confusion"], name))
        st.caption("Bottom-left = missed fraud (costly). Top-right = false alarms (annoying).")
    with t4:
        st.dataframe(pd.DataFrame(R["imbalance"]).T.round(3))
        st.pyplot(plots.imbalance_comparison(pd.DataFrame(R["imbalance"]).T))
        st.write("Tuned Random Forest parameters:", R["tuning"]["best_params"])


def regression_page():
    st.header("Regression results")
    target = st.radio("Target", list(R["regression"]), horizontal=True,
                      format_func=lambda t: "Transaction amount" if t == "Amount" else "Fraud risk score (0/1 label)")
    df = pd.DataFrame(R["regression"][target]).T
    st.dataframe(df.round(4))
    st.pyplot(plots.regression_comparison(df, target))
    if target == "Amount":
        st.info("R-squared near 0 means the anonymised features say almost nothing about the amount. "
                "That is a finding, not a bug. Compare train and test R-squared for the polynomial model "
                "to see overfitting.")
    else:
        st.info("A linear model on a 0/1 label is a linear probability model: a usable risk score and a "
                "feature-selection tool (Lasso), but logistic regression is the right classifier.")


def explain_page():
    st.header("Explainability: why does the model flag a transaction?")
    model, retrained = load_model()
    st.subheader("Global: which features does the model rely on?")
    st.pyplot(plots.permutation_importance(R["permutation"]))
    st.caption("Each feature is shuffled on the test set; the bigger the drop in PR-AUC, the more the model "
               "depends on it. Features are anonymised, so we know WHICH matter, not what they mean.")
    st.subheader("Local: explain one transaction")
    sample = load_csv("train.csv")
    kind = st.radio("Pick a", ["Fraud", "Legitimate"], horizontal=True)
    pool = sample[sample[TARGET] == (1 if kind == "Fraud" else 0)].reset_index(drop=True)
    i = st.number_input("Row", 0, len(pool) - 1, 0)
    base, contrib = explain.occlusion_explain(model, pool.loc[[int(i)]], R["feature_stats"]["median"])
    st.metric("Predicted fraud probability", f"{base:.1%}")
    st.pyplot(plots.local_explanation(contrib))
    st.caption("Each feature is reset to a typical value to see how the probability changes. "
               "Red pushes towards fraud, blue away.")


def show_one(model, row, thr, truth=None):
    p = float(model.predict_proba(row[FEATURES])[0, 1])
    c1, c2 = st.columns(2)
    c1.metric("Fraud probability", f"{p:.1%}")
    if p >= thr:
        c2.error("FLAGGED as fraud")
    else:
        c2.success("Looks legitimate")
    st.progress(min(max(p, 0.0), 1.0))
    if truth:
        st.caption(f"Ground truth for this sample: {truth}")


def predict_page():
    st.header("Score a transaction")
    model, retrained = load_model()
    if retrained:
        st.caption("Note: the saved model file could not be loaded here, so the model was retrained "
                   "from the saved training set.")
    stats = R["feature_stats"]
    thr = st.sidebar.slider("Flag as fraud if probability >=", 0.01, 0.99, 0.50, 0.01)
    mode = st.radio("Input", ["Manual", "Sample transaction", "Upload CSV"], horizontal=True)

    if mode == "Manual":
        top = [f for f in R["permutation"] if f != "Amount"][:8]
        st.caption("The 8 most influential features are adjustable; the rest stay at their typical value.")
        vals = dict(stats["median"])
        cols = st.columns(4)
        for i, f in enumerate(top):
            lo, hi = float(stats["q01"][f]), float(stats["q99"][f])
            if hi <= lo:
                hi = lo + 1.0
            vals[f] = cols[i % 4].slider(f, lo, hi, float(np.clip(stats["median"][f], lo, hi)))
        vals["Amount"] = st.number_input("Amount", min_value=0.0, value=float(max(stats["median"]["Amount"], 0.0)))
        show_one(model, pd.DataFrame([vals])[FEATURES], thr)
    elif mode == "Sample transaction":
        sample = load_csv("train.csv")
        kind = st.radio("Pick a", ["Fraud", "Legitimate"], horizontal=True, key="pk")
        pool = sample[sample[TARGET] == (1 if kind == "Fraud" else 0)].reset_index(drop=True)
        i = st.number_input("Row", 0, len(pool) - 1, 0, key="pr")
        show_one(model, pool.loc[[int(i)]], thr, truth=kind)
    else:
        up = st.file_uploader("CSV with columns V1..V28 and Amount", type="csv")
        if up is not None:
            df = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                out = df.copy()
                out["fraud_probability"] = model.predict_proba(df[FEATURES])[:, 1]
                out["flagged"] = out["fraud_probability"] >= thr
                st.metric("Flagged transactions", f"{int(out['flagged'].sum())} of {len(out)}")
                st.dataframe(out.sort_values("fraud_probability", ascending=False).head(200))
                st.download_button("Download scored CSV", out.to_csv(index=False), "scored.csv")

    st.divider()
    st.subheader("Threshold and cost analysis (held-out test set)")
    ts = load_csv("test_scores.csv")
    c1, c2 = st.columns(2)
    c_fn = c1.number_input("Cost of a missed fraud", min_value=1, value=500)
    c_fp = c2.number_input("Cost of a false alarm", min_value=1, value=5)
    grid, cost, best_thr = explain.cost_curve(ts["y_true"], ts["score"], c_fn, c_fp)
    m = explain.metrics_at(ts["y_true"], ts["score"], thr)
    c1, c2, c3 = st.columns(3)
    c1.metric("Recall at your threshold", f"{m['recall']:.1%}")
    c2.metric("Precision at your threshold", f"{m['precision']:.1%}")
    c3.metric("Cost-optimal threshold", f"{best_thr:.2f}")
    st.pyplot(plots.cost_curve(grid, cost, thr, best_thr))


def metrics_guide():
    st.header("Metrics guide (with your model's real numbers)")
    (tn, fp), (fn, tp) = CLS[BEST]["confusion"]
    n = tn + fp + fn + tp
    pre, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    st.table(pd.DataFrame([[f"TN = {tn}", f"FP = {fp}"], [f"FN = {fn}", f"TP = {tp}"]],
                          index=["Actual legit", "Actual fraud"], columns=["Predicted legit", "Predicted fraud"]))
    st.markdown(
        f"- **Accuracy** = (TP+TN)/all = {(tp + tn) / n:.3f}. Misleading: always saying 'legit' scores {1 - META['fraud_rate']:.3f}.\n"
        f"- **Precision** = TP/(TP+FP) = {pre:.3f}: of the flagged transactions, how many were really fraud.\n"
        f"- **Recall** = TP/(TP+FN) = {rec:.3f}: of all real fraud, how much we caught.\n"
        f"- **F1** = 2PR/(P+R) = {2 * pre * rec / max(pre + rec, 1e-9):.3f}.\n"
        "- **ROC-AUC**: chance a random fraud scores higher than a random legit transaction (0.5 = coin flip).\n"
        "- **PR-AUC**: area under the precision-recall curve; more honest than ROC-AUC when fraud is rare.\n"
        "- **MAE / MSE / RMSE**: average error / squared error / its square root. **R-squared**: share of variance "
        "explained (0 = no better than the mean).\n"
        "- **Ridge** shrinks coefficients, **Lasso** sets some to exactly zero (feature selection), "
        "**Elastic Net** mixes both.")


# ------------------------------------------------------------------------ router
PAGES = {"Overview": overview, "EDA": eda_page, "Classification": classification_page,
         "Regression": regression_page, "Explainability": explain_page,
         "Live prediction": predict_page, "Metrics guide": metrics_guide}
page = st.sidebar.radio("Go to", list(PAGES))
st.sidebar.caption("M.Tech ML project: academic demo, not a production fraud system.")
if META["data_source"].startswith("synthetic"):
    st.warning("These results come from SYNTHETIC demo data, not the real Kaggle dataset. "
               "Run `python train.py --data data/creditcard_2023.csv` and redeploy for real results.")
PAGES[page]()
