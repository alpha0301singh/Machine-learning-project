"""Run the whole ML study and write everything the Streamlit app needs to ./artifacts

    python train.py --data data/creditcard_2023.csv
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import (ElasticNetCV, LassoCV, LinearRegression,
                                  LogisticRegression, RidgeCV)
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import FEATURES, TARGET, V_COLS, load_data, make_prep

try:  # optional: SMOTE for the imbalance study
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTE = None

warnings.filterwarnings("ignore")
ART = Path("artifacts")
FS_K = 20  # features kept by SelectKBest


def dump(name, obj):
    (ART / name).write_text(json.dumps(
        obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ---------------------------------------------------------------- Unit I / II
def build_imbalanced(df, n_legit, fraud_rate, seed):
    """The 2023 Kaggle data is (near) 50/50. Real fraud is rare, so we re-create a
    realistic imbalance by keeping all sampled legit rows and only a few fraud rows."""
    legit, fraud = df[df[TARGET] == 0], df[df[TARGET] == 1]
    n_legit = min(n_legit, len(legit))
    n_fraud = min(int(round(n_legit * fraud_rate / (1 - fraud_rate))), len(fraud))
    out = pd.concat([legit.sample(n_legit, random_state=seed),
                     fraud.sample(n_fraud, random_state=seed)])
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def run_eda(df, seed):
    X, y = df[FEATURES], df[TARGET]
    q1, q3 = X.quantile(.25), X.quantile(.75)
    iqr = q3 - q1
    iqr_out = ((X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)).sum()
    iso = IsolationForest(n_estimators=200, contamination=float(y.mean()),
                          random_state=seed, n_jobs=-1).fit(X)
    flag = (iso.predict(X) == -1).astype(int)
    corr = df.corr()[TARGET].drop(TARGET)
    eda = {
        "rows": len(df),
        "class_counts": {"Legitimate": int((y == 0).sum()), "Fraud": int((y == 1).sum())},
        "fraud_rate": float(y.mean()),
        "missing_values": int(X.isna().sum().sum()),
        "iqr_outliers": iqr_out.sort_values(ascending=False).to_dict(),
        "isolation_forest": {"flagged": int(flag.sum()),
                             "precision": float(precision_score(y, flag, zero_division=0)),
                             "recall": float(recall_score(y, flag, zero_division=0))},
        "corr_with_class": corr.reindex(corr.abs().sort_values(ascending=False).index).round(4).to_dict(),
    }
    dump("eda.json", eda)
    df.sample(min(8000, len(df)), random_state=seed).round(4).to_csv(ART / "eda_sample.csv", index=False)
    return eda


# ------------------------------------------------------------ Unit IV helpers
def classifiers(seed):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5,
                                                class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=2,
                                                class_weight="balanced_subsample", n_jobs=-1,
                                                random_state=seed),
        "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced", random_state=seed),
        "Naive Bayes": GaussianNB(),
    }


def clf_pipe(model):
    return Pipeline([("prep", make_prep()),
                     ("select", SelectKBest(f_classif, k=FS_K)),
                     ("clf", model)])


def get_scores(pipe, X):
    return pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X)


def clf_metrics(y, pred, score):
    fpr, tpr, _ = roc_curve(y, score)
    idx = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)
    return {"accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, score),
            "pr_auc": average_precision_score(y, score),
            "confusion": confusion_matrix(y, pred).tolist(),
            "roc": {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}}


def imbalance_study(Xtr, ytr, Xte, yte, seed):
    """Same Logistic Regression, four ways of dealing with the rare fraud class."""
    def lr(cw=None):
        return LogisticRegression(max_iter=2000, class_weight=cw)

    def run(pipe, X, y):
        pipe.fit(X, y)
        s, p = pipe.predict_proba(Xte)[:, 1], pipe.predict(Xte)
        return {"precision": precision_score(yte, p, zero_division=0),
                "recall": recall_score(yte, p), "f1": f1_score(yte, p),
                "pr_auc": average_precision_score(yte, s)}

    res = {"No handling": run(Pipeline([("prep", make_prep()), ("clf", lr())]), Xtr, ytr),
           "Class weights": run(Pipeline([("prep", make_prep()), ("clf", lr("balanced"))]), Xtr, ytr)}
    fraud_idx = ytr[ytr == 1].index
    legit_idx = ytr[ytr == 0].sample(n=min(3 * len(fraud_idx), int((ytr == 0).sum())),
                                     random_state=seed).index
    keep = fraud_idx.union(legit_idx)
    res["Undersampling (1:3)"] = run(Pipeline([("prep", make_prep()), ("clf", lr())]),
                                     Xtr.loc[keep], ytr.loc[keep])
    if SMOTE is not None:
        res["SMOTE"] = run(ImbPipeline([("prep", make_prep()), ("smote", SMOTE(random_state=seed)),
                                        ("clf", lr())]), Xtr, ytr)
    return res


def model_importance(pipe):
    names = np.array([n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()])
    sel = pipe.named_steps["select"]
    est = pipe.named_steps["clf"]
    if hasattr(est, "feature_importances_"):
        imp, names = est.feature_importances_, names[sel.get_support()]
    elif hasattr(est, "coef_"):
        imp, names = np.abs(est.coef_.ravel()), names[sel.get_support()]
    else:  # e.g. Naive Bayes: fall back to the ANOVA F-score from feature selection
        imp = np.nan_to_num(sel.scores_)
    order = np.argsort(-imp)
    return {str(names[i]): float(imp[i]) for i in order}


# ------------------------------------------------------------------ Unit III
def regression_study(train_df, test_df, seed):
    out = {}
    for target in ["Amount", "Class"]:
        feats = V_COLS if target == "Amount" else FEATURES  # don't predict Amount from itself
        Xa, Xb, ya, yb = train_df[feats], test_df[feats], train_df[target], test_df[target]
        best1 = Xa.corrwith(ya).abs().idxmax()
        sc = lambda m: Pipeline([("sc", StandardScaler()), ("m", m)])  # noqa: E731
        specs = [
            (f"Simple Linear [{best1}]", sc(LinearRegression()), [best1]),
            ("Multiple Linear", sc(LinearRegression()), feats),
            ("Polynomial (deg 2)", Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                                             ("sc", StandardScaler()), ("m", LinearRegression())]), feats),
            ("Ridge (CV alpha)", sc(RidgeCV(alphas=np.logspace(-2, 3, 20))), feats),
            ("Lasso (CV alpha)", sc(LassoCV(cv=3, n_alphas=30, max_iter=5000, random_state=seed)), feats),
            ("Elastic Net (CV)", sc(ElasticNetCV(l1_ratio=[.2, .5, .8], cv=3, n_alphas=30,
                                                 max_iter=5000, random_state=seed)), feats),
        ]
        res = {}
        for name, pipe, cols in specs:
            pipe.fit(Xa[cols], ya)
            p = pipe.predict(Xb[cols])
            mse = mean_squared_error(yb, p)
            d = {"MAE": mean_absolute_error(yb, p), "MSE": mse, "RMSE": float(np.sqrt(mse)),
                 "R2": r2_score(yb, p), "R2_train": r2_score(ya, pipe.predict(Xa[cols]))}
            m = pipe.named_steps["m"]
            if hasattr(m, "alpha_"):
                d["alpha"] = float(m.alpha_)
            if hasattr(m, "coef_"):
                d["nonzero_coefs"] = int((np.abs(m.coef_) > 1e-8).sum())
            res[name] = d
        out[target] = res
    return out


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard_2023.csv")
    ap.add_argument("--n_legit", type=int, default=30000, help="legit rows to sample")
    ap.add_argument("--fraud_rate", type=float, default=0.02, help="target fraud share after re-balancing")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    ART.mkdir(exist_ok=True)

    t0 = time.time()
    raw = load_data(a.data)
    print(f"Loaded {len(raw):,} rows; raw fraud share = {raw[TARGET].mean():.3f}")
    df = build_imbalanced(raw, a.n_legit, a.fraud_rate, a.seed)
    print(f"Study set: {len(df):,} rows, {int(df[TARGET].sum())} fraud ({df[TARGET].mean():.2%})")

    eda = run_eda(df, a.seed)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=a.seed)
    Xtr, ytr, Xte, yte = train_df[FEATURES], train_df[TARGET], test_df[FEATURES], test_df[TARGET]

    print("Imbalance study ...")
    imbalance = imbalance_study(Xtr, ytr, Xte, yte, a.seed)

    print("Classification ...")
    skf = StratifiedKFold(3, shuffle=True, random_state=a.seed)
    results, fitted = {}, {}
    for name, model in classifiers(a.seed).items():
        s0, pipe = time.time(), clf_pipe(model)
        cv = cross_val_score(pipe, Xtr, ytr, cv=skf, scoring="average_precision").mean()
        pipe.fit(Xtr, ytr)
        m = clf_metrics(yte, pipe.predict(Xte), get_scores(pipe, Xte))
        m.update(cv_pr_auc=float(cv), fit_seconds=round(time.time() - s0, 1))
        results[name], fitted[name] = m, pipe
        print(f"  {name:20s} PR-AUC={m['pr_auc']:.3f} F1={m['f1']:.3f} recall={m['recall']:.3f}")

    print("Tuning Random Forest ...")
    grid = GridSearchCV(
        clf_pipe(RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample",
                                        n_jobs=-1, random_state=a.seed)),
        {"clf__max_depth": [8, 12], "clf__min_samples_leaf": [1, 3], "select__k": [FS_K, "all"]},
        scoring="average_precision", cv=skf).fit(Xtr, ytr)
    tuned = grid.best_estimator_
    m = clf_metrics(yte, tuned.predict(Xte), get_scores(tuned, Xte))
    m.update(cv_pr_auc=float(grid.best_score_), fit_seconds=0.0)
    results["Random Forest (tuned)"], fitted["Random Forest (tuned)"] = m, tuned
    tuning = {"best_params": {k: str(v) for k, v in grid.best_params_.items()},
              "best_cv_pr_auc": float(grid.best_score_)}

    # deploy the model with the best CROSS-VALIDATED PR-AUC (never pick on the test set)
    deployable = {n: r["cv_pr_auc"] for n, r in results.items() if hasattr(fitted[n], "predict_proba")}
    best = max(deployable, key=deployable.get)
    print(f"Best deployable model: {best}")
    joblib.dump(fitted[best], ART / "best_model.joblib", compress=3)
    pd.DataFrame({"y_true": yte.values, "score": fitted[best].predict_proba(Xte)[:, 1]}) \
        .to_csv(ART / "test_scores.csv", index=False)

    print("Regression ...")
    regression = regression_study(train_df, test_df, a.seed)

    dump("feature_importance.json", model_importance(fitted[best]))
    dump("feature_stats.json", {"median": Xtr.median().to_dict(),
                                "q01": Xtr.quantile(.01).to_dict(), "q99": Xtr.quantile(.99).to_dict()})
    dump("metrics.json", {
        "config": {"raw_rows": len(raw), "study_rows": len(df), "fraud_rate": eda["fraud_rate"],
                   "train_rows": len(train_df), "test_rows": len(test_df), "seed": a.seed,
                   "feature_selection_k": FS_K, "smote_available": SMOTE is not None},
        "classification": results, "tuning": tuning, "imbalance": imbalance,
        "regression": regression, "best_model": best})
    print(f"Done in {time.time() - t0:.0f}s -> artifacts/ (commit this folder to deploy)")


if __name__ == "__main__":
    main()
