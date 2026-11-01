"""STEP 4 - Run the whole study and save everything the Streamlit app needs.

    python train.py --data data/creditcard_2023.csv      # real dataset
    python train.py --demo                                # synthetic data, quick test

What happens: data_prep (clean, re-balance, split) -> eda -> imbalance study -> regression
-> classification (cross-validation + tuning) -> explainability -> save to artifacts/.
"""
import argparse
import json
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import eda
import explain
import models
import plots
from config import (ARTIFACTS, CV_FOLDS, DATA_FILE, FEATURES, FRAUD_RATE, N_LEGIT, REPORTS, SEED,
                    TARGET, V_COLS)
from data_prep import prepare

warnings.filterwarnings("ignore")


def save_fig(fig, name):
    REPORTS.mkdir(exist_ok=True)
    fig.savefig(REPORTS / name, dpi=150, bbox_inches="tight")


# ------------------------------------------------------------------ Unit II: imbalance
def imbalance_study(Xtr, ytr, Xte, yte):
    """Same Logistic Regression, three ways of treating the rare fraud class."""
    def run(class_weight, X, y):
        pipe = Pipeline([("prep", models.build_preprocessor()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight=class_weight))]).fit(X, y)
        s, p = pipe.predict_proba(Xte)[:, 1], pipe.predict(Xte)
        return {"precision": precision_score(yte, p, zero_division=0), "recall": recall_score(yte, p),
                "f1": f1_score(yte, p), "pr_auc": average_precision_score(yte, s)}

    fraud_idx = ytr[ytr == 1].index                                  # under-sampling: all fraud,
    legit_idx = ytr[ytr == 0].sample(n=min(3 * len(fraud_idx), int((ytr == 0).sum())), random_state=SEED).index
    keep = fraud_idx.union(legit_idx)                                # only 3 legit rows per fraud row
    return {"No handling": run(None, Xtr, ytr),
            "Class weights": run("balanced", Xtr, ytr),
            "Under-sampling (1:3)": run(None, Xtr.loc[keep], ytr.loc[keep])}


# -------------------------------------------------------------------- Unit III: regression
def regression_study(train, test):
    out = {}
    for target in ("Amount", "Class"):
        feats = V_COLS if target == "Amount" else FEATURES           # never predict Amount from itself
        Xa, Xb = train[feats], test[feats]
        Xa, Xb = Xa.fillna(Xa.median()), Xb.fillna(Xa.median())
        ya, yb = train[target], test[target]
        best_single = Xa.corrwith(ya).abs().idxmax()
        res = {}
        for name, pipe, cols in models.get_regressors(feats, best_single):
            pipe.fit(Xa[cols], ya)
            m = models.regression_metrics(yb, pipe.predict(Xb[cols]), ya, pipe.predict(Xa[cols]))
            est = pipe.named_steps["m"]
            if hasattr(est, "alpha_"):
                m["alpha"] = float(est.alpha_)
            if hasattr(est, "coef_"):
                m["nonzero_coefs"] = int((np.abs(est.coef_) > 1e-8).sum())
            res[name] = m
        out[target] = res
        save_fig(plots.regression_comparison(pd.DataFrame(res).T, target), f"regression_{target}.png")
    return out


# ----------------------------------------------------------------- Unit IV: classification
def classification_study(Xtr, ytr, Xte, yte):
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, model in models.get_classifiers().items():
        t0 = time.time()
        pipe = models.build_pipeline(model)
        cv_score = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="average_precision").mean()
        pipe.fit(Xtr, ytr)
        m = models.classification_metrics(yte, pipe.predict(Xte), models.get_scores(pipe, Xte))
        m["cv_pr_auc"] = float(cv_score)
        results[name], fitted[name] = m, pipe
        print(f"  {name:20s} cv PR-AUC={cv_score:.3f} | test PR-AUC={m['pr_auc']:.3f} "
              f"F1={m['f1']:.3f} recall={m['recall']:.3f}  ({time.time() - t0:.0f}s)")

    print("  tuning Random Forest ...")
    grid = GridSearchCV(models.tuned_rf_pipeline(), models.RF_GRID, scoring="average_precision", cv=cv).fit(Xtr, ytr)
    best_rf = grid.best_estimator_
    m = models.classification_metrics(yte, best_rf.predict(Xte), models.get_scores(best_rf, Xte))
    m["cv_pr_auc"] = float(grid.best_score_)
    results["Random Forest (tuned)"], fitted["Random Forest (tuned)"] = m, best_rf
    tuning = {"best_params": grid.best_params_, "best_cv_pr_auc": float(grid.best_score_)}

    # Pick the winner by CROSS-VALIDATED PR-AUC (never by test score). Only models that give
    # probabilities can be deployed, because the app needs a probability for its threshold.
    cands = {n: r["cv_pr_auc"] for n, r in results.items() if hasattr(fitted[n], "predict_proba")}
    best = max(cands, key=cands.get)
    return results, fitted, tuning, best


# ---------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_FILE), help="path to creditcard_2023.csv")
    ap.add_argument("--n_legit", type=int, default=N_LEGIT)
    ap.add_argument("--fraud_rate", type=float, default=FRAUD_RATE)
    ap.add_argument("--demo", action="store_true", help="use synthetic data (NOT for your report)")
    a = ap.parse_args()
    t0 = time.time()

    if a.demo:
        from make_demo_data import make
        a.data = str(make())
    source = "synthetic demo data" if a.demo else Path(a.data).name

    print("\n[1/6] Data preparation")
    train, test, info = prepare(a.data, a.n_legit, a.fraud_rate)
    Xtr, ytr, Xte, yte = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]

    print("\n[2/6] Exploratory data analysis")
    eda_res = eda.compute_eda(train)
    eda.print_summary(eda_res)
    eda.save_eda_charts(train, eda_res)

    print("\n[3/6] Imbalance study")
    imbalance = imbalance_study(Xtr, ytr, Xte, yte)
    print(pd.DataFrame(imbalance).T.round(3).to_string())
    save_fig(plots.imbalance_comparison(pd.DataFrame(imbalance).T), "imbalance.png")

    print("\n[4/6] Regression")
    regression = regression_study(train, test)
    for t, res in regression.items():
        print(f"  target {t}: best test R2 = {max(r['R2'] for r in res.values()):.3f}")

    print("\n[5/6] Classification")
    results, fitted, tuning, best = classification_study(Xtr, ytr, Xte, yte)
    print(f"  -> deployed model: {best}")
    table = pd.DataFrame(results).T.astype({c: float for c in ["precision", "recall", "f1", "pr_auc"]})
    save_fig(plots.model_comparison(table), "model_comparison.png")
    save_fig(plots.roc_curves(results), "roc_curves.png")
    save_fig(plots.pr_curves(results), "pr_curves.png")
    save_fig(plots.confusion(results[best]["confusion"], f"Confusion matrix: {best}"), "confusion_best.png")

    print("\n[6/6] Explainability + saving")
    model = fitted[best]
    perm = explain.permutation_importances(model, Xte, yte)
    save_fig(plots.permutation_importance(perm), "permutation_importance.png")
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "best_model.joblib", compress=3)
    pd.DataFrame({"y_true": yte.values, "score": model.predict_proba(Xte)[:, 1]}).round(5) \
        .to_csv(ARTIFACTS / "test_scores.csv", index=False)
    payload = {
        "meta": {"data_source": source, "trained_at": datetime.now().isoformat(timespec="seconds"),
                 "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
                 "seed": SEED, "cv_folds": CV_FOLDS, **{k: v for k, v in info.items() if k != "cleaning"}},
        "cleaning": info["cleaning"], "eda": eda_res, "imbalance": imbalance, "regression": regression,
        "classification": results, "tuning": tuning, "best_model": best, "permutation": perm,
        "feature_stats": {"median": Xtr.median().to_dict(), "q01": Xtr.quantile(.01).to_dict(),
                          "q99": Xtr.quantile(.99).to_dict()},
    }
    (ARTIFACTS / "results.json").write_text(json.dumps(
        payload, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"\nDone in {time.time() - t0:.0f}s. Start the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
