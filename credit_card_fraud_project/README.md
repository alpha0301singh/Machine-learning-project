# Credit Card Fraud Detection: M.Tech ML Project

End-to-end machine learning study plus a Streamlit app, on the **Credit Card Fraud Detection
Dataset 2023** (Kaggle, ~550,000 anonymised transactions from European cardholders; columns
`id, V1-V28, Amount, Class`).

> Dataset: search Kaggle for *"Credit Card Fraud Detection Dataset 2023"* (author: Nidula Elgiriyewithana),
> download `creditcard_2023.csv` and place it in `data/`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py --data creditcard_2023.csv             # or data/creditcard_2023.csv
streamlit run app.py
```

If the real CSV lives in `data/`, use that path; if it is in the project root, the app will auto-detect it.
No dataset yet? `python make_demo_data.py` creates a synthetic file with the same schema for a smoke test.
**Use the real file for your reported results.**

Useful options: `--n_legit 30000` (rows sampled), `--fraud_rate 0.02` (share of fraud after re-balancing),
`--seed 42`. Lower `--n_legit` if the SVM is slow on your machine.

## Why we re-balance the data

The 2023 file is (as far as I know) close to 50/50 fraud vs legitimate, which is unrealistic; real card
fraud is well under 1%. `train.py` therefore keeps a sample of legitimate rows and only enough fraud rows to
reach `--fraud_rate` (default 2%). This restores the imbalance problem your syllabus asks you to handle, and
makes accuracy a misleading metric, so models are compared on **recall, precision, F1 and PR-AUC**.
State this design decision explicitly in your report.

## How the project maps to your syllabus

| Unit | Topic | Where |
|---|---|---|
| I | Pandas loading, duplicates, missing values, scaling, train-test split | `common.load_data`, `common.make_prep`, stratified split in `train.main` |
| II | EDA, visualisation, correlation | `train.run_eda`, app **EDA** page |
| II | Outlier detection | IQR rule + Isolation Forest in `run_eda` |
| II | Feature selection / transformation | `SelectKBest(f_classif)`, `log(1+Amount)`, scaling |
| II | Imbalance handling | `imbalance_study`: none / class weights / under-sampling / SMOTE |
| II | Preprocessing pipelines | `Pipeline` + `ColumnTransformer` (no leakage; CV runs the whole pipeline) |
| III | Simple, multiple, polynomial regression | `regression_study` |
| III | Ridge, Lasso, Elastic Net | `RidgeCV`, `LassoCV`, `ElasticNetCV` (alpha chosen by CV) |
| III | MAE, MSE, RMSE, R² | Regression page (train *and* test R² to show overfitting) |
| IV | LR, KNN, Decision Tree, Random Forest, SVM, Naive Bayes | `classifiers()` |
| IV | Confusion matrix, accuracy, precision, recall, F1, ROC-AUC | Classification page (+ PR-AUC) |

Two regression targets are used: **Amount** (the anonymised features probably carry little signal, so expect
R² near 0; report it honestly and discuss why) and the **0/1 fraud label as a risk score** (a linear
probability model, useful for Lasso feature selection).

## Deploy on Streamlit Community Cloud

1. Run `train.py` locally (needs the real CSV). It creates `artifacts/` (model, metrics, small EDA sample).
2. Push the project to GitHub **including `artifacts/`** but not `data/` (already in `.gitignore`).
   The raw CSV is not needed at runtime.
3. On share.streamlit.io choose the repo, main file `app.py`, and deploy.
4. Version match: the model is pickled with your scikit-learn version. Run `pip freeze | grep -i -E
   "scikit-learn|numpy|pandas|joblib"` after training and pin those versions in `requirements.txt`.
5. If `best_model.joblib` exceeds GitHub's 100 MB limit, lower `n_estimators` / `max_depth` in `train.py`.

## App pages

Overview, EDA, Classification (metrics table, ROC curves, confusion matrices, imbalance comparison, feature
importance), Regression, and **Live prediction** (manual input, sample transaction, batch CSV upload, adjustable
decision threshold, and a cost-sensitive threshold analysis where you set the cost of a missed fraud vs a
false alarm).

## Suggested report outline

1. Introduction & problem statement  2. Dataset description & re-balancing rationale  3. Data preparation
4. EDA & feature engineering  5. Regression study  6. Classification study & imbalance handling
7. Model selection (CV PR-AUC) and threshold/cost analysis  8. Deployment  9. Limitations & future work

## Limitations to mention (examiners like honesty)

- Features V1-V28 are anonymised PCA components, so they cannot be interpreted in business terms.
- The Kaggle 2023 data may be partly synthetic or cleaned; very high scores can be too optimistic for real
  card traffic. Mention that results are benchmark results, not production guarantees.
- No time dimension, so no concept-drift or temporal validation. Future work: time-aware split, XGBoost/LightGBM,
  SHAP explanations, autoencoder anomaly detection.
- The app is an academic demo, not a production fraud system.
