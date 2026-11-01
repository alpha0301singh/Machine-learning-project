# Credit Card Fraud Detection (M.Tech ML project)

Machine-learning study + Streamlit web app on the **Credit Card Fraud Detection Dataset 2023**
(Kaggle, ~550,000 anonymised European cardholder transactions, columns `id, V1-V28, Amount, Class`).

Dataset: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023
(download `creditcard_2023.csv` and put it in the `data/` folder).

## Files (one job per file)

| File | Job | Syllabus unit |
|---|---|---|
| `config.py` | all settings (paths, seed, fraud rate...) | |
| `data_prep.py` | load, clean, re-balance to ~2% fraud, stratified train/test split | I |
| `eda.py` | statistics, correlation, outliers (IQR + Isolation Forest), charts | II |
| `models.py` | preprocessing pipeline, 6 classifiers, 6 regression models, metric functions | III, IV |
| `explain.py` | permutation importance, per-transaction explanation, cost/threshold analysis | extra |
| `plots.py` | every chart (used by `eda.py`, `train.py` and the app) | |
| `train.py` | runs everything, cross-validation + tuning, saves `artifacts/` and `reports/` | I-IV |
| `app.py` | Streamlit web app | |
| `make_demo_data.py` | fake data with the same columns, for testing only | |

## 1. Install (once)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows      (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## 2. Run

**Important:** run every command from inside this project folder.

```bash
python train.py --data data/creditcard_2023.csv     # real dataset, about 2-5 minutes
streamlit run app.py                                 # opens http://localhost:8501
```

Want to test without downloading anything?  `python train.py --demo` (synthetic data; the app shows a
warning banner so you never mistake it for real results).

**This zip already contains a quick demo run in `artifacts/`, so `streamlit run app.py` works immediately.
Those numbers are from synthetic data: run `train.py` on the real dataset before you use any result.**

You can also run the steps one by one: `python data_prep.py --data ...`, then `python eda.py`, then `python train.py`.
All charts for your report are saved in `reports/`.

## 3. Deploy on Streamlit Community Cloud (free)

1. Run `train.py` on the real data first (this refreshes `artifacts/`).
2. Create a GitHub repository and upload this folder **including `artifacts/`**
   (`data/*.csv` is ignored on purpose; the app does not need the big file).
3. Go to https://share.streamlit.io, sign in with GitHub, click **New app**, choose the repository,
   branch `main`, main file `app.py`, and press **Deploy**.
4. If the saved model cannot be loaded on the server (different scikit-learn version), the app
   retrains the same model automatically from `artifacts/train.csv`; nothing breaks.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Dataset not found` | put `creditcard_2023.csv` in `data/`, or use `--data path/to/file.csv`, or `--demo` |
| `ModuleNotFoundError` | activate the virtual environment and run `pip install -r requirements.txt` |
| `streamlit` is not recognised | use `python -m streamlit run app.py` |
| `No results found` in the app | run `python train.py ...` first (or use the `artifacts/` shipped in the zip) |
| training is slow | lower `N_LEGIT` in `config.py` (for example 10000) |
| app says "SYNTHETIC demo data" | you are viewing the demo run: train on the real CSV |

## Why the data is re-balanced

The 2023 file is (as far as I know) close to 50% fraud, which is unrealistic. `data_prep.py` keeps
20,000 legitimate rows and only enough fraud rows for a 2% fraud rate. That brings back the real difficulty
(rare fraud), so accuracy is misleading and models are compared on recall, precision, F1 and PR-AUC.
State this choice in your report.

## Limitations (mention them in your viva)

- V1-V28 are anonymised, so features cannot be interpreted in business terms.
- The 2023 dataset may be partly synthetic; very high scores may not carry over to real card traffic.
- No time dimension, so no drift or time-based validation. This is an academic demo, not a production system.
