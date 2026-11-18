#  Credit Card Fraud Detection Using Machine Learning

##  Project Overview

Banking Fraud Detection is a Machine Learning project developed to identify potentially fraudulent banking transactions.

The project uses transaction data and multiple Machine Learning algorithms to classify transactions as legitimate or fraudulent. It also provides an interactive web application built with Streamlit for data analysis, model comparison, and fraud prediction.

The main objective is to understand how Machine Learning can help identify suspicious transactions, especially when fraudulent transactions represent a very small portion of the dataset.

##  Objectives

- Analyze banking transaction data using Exploratory Data Analysis(EDA) .
- Perform data preprocessing and feature engineering.
- Implement multiple Machine Learning classification algorithms.
- Compare model performance using different evaluation metrics.
- Perform hyperparameter tuning to select suitable model configurations.
- Develop an interactive application for fraud prediction.
- Support individual transaction prediction and batch CSV prediction.

##  Dataset Information

The project uses the **Indian Banking Transactions 2019–2024** dataset.

| Dataset Details | Value |
|---|---:|
| Total Transactions | 550,000 |
| Total Features | 20 |
| Legitimate Transactions | 545,127 |
| Fraudulent Transactions | 4,873 |
| Fraud Percentage | Approximately 0.89% |
| Target Column | `is_fraud` |

**Dataset Source:** [Download the dataset from Kaggle](https://www.kaggle.com/datasets/belbino/indian-banking-transactions-20192024)

The target variable `is_fraud` indicates whether a transaction is fraudulent:

- `0` – Legitimate Transaction
- `1` – Fraudulent Transaction

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning algorithms |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive web application |
| Joblib | Saving and loading trained models |
| VS Code | Development environment |

## 🤖 Machine Learning Algorithms

Six Machine Learning algorithms were implemented and compared:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)
6. Gaussian Naive Bayes

### Hyperparameter Tuning

Different parameter configurations were tested across the six models, resulting in 40 experiments.

The selected configurations were:

| Model | Selected Configuration |
|---|---|
| Logistic Regression | C = 0.01 |
| KNN | K = 31, Distance Weighting |
| Decision Tree | Max Depth = 12, Min Samples Leaf = 10 |
| Random Forest | 100 Trees, Max Depth = 10 |
| SVM | RBF Kernel, C = 0.1 |
| Naive Bayes | Variance Smoothing = 1e-8 |

Random Forest achieved the highest validation PR-AUC of 0.0232 and was selected as the final model.



## 📁 Project Structure

The project is organized as follows:

```text
credit_card_fraud_project/
│
├── data/
│   └── indian_banking_transactions.csv
│
├── artifacts/
│   ├── best_model.joblib
│   └── results.json
│
├── reports/
│   ├── hyperparameter_experiments.csv
│   ├── model_comparison.csv
│   ├── test_predictions.csv
│   ├── feature_importance.csv
│   ├── model_comparison.png
│   ├── pr_curves.png
│   ├── roc_curves.png
│   └── other_visualizations.png
│
├── config.py
├── data_prep.py
├── models.py
├── train.py
├── app.py
├── requirements.txt
└── README.md


## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project Folder

```bash
cd credit_card_fraud_project
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Download the Dataset

Download the dataset from Kaggle and place the CSV file inside the `data` folder.

Expected dataset path:

```text
data/indian_banking_transactions.csv
```

### Step 6: Run the Application

If the trained model and required artifacts are already available, start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

### Training the Models

To train the models again and regenerate the model artifacts and reports, run:

```bash
python train.py
```

**Note:** Running `train.py` will train the six models again and overwrite the corresponding saved artifacts and reports.

## 📏 Evaluation Metrics

The following evaluation metrics are used:

- **Accuracy:** Percentage of correctly classified transactions.
- **Precision:** Percentage of predicted fraudulent transactions that are actually fraudulent.
- **Recall:** Percentage of actual fraudulent transactions correctly detected.
- **F1-score:** Harmonic mean of Precision and Recall.
- **ROC-AUC:** Measures the model's ability to distinguish between the two classes across classification thresholds.
- **PR-AUC:** Measures the trade-off between Precision and Recall and is particularly useful for imbalanced datasets.

## ⚠️ Limitations

- The dataset contains a highly imbalanced distribution of legitimate and fraudulent transactions.
- The selected Random Forest model has relatively low fraud detection recall.
- The final test evaluation uses a 50,000-record sample rather than the complete test set.
- Some models use balanced training samples while others use the full training data.
- The application supports manual and CSV-based predictions but is not connected to live banking systems.
- Model performance on new real-world transactions may differ from the recorded test results.

## 🚀 Future Improvements

- Improve fraud detection recall and precision.
- Experiment with additional class-imbalance handling techniques.
- Perform systematic classification threshold optimization.
- Evaluate models on the complete test dataset.
- Explore advanced models such as XGBoost and LightGBM.
- Integrate real-time transaction data through APIs.
- Develop an alert mechanism for suspicious transactions.

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

- Data preprocessing and exploratory data analysis.
- Handling imbalanced datasets.
- Implementing and comparing Machine Learning algorithms.
- Hyperparameter tuning and model evaluation.
- Building interactive applications using Streamlit.
- Integrating trained Machine Learning models into a web application.

## 👨‍💻 Author

**Mrigendra Singh kushwaha **

GitHub: [alpha0301singh](https://github.com/alpha0301singh)


⭐ If you find this project useful, feel free to explore the repository and share your feedback.
