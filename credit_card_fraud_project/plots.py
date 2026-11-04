
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

from config import (
    COMPARISON_FILE,
    PREDICTIONS_FILE,
    REPORTS
)


def plot_model_comparison():

    df = pd.read_csv(
        COMPARISON_FILE,
        index_col="Model"
    )

    plt.figure(figsize=(10, 6))

    df["pr_auc"].sort_values().plot(
        kind="barh"
    )

    plt.title("Model Comparison Using PR-AUC")

    plt.xlabel("PR-AUC")

    plt.ylabel("Classification Model")

    plt.tight_layout()

    plt.savefig(
        REPORTS / "model_comparison.png"
    )

    plt.close()


def plot_confusion_matrix():

    df = pd.read_csv(PREDICTIONS_FILE)

    ConfusionMatrixDisplay.from_predictions(
        df["actual"],
        df["predicted"],
        display_labels=[
            "Legitimate",
            "Fraud"
        ]
    )

    plt.title("Fraud Detection Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        REPORTS / "confusion_matrix.png"
    )

    plt.close()


if __name__ == "__main__":

    plot_model_comparison()

    plot_confusion_matrix()

    print("Model charts saved successfully.")
