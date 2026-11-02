
"""All project charts are created in this file."""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


# Chart colors
BLUE = "#4c78a8"
RED = "#d62728"


# Create a chart
def create_chart(width=6, height=4):
    fig = Figure(figsize=(width, height), layout="constrained")
    ax = fig.subplots()
    return fig, ax


# ------------------------- EDA CHARTS -------------------------

def class_distribution(counts):
    fig, ax = create_chart()

    labels = list(counts.keys())
    values = list(counts.values())

    ax.bar(labels, values, color=[BLUE, RED])

    ax.set_title("Class Distribution")
    ax.set_ylabel("Number of Transactions")

    return fig


def amount_by_class(df):
    fig = Figure(figsize=(9, 4), layout="constrained")
    ax1, ax2 = fig.subplots(1, 2)

    for ax, transform, title in [
        (ax1, lambda x: x, "Original Amount"),
        (ax2, np.log1p, "Log Transformed Amount")
    ]:
        for label, name, color in [
            (0, "Legitimate", BLUE),
            (1, "Fraud", RED)
        ]:
            values = df.loc[df["Class"] == label, "Amount"].dropna()
            ax.hist(
                transform(values),
                bins=40,
                alpha=0.6,
                label=name,
                color=color
            )

        ax.set_title(title)
        ax.set_ylabel("Frequency")
        ax.legend()

    return fig


def target_correlation(corr, top=15):
    corr = corr.abs().sort_values(ascending=False).head(top)
    corr = corr.sort_values()

    fig, ax = create_chart()

    ax.barh(corr.index, corr.values, color=BLUE)
    ax.set_title("Feature Correlation with Fraud")
    ax.set_xlabel("Absolute Correlation")

    return fig


def heatmap(df, cols):
    correlation = df[cols].corr()

    fig, ax = create_chart(7, 5)

    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(cols)), cols, rotation=90)
    ax.set_yticks(range(len(cols)), cols)

    fig.colorbar(image, ax=ax)
    ax.set_title("Feature Correlation Heatmap")

    return fig


def iqr_outliers(counts, top=12):
    values = pd.Series(counts).head(top).sort_values()

    fig, ax = create_chart()

    ax.barh(values.index, values.values, color=BLUE)
    ax.set_title("IQR Outliers")
    ax.set_xlabel("Number of Outliers")

    return fig


# ------------------------- MODEL CHARTS -------------------------

def imbalance_comparison(df):
    fig, ax = create_chart(7, 4)

    df[["precision", "recall", "f1", "pr_auc"]].plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Imbalance Handling Comparison")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=15)

    return fig


def model_comparison(df):
    fig, ax = create_chart(8, 4)

    df[["precision", "recall", "f1", "pr_auc"]].plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Classification Model Comparison")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=25)

    return fig


def roc_curves(results):
    fig, ax = create_chart()

    for name, result in results.items():
        roc = result["roc"]

        ax.plot(
            roc["fpr"],
            roc["tpr"],
            label=f"{name} ({result['roc_auc']:.3f})"
        )

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7)

    return fig


def pr_curves(results):
    fig, ax = create_chart()

    for name, result in results.items():
        pr = result["pr"]

        ax.plot(
            pr["recall"],
            pr["precision"],
            label=name
        )

    ax.set_title("Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=6)

    return fig


def confusion(cm, title="Confusion Matrix"):
    fig, ax = create_chart(4, 3.5)

    cm = np.array(cm)

    ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    return fig


# ------------------------- REGRESSION CHART -------------------------

def regression_comparison(df, target):
    fig, ax = create_chart(7, 4)

    df[["R2_train", "R2"]].rename(
        columns={
            "R2_train": "Training R2",
            "R2": "Testing R2"
        }
    ).plot(kind="bar", ax=ax)

    ax.axhline(0, color="black")
    ax.set_title(f"Regression Results: {target}")
    ax.set_ylabel("R-squared")
    ax.tick_params(axis="x", rotation=20)

    return fig


# ------------------------- EXPLAINABILITY CHARTS -------------------------

def permutation_importance(perm, top=15):
    df = pd.DataFrame(perm).T.head(top).sort_values("mean")

    fig, ax = create_chart()

    ax.barh(
        df.index,
        df["mean"],
        xerr=df["std"],
        color=BLUE
    )

    ax.set_title("Permutation Feature Importance")
    ax.set_xlabel("Decrease in Average Precision")

    return fig


def local_explanation(contrib, top=10):
    values = contrib.head(top).sort_values()

    fig, ax = create_chart()

    colors = [
        RED if value > 0 else BLUE
        for value in values.values
    ]

    ax.barh(values.index, values.values, color=colors)
    ax.axvline(0, color="black")

    ax.set_title("Local Transaction Explanation")
    ax.set_xlabel("Change in Fraud Probability")

    return fig


# ------------------------- COST CURVE -------------------------

def cost_curve(grid, cost, selected, optimal):
    fig, ax = create_chart()

    ax.plot(grid, cost)

    ax.axvline(
        selected,
        color=RED,
        linestyle="--",
        label="Selected Threshold"
    )

    ax.axvline(
        optimal,
        color="green",
        linestyle=":",
        label="Best Cost Threshold"
    )

    ax.set_title("Fraud Detection Cost Curve")
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Total Cost")
    ax.legend()

    return fig
