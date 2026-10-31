"""All charts, in one place. Every function RETURNS a matplotlib Figure, so train.py/eda.py
can save it as a PNG and app.py can show it with st.pyplot. Only matplotlib is needed."""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

BLUE, RED = "#4c78a8", "#d62728"


def _fig(w=6.0, h=4.0):
    fig = Figure(figsize=(w, h), layout="constrained")
    return fig, fig.subplots()


# ----------------------------------------------------------------------- EDA
def class_distribution(counts):
    fig, ax = _fig(4.5, 3.5)
    ax.bar(list(counts), list(counts.values()), color=[BLUE, RED])
    for i, v in enumerate(counts.values()):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set(ylabel="Transactions", title="Class distribution")
    return fig


def amount_by_class(df):
    fig = Figure(figsize=(9, 3.5), layout="constrained")
    a1, a2 = fig.subplots(1, 2)
    for ax, transform, title in ((a1, lambda x: x, "Amount (raw)"),
                                 (a2, np.log1p, "log(1 + Amount)  - after transformation")):
        for cls, name, col in ((0, "Legitimate", BLUE), (1, "Fraud", RED)):
            ax.hist(transform(df.loc[df["Class"] == cls, "Amount"].dropna()), bins=40, density=True,
                    alpha=.55, label=name, color=col)
        ax.set(title=title, ylabel="Density")
        ax.legend()
    return fig


def target_correlation(corr, top=15):
    s = corr.reindex(corr.abs().sort_values(ascending=False).index).head(top).iloc[::-1]
    fig, ax = _fig(6, 4.5)
    ax.barh(s.index, s.values, color=[RED if v > 0 else BLUE for v in s.values])
    ax.axvline(0, color="k", lw=.8)
    ax.set(xlabel="Correlation with fraud label", title=f"Top {top} features by |correlation|")
    return fig


def heatmap(df, cols):
    c = df[cols].corr()
    fig, ax = _fig(6.5, 5.5)
    im = ax.imshow(c.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90)
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.values[i, j]:.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation heatmap")
    return fig


def iqr_outliers(counts, top=12):
    s = pd.Series(counts).head(top).iloc[::-1]
    fig, ax = _fig(6, 4)
    ax.barh(s.index, s.values, color=BLUE)
    ax.set(xlabel="Rows outside the IQR fences", title="Outliers per feature (IQR rule)")
    return fig


# --------------------------------------------------------------------- models
def imbalance_comparison(df):
    fig, ax = _fig(7, 4)
    df[["precision", "recall", "f1", "pr_auc"]].plot.bar(ax=ax, rot=15)
    ax.set(ylabel="Score", title="Imbalance handling (same Logistic Regression)")
    return fig


def model_comparison(table):
    fig, ax = _fig(8, 4.2)
    table[["precision", "recall", "f1", "pr_auc"]].plot.bar(ax=ax, rot=25)
    ax.set(ylabel="Score", title="Classifier comparison (held-out test set)")
    return fig


def roc_curves(results):
    fig, ax = _fig(6, 4.5)
    for n, r in results.items():
        ax.plot(r["roc"]["fpr"], r["roc"]["tpr"], label=f"{n} ({r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend(fontsize=7)
    return fig


def pr_curves(results):
    fig, ax = _fig(5.5, 4.2)
    for n, r in results.items():
        ax.plot(r["pr"]["recall"], r["pr"]["precision"], label=n)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves")
    ax.legend(fontsize=6)
    return fig


def confusion(cm, title="Confusion matrix"):
    cm = np.array(cm)
    fig, ax = _fig(4, 3.4)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Legit", "Fraud"])
    ax.set_yticks([0, 1], ["Legit", "Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    return fig


def regression_comparison(df, target):
    fig, ax = _fig(7, 4)
    df[["R2_train", "R2"]].rename(columns={"R2": "R2 (test)", "R2_train": "R2 (train)"}).plot.bar(
        ax=ax, rot=20, color=[BLUE, RED])
    ax.axhline(0, color="k", lw=.8)
    ax.set(title=f"R-squared per model, target = {target}", ylabel="R2")
    return fig


# ----------------------------------------------------------- explain / decide
def permutation_importance(perm, top=15):
    d = pd.DataFrame(perm).T.head(top).iloc[::-1]
    fig, ax = _fig(6, 4.5)
    ax.barh(d.index, d["mean"], xerr=d["std"], color=BLUE)
    ax.set(xlabel="Drop in test PR-AUC when the feature is shuffled", title="Permutation importance")
    return fig


def local_explanation(contrib, top=10):
    s = contrib.head(top).iloc[::-1]
    fig, ax = _fig(6, 4)
    ax.barh(s.index, s.values, color=[RED if v > 0 else BLUE for v in s.values])
    ax.axvline(0, color="k", lw=.8)
    ax.set(xlabel="Change in fraud probability caused by this feature",
           title="Why this score? (red = towards fraud, blue = away)")
    return fig


def cost_curve(grid, cost, selected, optimal):
    fig, ax = _fig(6, 3)
    ax.plot(grid, cost)
    ax.axvline(selected, color=RED, ls="--", label="selected")
    ax.axvline(optimal, color="green", ls=":", label="cost-optimal")
    ax.set(xlabel="Threshold", ylabel="Total cost")
    ax.legend()
    return fig
