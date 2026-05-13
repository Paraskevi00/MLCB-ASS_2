"""Plotting helpers for EDA, rnCV results, feature selection, and SHAP."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})


def savefig(fig, name: str):
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    return path


def plot_class_distribution(y: pd.Series, save=True):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = y.value_counts().sort_index()
    ax.bar(["No disease (0)", "Disease (1)"], counts.values)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, f"{v}\n{100*v/len(y):.1f}%", ha="center")
    ax.set_title("Class distribution")
    ax.set_ylabel("Patients")
    fig.tight_layout()
    if save: savefig(fig, "class_distribution.png")
    return fig


def plot_missing_values(X: pd.DataFrame, save=True):
    miss = X.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        print("No missing values found.")
        return None
    fig, ax = plt.subplots(figsize=(6, max(3, len(miss) * 0.35)))
    ax.barh(miss.index[::-1], miss.values[::-1])
    ax.set_title("Missing values per feature")
    ax.set_xlabel("Missing count")
    fig.tight_layout()
    if save: savefig(fig, "missing_values.png")
    return fig


def plot_feature_boxplots(X: pd.DataFrame, y: pd.Series, cols: List[str], save=True):
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 3.6*nrows))
    axes = np.ravel(axes)
    df = X.copy(); df["num"] = y.values
    for ax, c in zip(axes, cols):
        sns.boxplot(data=df, x="num", y=c, ax=ax)
        ax.set_title(c)
        ax.set_xlabel("Heart disease")
    for ax in axes[len(cols):]: ax.set_visible(False)
    fig.tight_layout()
    if save: savefig(fig, "continuous_boxplots.png")
    return fig


def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series | None = None, save=True):
    df = X.copy()
    if y is not None: df["num"] = y.values
    corr = df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title("Spearman correlation heatmap")
    fig.tight_layout()
    if save: savefig(fig, "correlation_heatmap.png")
    return fig


def plot_pca_from_pipeline(preprocessor, X: pd.DataFrame, y: pd.Series, save=True):
    from sklearn.decomposition import PCA
    Xt = preprocessor.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xt)
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, label in [(0, "No disease"), (1, "Disease")]:
        mask = y.values == cls
        ax.scatter(Z[mask, 0], Z[mask, 1], label=label, alpha=.75)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA view after preprocessing")
    ax.legend()
    fig.tight_layout()
    if save: savefig(fig, "pca.png")
    return fig

def plot_categorical_distributions(df, target_col, cols, save=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.8 * nrows)
    )

    axes = np.ravel(axes)

    for ax, col in zip(axes, cols):
        sns.countplot(
            data=df,
            x=col,
            hue=target_col,
            ax=ax
        )

        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend(title="Heart disease")

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.tight_layout()

    if save:
        savefig(fig, "categorical_feature_distributions.png")

    return fig

def plot_metric_violin(results: Dict[str, pd.DataFrame], metric="MCC", save=True):
    rows = []
    for est, df in results.items():
        for v in df[metric]: rows.append({"Model": est, metric: v})
    plot_df = pd.DataFrame(rows)
    order = plot_df.groupby("Model")[metric].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(max(8, len(order)*1.1), 5))
    sns.violinplot(data=plot_df, x="Model", y=metric, order=order, cut=0, ax=ax)
    ax.set_title(f"Outer-fold {metric} distribution")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    if save: savefig(fig, f"violin_{metric.lower()}.png")
    return fig


def plot_summary_heatmap(summary: pd.DataFrame, save=True):
    med = summary[[c for c in summary.columns if c.endswith("_median")]].copy()
    med.columns = [c.replace("_median", "") for c in med.columns]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.55*len(med))))
    sns.heatmap(med, annot=True, fmt=".3f", cmap="YlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title("Median rnCV performance")
    fig.tight_layout()
    if save: savefig(fig, "rncv_summary_heatmap.png")
    return fig


def plot_feature_selection_frequency(fs_report: pd.DataFrame, estimator: str, top_n=20, save=True):
    df = fs_report[fs_report.estimator == estimator].head(top_n).sort_values("selection_frequency_%")
    fig, ax = plt.subplots(figsize=(7, max(4, .35*len(df))))
    ax.barh(df["feature"], df["selection_frequency_%"])
    ax.set_xlabel("Selection frequency (%)")
    ax.set_title(f"Stable selected features — {estimator}")
    fig.tight_layout()
    if save: savefig(fig, f"feature_selection_{estimator}.png")
    return fig


def plot_shap_summary(shap_values, X_model: pd.DataFrame, save=True):
    import shap
    shap.summary_plot(shap_values, X_model, show=False)
    fig = plt.gcf()
    if save: savefig(fig, "shap_summary.png")
    return fig
