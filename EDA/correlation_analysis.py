from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colors import Normalize

sns.set_theme(style="white", context="talk")


def compute_correlations(label_df: pd.DataFrame, method: str = "spearman") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute correlation coefficients and p-values for labels.

    Parameters
    ----------
    label_df:
        Dataframe containing label columns.
    method:
        Correlation method to apply; "spearman" or "pearson".

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Correlation matrix and p-value matrix aligned to label columns.
    """

    if method == "spearman":
        corr, p_values = stats.spearmanr(label_df)
        corr_df = pd.DataFrame(corr, index=label_df.columns, columns=label_df.columns)
        pval_df = pd.DataFrame(p_values, index=label_df.columns, columns=label_df.columns)
    elif method == "pearson":
        corr_df = label_df.corr(method="pearson")
        pval_df = pd.DataFrame(np.ones_like(corr_df), index=label_df.columns, columns=label_df.columns)
        for i, col_i in enumerate(label_df.columns):
            for j, col_j in enumerate(label_df.columns):
                coef, pval = stats.pearsonr(label_df[col_i], label_df[col_j])
                corr_df.loc[col_i, col_j] = coef
                pval_df.loc[col_i, col_j] = pval
    else:
        raise ValueError("method must be 'spearman' or 'pearson'")

    return corr_df, pval_df


def plot_label_relationships(label_df: pd.DataFrame, output_path: Path, method: str = "spearman") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Plot a 3x3 grid of label relationships and save correlation stats.

    Diagonal subplots show per-label histograms; off-diagonals show joint
    distributions with a heatmap. Correlation and p-value matrices are
    persisted alongside the visualization.
    """

    corr_df, pval_df = compute_correlations(label_df, method=method)

    labels = list(label_df.columns)
    n_labels = len(labels)
    fig, axes = plt.subplots(n_labels, n_labels, figsize=(4 * n_labels, 4 * n_labels))
    diag_palette = sns.color_palette("viridis", n_labels)
    zero_color = "#f0f0f0"
    norm = Normalize(vmin=0, vmax=1)
    shared_heatmap = None

    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            ax = axes[i, j]
            if i == j:
                counts = label_df[label_i].value_counts().sort_index()
                bar_colors = [zero_color if val == 0 else diag_palette[i] for val in counts.index]
                ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="white")
                ax.set_xlabel(label_i)
                ax.set_ylabel("Count")
                ax.set_xticks(counts.index)
            else:
                joint_counts = pd.crosstab(label_df[label_i], label_df[label_j]).astype(float)
                if 0 in joint_counts.index and 0 in joint_counts.columns:
                    joint_counts.loc[0, 0] = float("nan")
                max_count = np.nanmax(joint_counts.values) if not np.all(np.isnan(joint_counts.values)) else 1.0
                normalized_counts = joint_counts / max(max_count, 1e-9)
                cmap = sns.color_palette("viridis", as_cmap=True)
                heatmap = sns.heatmap(
                    normalized_counts,
                    cmap=cmap,
                    ax=ax,
                    cbar=False,
                    linewidths=0.5,
                    linecolor="white",
                    square=True,
                    vmin=0,
                    vmax=1,
                    norm=norm,
                )
                if shared_heatmap is None:
                    shared_heatmap = heatmap
                ax.set_xlabel(label_j)
                ax.set_ylabel(label_i)
                ax.set_title("")

    if shared_heatmap is not None:
        cbar = fig.colorbar(
            shared_heatmap.collections[0],
            ax=[axes[i, j] for i in range(n_labels) for j in range(n_labels) if i != j],
            fraction=0.05,
            pad=0.02,
            shrink=0.7,
        )
        cbar.set_label("Joint frequency (normalized)")
        cbar.set_ticks(np.linspace(0, 1, 5))

    fig.tight_layout()
    fig.savefig(output_path / "challenge1_correlations.png", dpi=300)
    plt.close(fig)

    return corr_df, pval_df


def plot_label_similarity_heatmap(label_df: pd.DataFrame, output_path: Path) -> Path:
    """Compute and visualize cosine similarity among labels."""

    similarity_matrix = cosine_similarity(label_df.T)
    sim_df = pd.DataFrame(similarity_matrix, index=label_df.columns, columns=label_df.columns)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        sim_df,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Label similarity matrix (cosine)")
    fig.tight_layout()

    output_file = output_path / "challenge1_label_similarity.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return output_file
