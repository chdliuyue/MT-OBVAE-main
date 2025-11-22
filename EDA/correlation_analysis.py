from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")


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

    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            ax = axes[i, j]
            if i == j:
                sns.histplot(label_df[label_i], bins="auto", discrete=True, ax=ax, color="#5B8FF9")
                ax.set_xlabel(label_i)
                ax.set_ylabel("Count")
                ax.set_title(f"{label_i} distribution")
            else:
                joint_counts = pd.crosstab(label_df[label_i], label_df[label_j])
                sns.heatmap(joint_counts, cmap="mako", ax=ax, cbar=True)
                ax.set_xlabel(label_j)
                ax.set_ylabel(label_i)
                ax.set_title(f"{label_i} vs {label_j}")

    fig.suptitle("Challenge 1: Label correlations & distributions", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "challenge1_correlations.png", dpi=300)
    plt.close(fig)

    corr_df.to_csv(output_path / "challenge1_correlation_matrix.csv", index=True)
    pval_df.to_csv(output_path / "challenge1_correlation_pvalues.csv", index=True)

    return corr_df, pval_df
