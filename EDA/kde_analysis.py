from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="white", context="talk")


def plot_pairwise_kde_panels(feature_df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Render KDE plots for every feature pair as standalone figures."""

    max_rows = 2000
    data = feature_df.copy()
    if len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=42)
    data = data.reset_index(drop=True)

    output_paths: List[Path] = []
    kde_cmap = sns.color_palette("crest", as_cmap=True)
    for x_col, y_col in combinations(data.columns, 2):
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        sns.kdeplot(
            data=data,
            x=x_col,
            y=y_col,
            fill=True,
            thresh=0.03,
            levels=25,
            cmap=kde_cmap,
            ax=ax,
        )
        ax.set_title(f"Challenge 3: KDE for {x_col} vs {y_col}", pad=12)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig.tight_layout()

        output_path = output_dir / f"challenge3_pairwise_kde_{x_col}_vs_{y_col}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        output_paths.append(output_path)

    # Combined lower-triangle KDE grid
    grid = sns.PairGrid(data, corner=True, diag_sharey=False, height=2.6, aspect=1.05)
    grid.map_lower(
        sns.kdeplot,
        fill=True,
        thresh=0.03,
        levels=20,
        cmap=kde_cmap,
    )
    grid.map_diag(
        sns.kdeplot,
        fill=True,
        color="#4c72b0",
        alpha=0.7,
        linewidth=1.2,
        common_norm=False,
    )
    for ax in grid.axes.flatten():
        if ax is None:
            continue
        ax.tick_params(axis="both", labelsize=7, labelbottom=True, labelleft=True)
    grid.fig.set_size_inches(max(12, 2.6 * data.shape[1] * 0.65), max(12, 2.6 * data.shape[1] * 0.65))
    grid.fig.tight_layout(rect=[0, 0, 1, 0.96])
    grid.fig.subplots_adjust(top=0.95)
    grid.fig.suptitle("Challenge 3: Pairwise KDE overview (lower triangle)", fontsize=14)

    combined_path = output_dir / "challenge3_pairwise_kde_overview.png"
    grid.savefig(combined_path, dpi=300)
    plt.close(grid.fig)
    output_paths.insert(0, combined_path)

    return output_paths


def compute_shape_metrics(feature_df: pd.DataFrame, key_features: Iterable[str]) -> pd.DataFrame:
    """Calculate skewness and kurtosis for selected features."""

    records = []
    for col in key_features:
        values = feature_df[col].dropna()
        skewness = stats.skew(values)
        kurtosis = stats.kurtosis(values, fisher=False)
        records.append({"feature": col, "skewness": skewness, "kurtosis": kurtosis})
    return pd.DataFrame(records)


def save_metrics(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    metrics_path = output_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    return metrics_path
