from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from scipy import stats

sns.set_theme(style="white", context="talk")


def plot_joint_kde(
    feature_df: pd.DataFrame,
    feature_pair: Tuple[str, str],
    output_dir: Path,
    subset_name: str,
) -> Path:
    """Plot a 2D kernel density estimation for a pair of key features."""

    x_col, y_col = feature_pair
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(
        data=feature_df,
        x=x_col,
        y=y_col,
        fill=True,
        thresh=0.05,
        cmap="crest",
        ax=ax,
    )
    ax.set_title(f"Challenge 3: KDE for {x_col} vs {y_col} ({subset_name})")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()

    output_path = output_dir / "challenge3_kde.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _annotated_hist(x, color, label=None, **kwargs):
    ax = plt.gca()
    series = pd.Series(x).dropna()
    sns.histplot(series, bins="auto", color=color, edgecolor="white", alpha=0.8, ax=ax)
    skewness = stats.skew(series)
    kurtosis = stats.kurtosis(series, fisher=False)
    annotation = AnchoredText(
        f"skew={skewness:.2f}\nkurt={kurtosis:.2f}",
        loc="upper right",
        frameon=True,
        prop={"size": 9},
    )
    annotation.patch.set_alpha(0.8)
    ax.add_artist(annotation)
    ax.set_ylabel("Count")


def plot_pairwise_kde_grid(feature_df: pd.DataFrame, output_dir: Path) -> Path:
    """Render KDE pairwise relationships for all feature combinations."""

    data = feature_df.copy()
    grid = sns.PairGrid(data, corner=True, diag_sharey=False)
    grid.map_lower(
        lambda x, y, **kwargs: sns.kdeplot(
            x=x,
            y=y,
            fill=True,
            thresh=0.02,
            levels=20,
            cmap="mako",
            **kwargs,
        )
    )
    grid.map_diag(_annotated_hist)

    grid.fig.subplots_adjust(top=0.95)
    grid.fig.suptitle("Challenge 3: Pairwise KDE across all features", fontsize=18)

    output_path = output_dir / "challenge3_pairwise_kde.png"
    grid.fig.savefig(output_path, dpi=300)
    plt.close(grid.fig)
    return output_path


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
