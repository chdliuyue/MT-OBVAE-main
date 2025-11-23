from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .data_loader import FEATURE_COLUMNS, LABEL_COLUMNS


ANCHOR_COLOR = "#1f78b4"  # deep blue for clear contrast
NEIGHBOR_COLOR = "#e31a1c"  # vivid red to stand apart from blue
ANCHOR_BAR_COLOR = "#a6cee3"  # light blue for bars
NEIGHBOR_BAR_COLOR = "#fcae91"  # warm coral for bars


@dataclass
class DivergentPair:
    """Container for a pair of samples with similar features but divergent labels."""

    anchor_index: int
    neighbor_index: int
    similarity: float
    label_gap: float


def find_divergent_pairs(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    similarity_threshold: float = 0.95,
    min_label_gap: float = 2.0,
    max_pairs: int = 10,
    neighbor_count: int = 25,
) -> List[DivergentPair]:
    """Find sample pairs with high feature similarity but distinct labels.

    Parameters
    ----------
    feature_df:
        Input feature dataframe.
    label_df:
        Label dataframe aligned to ``feature_df``.
    similarity_threshold:
        Minimum cosine similarity to qualify as a candidate pair.
    min_label_gap:
        Minimum L1 gap between label triplets to be considered divergent.
    max_pairs:
        Maximum number of pairs to return.
    neighbor_count:
        Number of nearest neighbors to scan per sample when searching for pairs.
    """

    if len(feature_df) < 2:
        return []

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df.values)

    n_neighbors = min(neighbor_count + 1, len(feature_df))
    nn = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors)
    nn.fit(scaled_features)
    distances, indices = nn.kneighbors(scaled_features, return_distance=True)
    similarities = 1.0 - distances

    candidates: List[DivergentPair] = []
    seen: set[Tuple[int, int]] = set()

    for i in range(similarities.shape[0]):
        for sim_value, neighbor_idx in zip(similarities[i, 1:], indices[i, 1:]):
            pair = tuple(sorted((i, neighbor_idx)))
            if pair in seen:
                continue

            similarity_score = float(sim_value)
            if similarity_score < similarity_threshold:
                break

            label_gap = np.abs(label_df.iloc[i].values - label_df.iloc[neighbor_idx].values).sum()
            if label_gap >= min_label_gap:
                candidates.append(
                    DivergentPair(
                        anchor_index=i,
                        neighbor_index=neighbor_idx,
                        similarity=float(similarity_score),
                        label_gap=float(label_gap),
                    )
                )
                seen.add(pair)
                if len(candidates) >= max_pairs:
                    return sorted(candidates, key=lambda p: (-p.label_gap, -p.similarity))

    return sorted(candidates, key=lambda p: (-p.label_gap, -p.similarity))


def _radar_factory(num_vars: int) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    return angles, np.cos(angles)


def plot_pair_comparison(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    pair: DivergentPair,
    output_dir: Path,
    pair_id: int,
) -> Path:
    """Plot radar and bar comparisons for a divergent pair."""

    feature_cols = FEATURE_COLUMNS
    label_cols = LABEL_COLUMNS
    feature_bounds = feature_df[feature_cols].agg(["min", "max"])
    denom = (feature_bounds.loc["max"] - feature_bounds.loc["min"]).replace(0, 1)
    sample_a_raw = feature_df.iloc[pair.anchor_index][feature_cols]
    sample_b_raw = feature_df.iloc[pair.neighbor_index][feature_cols]
    sample_a = ((sample_a_raw - feature_bounds.loc["min"]) / denom).values
    sample_b = ((sample_b_raw - feature_bounds.loc["min"]) / denom).values
    labels_a = label_df.iloc[pair.anchor_index][label_cols].values
    labels_b = label_df.iloc[pair.neighbor_index][label_cols].values

    angles, _ = _radar_factory(len(feature_cols))

    fig = plt.figure(figsize=(10, 5.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.8, 1], wspace=0.25)

    # Radar chart for features
    ax_radar = fig.add_subplot(grid[0, 0], polar=True)
    values_a = np.concatenate((sample_a, [sample_a[0]]))
    values_b = np.concatenate((sample_b, [sample_b[0]]))
    ax_radar.plot(angles, values_a, label=f"Anchor idx {pair.anchor_index}", color=ANCHOR_COLOR, linewidth=2)
    ax_radar.fill(angles, values_a, alpha=0.18, color=ANCHOR_COLOR)
    ax_radar.plot(angles, values_b, label=f"Neighbor idx {pair.neighbor_index}", color=NEIGHBOR_COLOR, linewidth=2)
    ax_radar.fill(angles, values_b, alpha=0.18, color=NEIGHBOR_COLOR)
    ax_radar.set_thetagrids(angles[:-1] * 180 / np.pi, labels=feature_cols, fontsize=9)
    ax_radar.set_title("Feature similarity (radar, normalized 0-1)")
    ax_radar.set_ylim(0, 1)

    # Bar chart for labels
    ax_bar = fig.add_subplot(grid[0, 1])
    positions = np.arange(len(label_cols))
    width = 0.35
    bars_a = ax_bar.bar(
        positions - width / 2,
        labels_a,
        width=width,
        label=f"Anchor idx {pair.anchor_index}",
        color=ANCHOR_BAR_COLOR,
    )
    bars_b = ax_bar.bar(
        positions + width / 2,
        labels_b,
        width=width,
        label=f"Neighbor idx {pair.neighbor_index}",
        color=NEIGHBOR_BAR_COLOR,
        alpha=0.8,
    )
    ax_bar.set_xticks(positions, label_cols)
    ax_bar.set_ylabel("Label level")
    ax_bar.set_title("Label divergence (higher = riskier)")
    ax_bar.set_yticks(np.arange(0, 4, 1))
    ax_bar.tick_params(axis="x", labelrotation=15, labelsize=10)

    y_max = max(np.max(labels_a), np.max(labels_b)) if len(labels_a) and len(labels_b) else 3
    ax_bar.set_ylim(0, max(3.2, y_max + 0.4))

    for bar_a, bar_b in zip(bars_a, bars_b):
        ax_bar.text(bar_a.get_x() + bar_a.get_width() / 2, bar_a.get_height() + 0.05, f"{bar_a.get_height():.0f}", ha="center", va="bottom", fontsize=11)
        ax_bar.text(bar_b.get_x() + bar_b.get_width() / 2, bar_b.get_height() + 0.05, f"{bar_b.get_height():.0f}", ha="center", va="bottom", fontsize=11)
        delta = bar_b.get_height() - bar_a.get_height()
        midpoint = (bar_a.get_x() + bar_b.get_x() + bar_b.get_width()) / 2
        ax_bar.text(midpoint, max(bar_a.get_height(), bar_b.get_height()) + 0.15, f"Δ={delta:.1f}", ha="center", va="bottom", fontsize=10, color="#4b4b4b")

    handles, labels = [], []
    for ax in (ax_radar, ax_bar):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=2,
        frameon=False,
    )

    fig.suptitle(
        "Challenge 2: Top high-similarity pairs ranked by label gap\n"
        f"Pair {pair_id}: sim={pair.similarity:.3f}, label gap={pair.label_gap:.1f} (indices from merged dataset)",
        fontsize=14,
        y=1.08,
    )

    output_path = output_dir / f"challenge2_radar_pair_{pair_id}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_pair_summary(pairs: Sequence[DivergentPair], output_dir: Path) -> Path:
    """Save a tabular summary of the divergent pairs."""

    df = pd.DataFrame(
        [
            {
                "pair_id": idx + 1,
                "anchor_index": pair.anchor_index,
                "neighbor_index": pair.neighbor_index,
                "similarity": pair.similarity,
                "label_gap": pair.label_gap,
            }
            for idx, pair in enumerate(pairs)
        ]
    )
    summary_path = output_dir / "challenge2_similarity_pairs.csv"
    df.to_csv(summary_path, index=False)
    return summary_path
