import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


NATURE_PALETTE = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#7E6148",
    "#B09C85",
]


def _normalize_sensitivities(sens: np.ndarray) -> np.ndarray:
    abs_vals = np.abs(sens.squeeze())
    max_val = abs_vals.max() if abs_vals.max() != 0 else 1.0
    return abs_vals / max_val


def load_sensitivities(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


def build_sensitivity_table(sensitivities: Dict[str, np.ndarray], sort_dimensions: bool = False) -> pd.DataFrame:
    normalized = {task: _normalize_sensitivities(values) for task, values in sensitivities.items()}
    df = pd.DataFrame(normalized)
    df.index = [f"z{i}" for i in range(len(df))]

    if sort_dimensions:
        order = df.mean(axis=1).sort_values(ascending=False).index
        df = df.loc[order]
    return df


def plot_heatmap(table: pd.DataFrame, output_dir: str, figsize=(7, 5), prefix: str = "3_") -> None:
    os.makedirs(output_dir, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list("beta_nature", [NATURE_PALETTE[1], NATURE_PALETTE[2], NATURE_PALETTE[0]])
    plt.figure(figsize=figsize)
    plt.imshow(table.values, aspect="auto", cmap=cmap)
    plt.colorbar(label="Normalized |beta|")
    plt.xticks(ticks=range(table.shape[1]), labels=table.columns)
    plt.yticks(ticks=range(table.shape[0]), labels=table.index)
    plt.xlabel("Task")
    plt.ylabel("Latent dimension")
    plt.title("Decoder sensitivity profile")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}decoder_sensitivity_heatmap.png"), dpi=300)
    plt.close()


def plot_task_bars(table: pd.DataFrame, output_dir: str, figsize=(9, 4), prefix: str = "3_") -> None:
    os.makedirs(output_dir, exist_ok=True)
    num_tasks = table.shape[1]
    fig, axes = plt.subplots(1, num_tasks, figsize=figsize, sharey=True)
    if num_tasks == 1:
        axes = [axes]

    palette = NATURE_PALETTE
    for ax, (task, values) in zip(axes, table.items()):
        values.plot(kind="bar", ax=ax, color=palette[: len(values)])
        ax.set_title(task)
        ax.set_xlabel("Latent dim")
        ax.set_ylabel("Normalized |beta|")
        ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}decoder_sensitivity_bars.png"), dpi=300)
    plt.close()


def visualize_decoder_sensitivities(
    npz_path: str, output_dir: str, sort_dimensions: bool = False, prefix: str = "3_"
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    sensitivities = load_sensitivities(npz_path)
    table = build_sensitivity_table(sensitivities, sort_dimensions=sort_dimensions)
    csv_path = os.path.join(output_dir, f"{prefix}decoder_sensitivity_normalized.csv")
    table.to_csv(csv_path)

    plot_heatmap(table, output_dir, prefix=prefix)
    plot_task_bars(table, output_dir, prefix=prefix)
    return table
