import json
import os
from typing import Dict, List

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
        ax.set_ylabel(r"Normalized |$\beta$|")
        ax.grid(True, linestyle="--", alpha=0.5)
        for patch in ax.patches:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height(),
                f"{patch.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                rotation=0,
            )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}decoder_sensitivity_bars.png"), dpi=300)
    plt.close()


def visualize_decoder_sensitivities(
    npz_path: str,
    output_dir: str,
    sort_dimensions: bool = False,
    prefix: str = "3_",
    latent_df: pd.DataFrame | None = None,
    feature_limit: int = 12,
    top_k: int = 3,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    sensitivities = load_sensitivities(npz_path)
    table = build_sensitivity_table(sensitivities, sort_dimensions=sort_dimensions)
    csv_path = os.path.join(output_dir, f"{prefix}decoder_sensitivity_normalized.csv")
    table.to_csv(csv_path)

    plot_task_bars(table, output_dir, prefix=prefix)

    if latent_df is not None:
        influence_json = os.path.join(output_dir, f"{prefix}latent_feature_influence.json")
        summary = summarize_feature_latent_links(
            latent_df, table, output_path=influence_json, feature_limit=feature_limit, top_k=top_k
        )
        print(f"特征-隐空间影响关系已保存至 {influence_json}:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return table


def _select_observable_features(df: pd.DataFrame, limit: int = 12) -> List[str]:
    excluded_prefixes = ("mu_z_", "log_var_z_", "y_", "tsne_", "umap_", "risk_", "sample_id")
    candidates = [
        col for col in df.columns if not any(col.startswith(prefix) for prefix in excluded_prefixes)
    ]
    return candidates[:limit]


def summarize_feature_latent_links(
    latent_df: pd.DataFrame,
    sensitivity_table: pd.DataFrame,
    output_path: str,
    feature_limit: int = 12,
    top_k: int = 3,
) -> dict:
    feature_cols = _select_observable_features(latent_df, limit=feature_limit)
    mu_cols = [col for col in latent_df.columns if col.startswith("mu_z_")]

    correlation_rows = []
    for dim_idx, mu_col in enumerate(mu_cols):
        for feature in feature_cols:
            corr = np.corrcoef(latent_df[feature], latent_df[mu_col])[0, 1]
            if np.isnan(corr):
                corr = 0.0
            correlation_rows.append(
                {"latent_dim": dim_idx, "feature": feature, "corr": float(corr), "abs_corr": float(abs(corr))}
            )

    corr_df = pd.DataFrame(correlation_rows)
    corr_df.to_csv(output_path.replace(".json", "_matrix.csv"), index=False)

    latent_top_features = {}
    for dim_idx in range(len(mu_cols)):
        top_features = (
            corr_df[corr_df["latent_dim"] == dim_idx]
            .sort_values("abs_corr", ascending=False)
            .head(top_k)
        )
        latent_top_features[f"z{dim_idx}"] = top_features[["feature", "corr", "abs_corr"]].to_dict(orient="records")

    task_top_dims = {}
    for task, values in sensitivity_table.items():
        sorted_dims = values.sort_values(ascending=False).head(top_k)
        task_top_dims[task] = [
            {"latent_dim": dim, "normalized_weight": float(weight)} for dim, weight in sorted_dims.items()
        ]

    summary = {
        "task_top_latent_dims": task_top_dims,
        "latent_dim_top_features": latent_top_features,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
