import importlib
import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE


plt.rcParams.update({"font.family": "Times New Roman"})

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


def _infer_density_feature(df: pd.DataFrame) -> Optional[str]:
    density_cols = [col for col in df.columns if "density" in col.lower()]
    return density_cols[0] if density_cols else None


def _ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def _prefixed_name(prefix: str, filename: str) -> str:
    return f"{prefix}{filename}" if prefix else filename


def _get_nature_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "nature_continuous", [NATURE_PALETTE[1], NATURE_PALETTE[2], NATURE_PALETTE[0]]
    )


def _get_palette(num_colors: int) -> list[str]:
    if num_colors <= len(NATURE_PALETTE):
        return NATURE_PALETTE[:num_colors]
    repeats = (num_colors + len(NATURE_PALETTE) - 1) // len(NATURE_PALETTE)
    return (NATURE_PALETTE * repeats)[:num_colors]


def compute_embeddings(
    latent_df: pd.DataFrame,
    random_state: int = 42,
    perplexity: int = 30,
    method: str = "both",
) -> tuple[pd.DataFrame, str, list[str]]:
    mu_cols = [col for col in latent_df.columns if col.startswith("mu_z_")]
    latent_df = latent_df.copy()
    requested = method.lower()
    available_methods = []

    if requested in {"tsne", "both"}:
        tsne_coords = TSNE(
            n_components=2, perplexity=perplexity, random_state=random_state, init="random"
        ).fit_transform(latent_df[mu_cols])
        latent_df["tsne_1"], latent_df["tsne_2"] = tsne_coords[:, 0], tsne_coords[:, 1]
        available_methods.append("tsne")

    if requested in {"umap", "both"}:
        spec = importlib.util.find_spec("umap")
        if spec is not None:
            umap_module = importlib.import_module("umap")
            reducer = umap_module.UMAP(random_state=random_state)
            umap_coords = reducer.fit_transform(latent_df[mu_cols])
            latent_df["umap_1"], latent_df["umap_2"] = umap_coords[:, 0], umap_coords[:, 1]
            available_methods.append("umap")
        else:
            print("UMAP 未安装，已自动跳过。若需UMAP嵌入，请安装 umap-learn 包。")

    if not available_methods:
        raise ValueError("未能生成任何嵌入，至少需要支持 t-SNE 或 UMAP。")

    if requested == "umap" and "umap" in available_methods:
        chosen = "umap"
    elif requested == "both" and "umap" in available_methods:
        chosen = "umap"
    else:
        chosen = available_methods[0]

    return latent_df, chosen, available_methods


def _infer_state_column(df: pd.DataFrame) -> Optional[str]:
    candidates = {"traffic_state", "traffic_phase", "flow_state", "traffic_mode"}
    for col in df.columns:
        lower = col.lower()
        if lower in candidates:
            return col
    return None


def _map_traffic_state(value: object) -> str:
    mapping = {0: "Free flow", 1: "Synchronized flow", 2: "Congested flow"}
    if isinstance(value, str):
        lower = value.strip().lower()
        if "free" in lower:
            return "Free flow"
        if "sync" in lower or "synchron" in lower:
            return "Synchronized flow"
        if "congest" in lower:
            return "Congested flow"
        return value
    return mapping.get(int(value), str(value))


def _dominant_risk_label(row: pd.Series) -> str:
    levels = {"TTC": row.get("y_ttc"), "DRAC": row.get("y_drac"), "PSD": row.get("y_psd")}
    valid_levels = {k: int(v) for k, v in levels.items() if pd.notna(v)}
    if not valid_levels:
        return "Unknown"

    level_order = ["TTC", "DRAC", "PSD"]
    max_val = max(valid_levels.values())
    level_names = {0: "Low", 1: "Medium", 2: "High", 3: "Level 3"}
    level_label = level_names.get(max_val, f"Level {max_val}")

    for task in level_order:
        if valid_levels.get(task, -1) == max_val:
            return f"{task}-{level_label}"

    return "Unknown"

def _get_embedding_columns(df: pd.DataFrame, method: str) -> tuple[str, str]:
    col1, col2 = f"{method}_1", f"{method}_2"
    if col1 in df.columns and col2 in df.columns:
        return col1, col2
    raise KeyError(f"Embedding columns for method {method} not found in dataframe.")


def _task_risk_palette() -> dict[int, str]:
    return {0: "#00ff00", 1: "#00e5ff", 2: "#a3af9e", 3: "#3f4a3c"}


def plot_task_risk_grids(
    df: pd.DataFrame,
    output_dir: str,
    available_methods: list[str],
    prefix: str = "1_",
) -> None:
    _ensure_output_dir(output_dir)
    risk_labels = {0: "No risk", 1: "Low risk", 2: "Medium risk", 3: "High risk"}
    palette = _task_risk_palette()
    tasks = [("TTC", "y_ttc"), ("DRAC", "y_drac"), ("PSD", "y_psd")]

    for method in available_methods:
        coord_cols = _get_embedding_columns(df, method)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)

        for ax, (task_label, target_col) in zip(axes, tasks):
            if target_col not in df.columns:
                ax.set_visible(False)
                continue
            for risk_value, group in df.groupby(target_col):
                if pd.isna(risk_value):
                    continue
                try:
                    risk_value = int(risk_value)
                except (TypeError, ValueError):
                    continue
                ax.scatter(
                    group[coord_cols[0]],
                    group[coord_cols[1]],
                    s=18,
                    alpha=0.72,
                    label=risk_labels.get(risk_value, str(risk_value)),
                    color=palette.get(risk_value, "#777777"),
                )

            ax.set_title(f"{task_label}", fontsize=16)
            ax.set_xlabel(f"{method.upper()} 1", fontsize=16)
            ax.set_ylabel(f"{method.upper()} 2", fontsize=16)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(labelsize=16)
            ax.legend(fontsize=16)

        # fig.suptitle(f"Conflict phase space ({method.upper()}) — task-wise risk types", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        plt.savefig(
            os.path.join(output_dir, _prefixed_name(prefix, f"phase_space_dominant_risk_{method}.png")), dpi=300
        )
        plt.close(fig)

def _select_feature_columns(df: pd.DataFrame, limit: int = 12) -> list[str]:
    excluded_prefixes = ("mu_z_", "log_var_z_", "y_", "tsne_", "umap_", "risk_", "sample_id")
    candidates = [
        col
        for col in df.columns
        if not any(col.startswith(prefix) for prefix in excluded_prefixes)
    ]
    return candidates[:limit]


def plot_risk_ambiguity_feature_relationships(
    df: pd.DataFrame, output_dir: str, prefix: str = "2_", max_features: int = 12
) -> None:
    if "risk_ambiguity_index" not in df.columns:
        return

    feature_cols = _select_feature_columns(df, limit=max_features)
    if not feature_cols:
        return

    _ensure_output_dir(output_dir)
    num_features = len(feature_cols)
    rows = int(np.ceil(num_features / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows), squeeze=False)

    for ax, feature in zip(axes.flatten(), feature_cols):
        x = df[feature].values
        y = df["risk_ambiguity_index"].values

        ax.scatter(x, y, alpha=0.55, color="#2E8B57", s=20, label="Samples")
        try:
            coeffs = np.polyfit(x, y, deg=2)
            poly = np.poly1d(coeffs)
            x_smooth = np.linspace(x.min(), x.max(), 200)
            ax.plot(
                x_smooth,
                poly(x_smooth),
                color="#000000",
                linewidth=3,
                label="Quadratic fit",
            )
        except np.linalg.LinAlgError:
            pass

        ax.set_xlabel(feature, fontsize=16)
        ax.set_ylabel("Risk Ambiguity", fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=16)
        ax.tick_params(labelsize=16)

    for ax in axes.flatten()[num_features:]:
        ax.axis("off")

    # fig.suptitle("Risk ambiguity vs. traffic features", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(os.path.join(output_dir, _prefixed_name(prefix, "risk_ambiguity_features.png")), dpi=300)
    plt.close(fig)


def generate_phase_space(
    latent_csv: str,
    output_dir: str,
    density_feature: Optional[str] = None,
    prefix: str = "2_",
    embedding_method: str = "both",
    perplexity: int = 30,
    latent_prefix: Optional[str] = None,
) -> pd.DataFrame:
    _ensure_output_dir(output_dir)
    latent_df = pd.read_csv(latent_csv)
    latent_df, chosen_method, available_methods = compute_embeddings(
        latent_df, method=embedding_method, perplexity=perplexity
    )

    embedding_label = chosen_method.upper()
    latent_prefix = latent_prefix if latent_prefix is not None else prefix
    enriched_path = os.path.join(
        output_dir, _prefixed_name(latent_prefix, f"latent_space_with_{embedding_label.lower()}.csv")
    )
    latent_df.to_csv(enriched_path, index=False)

    plot_task_risk_grids(latent_df, output_dir, available_methods, prefix="1_")

    plot_risk_ambiguity_feature_relationships(latent_df, output_dir, prefix=prefix)
    return latent_df
