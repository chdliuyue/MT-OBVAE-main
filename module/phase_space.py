import importlib
import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE


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
    method: str = "tsne",
) -> tuple[pd.DataFrame, str]:
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

    return latent_df, chosen


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

    max_val = max(valid_levels.values())
    dominant = [k for k, v in valid_levels.items() if v == max_val]
    level_names = {0: "Low", 1: "Medium", 2: "High"}
    level_label = level_names.get(max_val, f"Level {max_val}")

    if len(dominant) == 1:
        return f"{dominant[0]}-{level_label}"
    return f"Mixed-{level_label}"


def _get_embedding_columns(df: pd.DataFrame, method: str) -> tuple[str, str]:
    col1, col2 = f"{method}_1", f"{method}_2"
    if col1 in df.columns and col2 in df.columns:
        return col1, col2
    raise KeyError(f"Embedding columns for method {method} not found in dataframe.")


def plot_density_coloring(
    df: pd.DataFrame,
    output_dir: str,
    density_feature: Optional[str] = None,
    figsize=(7, 6),
    prefix: str = "2_",
    coord_cols: tuple[str, str] = ("tsne_1", "tsne_2"),
    embedding_label: str = "t-SNE",
) -> None:
    _ensure_output_dir(output_dir)
    density_feature = density_feature or _infer_density_feature(df)
    if density_feature is None:
        return

    plt.figure(figsize=figsize)
    cmap = _get_nature_cmap()
    sc = plt.scatter(
        df[coord_cols[0]], df[coord_cols[1]], c=df[density_feature], cmap=cmap, alpha=0.75, s=20
    )
    plt.colorbar(sc, label=density_feature)
    plt.xlabel(f"{embedding_label} 1")
    plt.ylabel(f"{embedding_label} 2")
    plt.title(f"Conflict phase space ({embedding_label}) — traffic density")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, _prefixed_name(prefix, f"phase_space_density_{embedding_label.lower()}.png")), dpi=300
    )
    plt.close()


def plot_traffic_state_coloring(
    df: pd.DataFrame,
    output_dir: str,
    figsize=(7, 6),
    prefix: str = "2_",
    coord_cols: tuple[str, str] = ("tsne_1", "tsne_2"),
    embedding_label: str = "t-SNE",
) -> None:
    _ensure_output_dir(output_dir)
    state_col = _infer_state_column(df)
    if state_col is None:
        return

    df = df.copy()
    df["traffic_state_label"] = df[state_col].apply(_map_traffic_state)
    palette = _get_palette(df["traffic_state_label"].nunique())

    plt.figure(figsize=figsize)
    for color, (state, group) in zip(palette, df.groupby("traffic_state_label")):
        plt.scatter(group[coord_cols[0]], group[coord_cols[1]], label=state, alpha=0.7, s=20, color=color)

    plt.xlabel(f"{embedding_label} 1")
    plt.ylabel(f"{embedding_label} 2")
    plt.title(f"Conflict phase space ({embedding_label}) — traffic regimes")
    plt.legend(title="Traffic state")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, _prefixed_name(prefix, f"phase_space_traffic_states_{embedding_label.lower()}.png")),
        dpi=300,
    )
    plt.close()


def plot_dominant_risk_coloring(
    df: pd.DataFrame,
    output_dir: str,
    figsize=(7, 6),
    prefix: str = "2_",
    coord_cols: tuple[str, str] = ("tsne_1", "tsne_2"),
    embedding_label: str = "t-SNE",
) -> None:
    _ensure_output_dir(output_dir)
    df = df.copy()
    df["dominant_risk"] = df.apply(_dominant_risk_label, axis=1)
    palette = _get_palette(df["dominant_risk"].nunique())

    plt.figure(figsize=figsize)
    for color, (risk_label, group) in zip(palette, df.groupby("dominant_risk")):
        plt.scatter(group[coord_cols[0]], group[coord_cols[1]], label=risk_label, alpha=0.7, s=18, color=color)

    plt.xlabel(f"{embedding_label} 1")
    plt.ylabel(f"{embedding_label} 2")
    plt.title(f"Conflict phase space ({embedding_label}) — dominant risk")
    plt.legend(title="Risk type")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, _prefixed_name(prefix, f"phase_space_dominant_risk_{embedding_label.lower()}.png")),
        dpi=300,
    )
    plt.close()


def plot_risk_ambiguity_vs_density(
    latent_csv: str, output_dir: str, density_feature: Optional[str] = None, figsize=(7, 5), prefix: str = "2_"
) -> None:
    _ensure_output_dir(output_dir)
    df = pd.read_csv(latent_csv)
    density_feature = density_feature or _infer_density_feature(df)
    if density_feature is None or "risk_ambiguity_index" not in df.columns:
        return

    x = df[density_feature].values
    y = df["risk_ambiguity_index"].values
    coeffs = np.polyfit(x, y, deg=2)
    poly = np.poly1d(coeffs)
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_smooth = poly(x_smooth)

    a, b, _ = coeffs
    peak_x = -b / (2 * a) if a != 0 else float("nan")
    peak_y = poly(peak_x) if not np.isnan(peak_x) else float("nan")

    plt.figure(figsize=figsize)
    plt.scatter(x, y, alpha=0.55, label="Samples", s=18, color=NATURE_PALETTE[0])
    plt.plot(x_smooth, y_smooth, color=NATURE_PALETTE[3], label="Quadratic fit", linewidth=2)
    if not np.isnan(peak_x):
        plt.axvline(peak_x, color=NATURE_PALETTE[2], linestyle="--", alpha=0.7)
        plt.scatter([peak_x], [peak_y], color=NATURE_PALETTE[4], zorder=5)
        plt.text(peak_x, peak_y, f"Peak=({peak_x:.2f}, {peak_y:.2f})", fontsize=9)

    plt.xlabel(density_feature)
    plt.ylabel("Risk Ambiguity Index")
    plt.title("Risk ambiguity vs traffic density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, _prefixed_name(prefix, "risk_ambiguity_density.png")), dpi=300)
    plt.close()

    summary = {
        "density_feature": density_feature,
        "quadratic_coefficients": coeffs.tolist(),
        "peak_density": peak_x,
        "peak_risk_ambiguity": float(peak_y),
    }
    json_path = os.path.join(output_dir, _prefixed_name(prefix, "risk_ambiguity_fit.json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


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

        ax.scatter(x, y, alpha=0.5, color=NATURE_PALETTE[0], s=18, label="Samples")
        try:
            coeffs = np.polyfit(x, y, deg=2)
            poly = np.poly1d(coeffs)
            x_smooth = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_smooth, poly(x_smooth), color=NATURE_PALETTE[3], linewidth=2, label="Quadratic fit")
        except np.linalg.LinAlgError:
            pass

        ax.set_xlabel(feature)
        ax.set_ylabel("Risk Ambiguity")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    for ax in axes.flatten()[num_features:]:
        ax.axis("off")

    fig.suptitle("Risk ambiguity vs. traffic features", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(os.path.join(output_dir, _prefixed_name(prefix, "risk_ambiguity_features.png")), dpi=300)
    plt.close(fig)


def generate_phase_space(
    latent_csv: str,
    output_dir: str,
    density_feature: Optional[str] = None,
    prefix: str = "2_",
    embedding_method: str = "tsne",
    perplexity: int = 30,
) -> pd.DataFrame:
    _ensure_output_dir(output_dir)
    latent_df = pd.read_csv(latent_csv)
    latent_df, chosen_method = compute_embeddings(latent_df, method=embedding_method, perplexity=perplexity)
    coord_cols = _get_embedding_columns(latent_df, chosen_method)
    embedding_label = chosen_method.upper()
    enriched_path = os.path.join(
        output_dir, _prefixed_name(prefix, f"latent_space_with_{embedding_label.lower()}.csv")
    )
    latent_df.to_csv(enriched_path, index=False)

    plot_density_coloring(
        latent_df, output_dir, density_feature=density_feature, prefix=prefix, coord_cols=coord_cols, embedding_label=embedding_label
    )
    plot_traffic_state_coloring(
        latent_df, output_dir, prefix=prefix, coord_cols=coord_cols, embedding_label=embedding_label
    )
    plot_dominant_risk_coloring(
        latent_df, output_dir, prefix=prefix, coord_cols=coord_cols, embedding_label=embedding_label
    )
    plot_risk_ambiguity_vs_density(latent_csv, output_dir, density_feature=density_feature, prefix=prefix)
    plot_risk_ambiguity_feature_relationships(latent_df, output_dir, prefix=prefix)
    return latent_df
