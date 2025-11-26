import importlib
import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def _infer_density_feature(df: pd.DataFrame) -> Optional[str]:
    density_cols = [col for col in df.columns if "density" in col.lower()]
    return density_cols[0] if density_cols else None


def _ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def compute_embeddings(latent_df: pd.DataFrame, random_state: int = 42, perplexity: int = 30) -> pd.DataFrame:
    mu_cols = [col for col in latent_df.columns if col.startswith("mu_z_")]
    tsne_coords = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="random").fit_transform(
        latent_df[mu_cols]
    )
    latent_df = latent_df.copy()
    latent_df["tsne_1"], latent_df["tsne_2"] = tsne_coords[:, 0], tsne_coords[:, 1]

    spec = importlib.util.find_spec("umap")
    if spec is not None:
        umap_module = importlib.import_module("umap")
        reducer = umap_module.UMAP(random_state=random_state)
        umap_coords = reducer.fit_transform(latent_df[mu_cols])
        latent_df["umap_1"], latent_df["umap_2"] = umap_coords[:, 0], umap_coords[:, 1]

    return latent_df


def _assign_risk_label(row: pd.Series) -> str:
    if row.get("y_ttc", 0) >= 2 and row.get("y_drac", 0) >= 2:
        return "TTC&DRAC_high"
    if row.get("y_ttc", 0) >= 2:
        return "TTC_high"
    if row.get("y_drac", 0) >= 2:
        return "DRAC_high"
    if row.get("y_psd", 0) >= 2:
        return "PSD_high"
    return "Low risk"


def plot_density_coloring(df: pd.DataFrame, output_dir: str, density_feature: Optional[str] = None, figsize=(7, 6)) -> None:
    _ensure_output_dir(output_dir)
    density_feature = density_feature or _infer_density_feature(df)
    if density_feature is None:
        return

    plt.figure(figsize=figsize)
    sc = plt.scatter(df["tsne_1"], df["tsne_2"], c=df[density_feature], cmap="viridis", alpha=0.7, s=20)
    plt.colorbar(sc, label=density_feature)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Latent Space colored by traffic density")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase_space_density.png"), dpi=300)
    plt.close()


def plot_risk_coloring(df: pd.DataFrame, output_dir: str, figsize=(7, 6)) -> None:
    _ensure_output_dir(output_dir)
    df = df.copy()
    df["risk_type"] = df.apply(_assign_risk_label, axis=1)

    plt.figure(figsize=figsize)
    for risk_type, group in df.groupby("risk_type"):
        plt.scatter(group["tsne_1"], group["tsne_2"], label=risk_type, alpha=0.6, s=18)

    centroids = df.groupby("risk_type")[['tsne_1', 'tsne_2']].mean().reset_index()
    for _, row in centroids.iterrows():
        plt.scatter(row["tsne_1"], row["tsne_2"], color="black", marker="x", s=60)
        plt.text(row["tsne_1"], row["tsne_2"], row["risk_type"], fontsize=9, ha="left", va="bottom")

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Latent Space colored by risk type")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase_space_risk.png"), dpi=300)
    plt.close()

    centroids_path = os.path.join(output_dir, "risk_centroids.json")
    centroids.to_json(centroids_path, orient="records", indent=2)


def plot_risk_ambiguity_vs_density(latent_csv: str, output_dir: str, density_feature: Optional[str] = None, figsize=(7, 5)) -> None:
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
    plt.scatter(x, y, alpha=0.4, label="Samples", s=15)
    plt.plot(x_smooth, y_smooth, color="red", label="Quadratic fit")
    if not np.isnan(peak_x):
        plt.axvline(peak_x, color="gray", linestyle="--", alpha=0.7)
        plt.scatter([peak_x], [peak_y], color="black", zorder=5)
        plt.text(peak_x, peak_y, f"Peak=({peak_x:.2f}, {peak_y:.2f})", fontsize=9)

    plt.xlabel(density_feature)
    plt.ylabel("Risk Ambiguity Index")
    plt.title("Risk ambiguity vs traffic density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_ambiguity_density.png"), dpi=300)
    plt.close()

    summary = {
        "density_feature": density_feature,
        "quadratic_coefficients": coeffs.tolist(),
        "peak_density": peak_x,
        "peak_risk_ambiguity": float(peak_y),
    }
    json_path = os.path.join(output_dir, "risk_ambiguity_fit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def generate_phase_space(latent_csv: str, output_dir: str, density_feature: Optional[str] = None) -> pd.DataFrame:
    _ensure_output_dir(output_dir)
    latent_df = pd.read_csv(latent_csv)
    latent_df = compute_embeddings(latent_df)
    enriched_path = os.path.join(output_dir, "latent_space_with_tsne.csv")
    latent_df.to_csv(enriched_path, index=False)

    plot_density_coloring(latent_df, output_dir, density_feature=density_feature)
    plot_risk_coloring(latent_df, output_dir)
    plot_risk_ambiguity_vs_density(latent_csv, output_dir, density_feature=density_feature)
    return latent_df
