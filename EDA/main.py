from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from . import data_loader
from .correlation_analysis import plot_label_relationships, plot_label_similarity_heatmap
from .data_loader import FEATURE_COLUMNS
from .kde_analysis import compute_shape_metrics, plot_joint_kde, plot_pairwise_kde_grid, save_metrics
from .similarity_analysis import find_divergent_pairs, plot_pair_comparison, save_pair_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA pipeline for highD ratio datasets")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="highD_ratio_20",
        help="Optional dataset identifier for logging and reports (defaults to data directory name)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/highD_ratio_20"),
        help="Directory containing the target dataset splits",
    )
    parser.add_argument(
        "--split-version",
        choices=["current", "legacy"],
        default="legacy",
        help="Select which train/test pair to merge for analysis (current: train.csv/test.csv, legacy: train_old.csv/test_old.csv)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/EDA"),
        help="Directory for generated assets",
    )
    parser.add_argument(
        "--correlation-method",
        choices=["spearman", "pearson"],
        default="spearman",
        help="Correlation method for label association analysis",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for retrieving divergent outcome pairs",
    )
    parser.add_argument(
        "--min-label-gap",
        type=float,
        default=2.0,
        help="Minimum L1 difference across labels to mark outcomes as divergent",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=3,
        help="Maximum number of divergent pairs to visualize",
    )
    parser.add_argument(
        "--neighbor-count",
        type=int,
        default=25,
        help="Number of nearest neighbors to scan for divergent-outcome retrieval",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG)",
    )
    return parser.parse_args()


def write_report(
    output_dir: Path,
    dataset_name: str,
    split_version: str,
    correlation_paths: Path,
    pair_summary: Path | None,
    pair_plots: List[Path],
    kde_path: Path,
    pairwise_kde_path: Path,
    metrics_path: Path,
    similarity_map: Path,
) -> None:
    report_path = output_dir / "eda_report.md"
    pair_section = "\n".join([f"- {p.name}" for p in pair_plots]) if pair_plots else "- (no qualifying pairs found)"
    pair_summary_line = pair_summary.name if pair_summary else "(no qualifying pairs found)"

    split_label = "train.csv + test.csv" if split_version == "current" else "train_old.csv + test_old.csv"

    report_content = f"""
# EDA outputs for {dataset_name} (merged {split_label})

Generated artifacts are stored under: `{output_dir}`.

## Challenge 1: Label relationships
- Correlation grid & distributions: {correlation_paths / 'challenge1_correlations.png'}
- Label similarity (cosine): {similarity_map.name}

## Challenge 2: Similar features, divergent outcomes
- Pair summary: {pair_summary_line}
- Pair visuals:\n{pair_section}

## Challenge 3: Key feature KDE & tail statistics
- KDE plot (UF vs UAS): {kde_path.name}
- Pairwise KDE grid: {pairwise_kde_path.name}
- Skewness/kurtosis table: {metrics_path.name}
"""
    report_path.write_text(report_content.strip() + "\n", encoding="utf-8")


def run_full_pipeline(args: argparse.Namespace, logger: logging.Logger) -> None:
    dataset_name = args.dataset_name or args.data_root.name
    output_dir = args.output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading merged dataset from %s (%s splits)", args.data_root, args.split_version)
    features, labels = data_loader.load_full_dataset(args.data_root, split_version=args.split_version)

    corr_dir = output_dir
    _ = plot_label_relationships(labels, corr_dir, method=args.correlation_method)
    similarity_map = plot_label_similarity_heatmap(labels, corr_dir)
    logger.info("Correlation and similarity analysis completed")

    pairs = find_divergent_pairs(
        features,
        labels,
        similarity_threshold=args.similarity_threshold,
        min_label_gap=args.min_label_gap,
        max_pairs=args.max_pairs,
        neighbor_count=args.neighbor_count,
    )
    pair_plots: List[Path] = []
    pair_summary_path: Path | None = None
    if pairs:
        pair_summary_path = save_pair_summary(pairs, output_dir)
        for idx, pair in enumerate(pairs, start=1):
            pair_plots.append(plot_pair_comparison(features, labels, pair, output_dir, idx))
        logger.info("Identified %d divergent pairs", len(pairs))
    else:
        logger.warning("No divergent pairs found with current thresholds")

    kde_path = plot_joint_kde(features, ("UF", "UAS"), output_dir, dataset_name)
    pairwise_kde_path = plot_pairwise_kde_grid(features[FEATURE_COLUMNS], output_dir)
    metrics_df = compute_shape_metrics(features, FEATURE_COLUMNS)
    metrics_path = save_metrics(metrics_df, output_dir)

    write_report(
        output_dir,
        dataset_name,
        args.split_version,
        corr_dir,
        pair_summary_path,
        pair_plots,
        kde_path,
        pairwise_kde_path,
        metrics_path,
        similarity_map,
    )
    logger.info("Finished processing. Outputs saved to %s", output_dir)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("eda")

    dataset_name = args.dataset_name or args.data_root.name
    logger.info("Running unified EDA for %s", dataset_name)
    run_full_pipeline(args, logger)
    logger.info("EDA generation complete. Root outputs live under %s", args.output_root)


if __name__ == "__main__":
    main()
