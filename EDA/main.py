from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from . import data_loader
from .correlation_analysis import plot_label_relationships
from .kde_analysis import compute_shape_metrics, plot_joint_kde, save_metrics
from .similarity_analysis import find_divergent_pairs, plot_pair_comparison, save_pair_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA pipeline for highD ratio datasets")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory that contains highD_ratio_* folders",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=None,
        help="Specific highD_ratio_* folders to process (defaults to discovering all)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root directory for generated assets; per subset outputs go to output/<ratio>/EDA",
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
        "--kde-features",
        nargs=2,
        metavar=("FEATURE_X", "FEATURE_Y"),
        default=["UF", "UAS"],
        help="Two key features for 2D KDE visualizations",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG)",
    )
    return parser.parse_args()


def discover_ratios(data_root: Path) -> List[str]:
    return sorted([p.name for p in data_root.glob("highD_ratio_*") if p.is_dir()])


def write_report(
    subset_name: str,
    output_dir: Path,
    correlation_paths: Path,
    pair_summary: Path | None,
    pair_plots: List[Path],
    kde_path: Path,
    metrics_path: Path,
) -> None:
    report_path = output_dir / "eda_report.md"
    pair_section = "\n".join([f"- {p.name}" for p in pair_plots]) if pair_plots else "- (no qualifying pairs found)"
    pair_summary_line = pair_summary.name if pair_summary else "(no qualifying pairs found)"

    report_content = f"""
# EDA outputs for {subset_name}

Generated artifacts are stored under: `{output_dir}`.

## Challenge 1: Label relationships
- Correlation grid & distributions: {correlation_paths / 'challenge1_correlations.png'}
- Correlation coefficients: {correlation_paths / 'challenge1_correlation_matrix.csv'}
- Correlation p-values: {correlation_paths / 'challenge1_correlation_pvalues.csv'}

## Challenge 2: Similar features, divergent outcomes
- Pair summary: {pair_summary_line}
- Pair visuals:\n{pair_section}

## Challenge 3: Key feature KDE & tail statistics
- KDE plot: {kde_path.name}
- Skewness/kurtosis table: {metrics_path.name}
"""
    report_path.write_text(report_content.strip() + "\n", encoding="utf-8")


def run_for_subset(args: argparse.Namespace, subset_name: str, logger: logging.Logger) -> None:
    data_dir = args.data_root / subset_name
    output_dir = args.output_root / subset_name / "EDA"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing subset %s", subset_name)
    features, labels = data_loader.load_training_split(data_dir)

    missing_kde_features = [f for f in args.kde_features if f not in features.columns]
    if missing_kde_features:
        raise ValueError(
            f"Requested KDE features {missing_kde_features} not found in dataset columns"
        )

    corr_dir = output_dir
    _ = plot_label_relationships(labels, corr_dir, method=args.correlation_method)
    logger.info("Correlation analysis completed for %s", subset_name)

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
        logger.info("Identified %d divergent pairs for %s", len(pairs), subset_name)
    else:
        logger.warning("No divergent pairs found for %s with current thresholds", subset_name)

    kde_path = plot_joint_kde(features, tuple(args.kde_features), output_dir, subset_name)
    metrics_df = compute_shape_metrics(features, args.kde_features)
    metrics_path = save_metrics(metrics_df, output_dir)

    write_report(
        subset_name,
        output_dir,
        corr_dir,
        pair_summary_path,
        pair_plots,
        kde_path,
        metrics_path,
    )
    logger.info("Finished processing %s. Outputs saved to %s", subset_name, output_dir)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("eda")

    subsets = args.ratios or discover_ratios(args.data_root)
    if not subsets:
        raise ValueError(f"No highD_ratio_* directories found under {args.data_root}")

    if len(args.kde_features) != 2:
        raise ValueError("--kde-features requires exactly two feature names")

    logger.info(
        "Running EDA for %d subset(s): %s", len(subsets), ", ".join(subsets)
    )
    for subset in subsets:
        run_for_subset(args, subset, logger)

    logger.info("EDA generation complete. Root outputs live under %s", args.output_root)


if __name__ == "__main__":
    main()
