# Exploratory Data Analysis (EDA)

This module generates correlation, similarity-divergence, and key-feature density reports for each available `highD_ratio_*` subset. Outputs follow the `output/<ratio>/EDA` structure so every subset keeps its own report folder.

## Usage

Run the pipeline from the repository root:

```bash
python -m EDA.main --data-root data --output-root output --kde-features UF UAS
```

Key arguments:

- `--ratios`: optional list of subfolders (defaults to auto-discovery of all `highD_ratio_*`).
- `--correlation-method`: `spearman` (default) or `pearson` for label correlation.
- `--similarity-threshold` / `--min-label-gap` / `--max-pairs` / `--neighbor-count`: control "同征不同果" retrieval.
- `--kde-features`: two features to plot for the KDE long-tail check.

## Generated artifacts

For each subset a dedicated `EDA` folder contains:

- `challenge1_correlations.png` plus CSV correlation matrices for label relationships.
- `challenge2_similarity_pairs.csv` with summary rows and `challenge2_radar_pair_<n>.png` visuals for divergent-outcome pairs (if any qualify).
- `challenge3_kde.png` KDE of the configured feature pair and `metrics_summary.csv` for skewness/kurtosis.
- `eda_report.md` pointing to all outputs so the generated paths are easy to find.
