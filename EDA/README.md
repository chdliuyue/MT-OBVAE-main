# Exploratory Data Analysis (EDA)

This module generates correlation, similarity-divergence, and key-feature density reports for a selected dataset split (merged `train_old.csv` and `test_old.csv`). Outputs are organized under `output/EDA/` to keep artifacts in a single, predictable location.

## Usage

Run the pipeline from the repository root:

```bash
python -m EDA.main --data-root data/highD_ratio_20 --output-root output/EDA
```

Key arguments:

- `--data-root`: folder containing `train_old.csv` and `test_old.csv`.
- `--dataset-name`: optional label used in logs/report titles (defaults to `highD_ratio_20`).
- `--output-root`: base directory where results are written (artifacts now sit directly under this path).
- `--correlation-method`: `spearman` (default) or `pearson` for label correlation.
- `--similarity-threshold` / `--min-label-gap` / `--max-pairs` / `--neighbor-count`: control "同征不同果" retrieval (up to 10 pairs by default).
- `--log-level`: customize verbosity when running the pipeline.

## Generated artifacts

Artifacts include:

- `challenge1_correlations.png` (label histograms/joint grids) and `challenge1_label_similarity.png` (cosine similarity with three-decimal annotations).
- `challenge2_similarity_pairs.csv` with summary rows and `challenge2_radar_pair_<n>.png` visuals for divergent-outcome pairs (if any qualify).
- `challenge3_pairwise_kde_<x>_vs_<y>.png` standalone KDE plots for every feature combination (sampled to 2000 rows when needed), and `metrics_summary.csv` for skewness/kurtosis.
- `eda_report.md` pointing to all outputs so the generated paths are easy to find.
