# Exploratory Data Analysis (EDA)

This module generates correlation, similarity-divergence, and key-feature density reports for a selected dataset split (merged `train_old.csv` and `test_old.csv`). Outputs are organized under `output/EDA/<dataset_name>` so every dataset keeps its own report folder.

## Usage

Run the pipeline from the repository root:

```bash
python -m EDA.main --data-root data/highD_ratio_20 --output-root output/EDA --dataset-name highD_ratio_20
```

Key arguments:

- `--data-root`: folder containing `train_old.csv` and `test_old.csv`.
- `--dataset-name`: optional label used in logs/report titles (defaults to the name of `--data-root`).
- `--output-root`: base directory where results are written (each dataset gets its own subfolder).
- `--correlation-method`: `spearman` (default) or `pearson` for label correlation.
- `--similarity-threshold` / `--min-label-gap` / `--max-pairs` / `--neighbor-count`: control "同征不同果" retrieval.
- `--log-level`: customize verbosity when running the pipeline.

## Generated artifacts

For each dataset a dedicated `EDA/<dataset_name>` folder contains:

- `challenge1_correlations.png` plus CSV correlation matrices for label relationships.
- `challenge2_similarity_pairs.csv` with summary rows and `challenge2_radar_pair_<n>.png` visuals for divergent-outcome pairs (if any qualify).
- `challenge3_kde.png` KDE of the configured feature pair and `metrics_summary.csv` for skewness/kurtosis.
- `eda_report.md` pointing to all outputs so the generated paths are easy to find.
