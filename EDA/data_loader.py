from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

FEATURE_COLUMNS = [
    "UF",
    "UAS",
    "UD",
    "UAL",
    "DF",
    "DAS",
    "DD",
    "DAL",
    "rq_rel",
    "rk_rel",
    "CV_v",
    "E_BRK",
]

LABEL_COLUMNS = ["TTC_cls4", "DRAC_cls4", "PSD_cls4"]


def load_training_split(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the training split for a given data directory.

    Parameters
    ----------
    data_dir:
        Directory that contains ``train.csv``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Feature and label dataframes in the canonical column order.
    """

    csv_path = data_dir / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected training data at {csv_path}")

    df = pd.read_csv(csv_path)
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    missing_labels = [col for col in LABEL_COLUMNS if col not in df.columns]
    if missing_features or missing_labels:
        missing = ", ".join(missing_features + missing_labels)
        raise ValueError(f"Training data missing required columns: {missing}")

    feature_df = df[FEATURE_COLUMNS].copy()
    label_df = df[LABEL_COLUMNS].copy()
    return feature_df, label_df


def load_full_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and combine the legacy train/test splits for a unified EDA run.

    Parameters
    ----------
    data_dir:
        Directory that contains ``train_old.csv`` and ``test_old.csv``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Feature and label dataframes spanning the merged splits.
    """

    train_path = data_dir / "train_old.csv"
    test_path = data_dir / "test_old.csv"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Expected dataset split at {path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    missing_labels = [col for col in LABEL_COLUMNS if col not in df.columns]
    if missing_features or missing_labels:
        missing = ", ".join(missing_features + missing_labels)
        raise ValueError(f"Dataset missing required columns: {missing}")

    feature_df = df[FEATURE_COLUMNS].copy()
    label_df = df[LABEL_COLUMNS].copy()
    return feature_df, label_df
