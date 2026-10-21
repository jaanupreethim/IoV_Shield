"""Data loading, stratification, scaling, and duplicate analysis module.
Loads all 1.4M rows of CIC-IoV2024 binary CSVs as uint8 without deduplicating before split.
Performs stratified 70/15/15 split on level3 (seed 42).
Calculates deduplication and novel pattern statistics.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import (
    CACHE_DIR,
    DATA_DIR,
    DEDUP_BEFORE_SPLIT,
    FILENAME_LABEL_MAP,
    MODELS_DIR,
    RANDOM_SEED,
    RAW_LABEL_COLS,
    RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_csv_chunked(file_path: Path, chunk_size: int = 100000) -> pd.DataFrame:
    """Read large CSV in chunks and downcast features to uint8 to conserve memory."""
    filename = file_path.name
    if filename not in FILENAME_LABEL_MAP:
        logger.warning(f"Skipping unrecognized file: {filename}")
        return pd.DataFrame()

    l1, l2, l3 = FILENAME_LABEL_MAP[filename]
    chunks: List[pd.DataFrame] = []

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Drop raw label leakage columns
        cols_to_drop = [c for c in RAW_LABEL_COLS if c in chunk.columns]
        if cols_to_drop:
            chunk = chunk.drop(columns=cols_to_drop)

        # Downcast all feature columns to uint8
        feature_cols = [c for c in chunk.columns if c not in ["level1", "level2", "level3"]]
        chunk[feature_cols] = chunk[feature_cols].astype(np.uint8)

        # Assign hierarchical targets
        chunk["level1"] = l1
        chunk["level2"] = l2
        chunk["level3"] = l3

        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    return df


def prepare_data_full(force_recompute: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Loads all CSVs, builds labels, keeps ALL rows, performs stratified 70/15/15 split on level3,

    fits MinMaxScaler on train only, caches partitions, and computes duplicate stats.
    """
    clean_cache_path = CACHE_DIR / "clean.parquet"
    train_cache_path = CACHE_DIR / "train.parquet"
    val_cache_path = CACHE_DIR / "val.parquet"
    test_cache_path = CACHE_DIR / "test.parquet"
    features_json_path = MODELS_DIR / "all_features.json"
    scaler_path = MODELS_DIR / "scaler.joblib"

    if (
        not force_recompute
        and clean_cache_path.exists()
        and train_cache_path.exists()
        and val_cache_path.exists()
        and test_cache_path.exists()
        and features_json_path.exists()
        and scaler_path.exists()
    ):
        logger.info(f"Loading cached partitions from {CACHE_DIR}...")
        train_df = pd.read_parquet(train_cache_path)
        val_df = pd.read_parquet(val_cache_path)
        test_df = pd.read_parquet(test_cache_path)
        with open(features_json_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
        return train_df, val_df, test_df, feature_cols

    logger.info("Reading all raw CSV files from %s...", DATA_DIR)
    dfs = []
    for csv_file in sorted(DATA_DIR.glob("binary_*.csv")):
        logger.info(f"Loading {csv_file.name} (size: {csv_file.stat().st_size / (1024*1024):.2f} MB)...")
        df_sub = load_raw_csv_chunked(csv_file)
        if not df_sub.empty:
            dfs.append(df_sub)

    full_df = pd.concat(dfs, ignore_index=True)
    feature_cols = [c for c in full_df.columns if c not in ["level1", "level2", "level3"]]

    if DEDUP_BEFORE_SPLIT:
        before = len(full_df)
        full_df = full_df.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)
        logger.info(f"DEDUP variant: {before:,} -> {len(full_df):,} rows (unique CAN patterns only).")
        min_cls = full_df["level3"].value_counts().min()
        logger.info(f"Smallest class after dedup: {min_cls} rows.")
        if min_cls < 10:
            raise ValueError("A class has <10 unique patterns after dedup; stratified 70/15/15 split is not possible.")
    logger.info(f"Total rows in dataset: {len(full_df):,}, Total features: {len(feature_cols)}")

    # Class distribution
    full_dist = full_df["level3"].value_counts().to_dict()
    logger.info("Class distribution across full dataset:")
    for cls, cnt in full_dist.items():
        logger.info(f"  {cls:15s}: {cnt:10,d} ({cnt/len(full_df)*100:6.2f}%)")

    # Save feature list
    with open(features_json_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=4)

    # Cache full cleaned dataset
    logger.info("Caching full dataset to %s...", clean_cache_path)
    full_df.to_parquet(clean_cache_path, index=False)

    # Stratified Split: 70% Train, 15% Val, 15% Test on level3 (seed 42)
    logger.info("Performing 70/15/15 stratified train/val/test split on level3 (seed=%d)...", RANDOM_SEED)
    train_df, test_val_df = train_test_split(
        full_df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=full_df["level3"],
    )
    val_df, test_df = train_test_split(
        test_val_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=test_val_df["level3"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(f"Split sizes: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    # Fit MinMaxScaler on Train only
    logger.info("Fitting MinMaxScaler on Train set only...")
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    joblib.dump(scaler, scaler_path)
    logger.info("Saved MinMaxScaler to %s", scaler_path)

    # Cache partitions
    train_df.to_parquet(train_cache_path, index=False)
    val_df.to_parquet(val_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)
    logger.info("Saved cached train, val, and test parquet files to %s", CACHE_DIR)

    # -------------------------------------------------------------
    # Duplicate Analysis (Full data unique patterns + Train vs Test overlap)
    # -------------------------------------------------------------
    logger.info("Computing duplicate and novel pattern analysis...")
    dedup_stats = {}
    for cls in full_dist.keys():
        sub_full = full_df[full_df["level3"] == cls]
        cnt_full = len(sub_full)
        unique_full = len(sub_full.drop_duplicates(subset=feature_cols))
        dropped_full = cnt_full - unique_full

        sub_train = train_df[train_df["level3"] == cls]
        sub_test = test_df[test_df["level3"] == cls]

        # Extract unique feature tuples in train for this class
        train_patterns = set(map(tuple, sub_train[feature_cols].values))
        test_patterns = list(map(tuple, sub_test[feature_cols].values))

        # Check how many test rows match a train pattern
        overlap_mask = [pat in train_patterns for pat in test_patterns]
        n_test_overlap = sum(overlap_mask)
        n_test_total = len(sub_test)
        pct_test_in_train = (n_test_overlap / n_test_total * 100.0) if n_test_total > 0 else 0.0

        n_novel_test = n_test_total - n_test_overlap

        dedup_stats[cls] = {
            "total_rows": cnt_full,
            "unique_patterns": unique_full,
            "dropped_if_dedup": dropped_full,
            "drop_percent": (dropped_full / cnt_full * 100.0) if cnt_full > 0 else 0.0,
            "train_rows": len(sub_train),
            "test_rows": n_test_total,
            "test_rows_in_train": n_test_overlap,
            "pct_test_in_train": pct_test_in_train,
            "novel_test_rows": n_novel_test,
        }

    with open(RESULTS_DIR / "duplicate_analysis.json", "w", encoding="utf-8") as f:
        json.dump(dedup_stats, f, indent=4)

    print("\n" + "=" * 100)
    print("DUPLICATE PATTERN & TRAIN-TEST OVERLAP ANALYSIS")
    print("=" * 100)
    print(f"{'Class':<14} {'Total Rows':>12} {'Unique Pats':>12} {'Train Rows':>12} {'Test Rows':>12} {'% Test in Train':>16} {'Novel Test Rows':>16}")
    print("-" * 100)
    for cls, s in dedup_stats.items():
        print(f"{cls:<14} {s['total_rows']:>12,d} {s['unique_patterns']:>12,d} {s['train_rows']:>12,d} {s['test_rows']:>12,d} {s['pct_test_in_train']:>15.2f}% {s['novel_test_rows']:>16,d}")
    print("=" * 100 + "\n")

    return train_df, val_df, test_df, feature_cols


if __name__ == "__main__":
    prepare_data_full(force_recompute=True)
