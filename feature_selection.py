"""Feature Selection module using Boruta + Random Forest refined with SHAP.
Subsamples ~150k rows from FULL training data, keeping ALL Gas, SW, Speed, and RPM rows.
Derives L1 (11 features), L2 (11 features), and L3 (18 features) strictly from data.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-GUI backend: avoids Tk crash when plotting from worker threads
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

from config import (
    CACHE_DIR,
    MODELS_DIR,
    N_FEATURES_L1,
    N_FEATURES_L2,
    N_FEATURES_L3,
    RANDOM_SEED,
    RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_stratified_subsample(train_df: pd.DataFrame, target_size: int = 150000) -> pd.DataFrame:
    """Creates a subsample of target_size keeping 100% of Gas, SW, Speed, and RPM rows."""
    minority_classes = ["Gas", "SW", "Speed", "RPM"]
    minority_dfs = [train_df[train_df["level3"] == c] for c in minority_classes]
    minority_total = sum(len(df) for df in minority_dfs)
    logger.info(f"Retaining 100% of minority classes: {minority_total:,} rows ({[f'{c}: {len(df):,}' for c, df in zip(minority_classes, minority_dfs)]})")

    remaining_budget = target_size - minority_total
    if remaining_budget <= 0:
        return pd.concat(minority_dfs, ignore_index=True)

    # Allocate remaining budget between DoS and Benign (e.g. 25% DoS, 75% Benign)
    dos_df = train_df[train_df["level3"] == "DoS"]
    benign_df = train_df[train_df["level3"] == "Benign"]

    n_dos = min(len(dos_df), int(remaining_budget * 0.30))
    n_benign = remaining_budget - n_dos

    dos_sub = dos_df.sample(n=n_dos, random_state=RANDOM_SEED)
    benign_sub = benign_df.sample(n=n_benign, random_state=RANDOM_SEED)

    subsample_df = pd.concat(minority_dfs + [dos_sub, benign_sub], ignore_index=True)
    subsample_df = subsample_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    logger.info(f"Created subsample of {len(subsample_df):,} rows. Distribution:")
    for cls, cnt in subsample_df["level3"].value_counts().items():
        logger.info(f"  {cls:15s}: {cnt:8,d} ({cnt/len(subsample_df)*100:5.2f}%)")

    return subsample_df


def run_feature_selection() -> Dict[str, Any]:
    """Runs Boruta on training subsample, computes SHAP values for L1, L2, L3, and saves feature sets."""
    train_path = CACHE_DIR / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run data_prep.py first.")

    train_df = pd.read_parquet(train_path)
    with open(MODELS_DIR / "all_features.json", "r", encoding="utf-8") as f:
        all_features = json.load(f)

    # 1. Stratified subsample keeping ALL minority attack rows
    subsample_df = create_stratified_subsample(train_df, target_size=150000)

    X_sub = subsample_df[all_features].values
    y_l1 = (subsample_df["level1"] == "Attack").astype(int).values
    y_l2 = subsample_df["level2"].astype("category").cat.codes.values
    y_l3 = subsample_df["level3"].astype("category").cat.codes.values

    # 2. Boruta on binary (L1)
    logger.info(f"Running Boruta feature selection on {len(X_sub):,} rows with {X_sub.shape[1]} features...")
    rf_boruta = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    boruta_selector = BorutaPy(
        rf_boruta,
        n_estimators="auto",
        verbose=1,
        random_state=RANDOM_SEED,
        max_iter=30,
    )
    boruta_selector.fit(X_sub, y_l1)

    confirmed_mask = boruta_selector.support_
    tentative_mask = boruta_selector.support_weak_
    ranking = boruta_selector.ranking_

    confirmed_features = [f for f, conf in zip(all_features, confirmed_mask) if conf]
    logger.info(f"Boruta confirmed {len(confirmed_features)} features, {sum(tentative_mask)} tentative.")

    # Sort all features by Boruta ranking
    ranked_indices = np.argsort(ranking)
    ranked_features = [all_features[i] for i in ranked_indices]

    # Plot Boruta Ranking
    plt.figure(figsize=(12, 6))
    top_30_idx = ranked_indices[:30]
    plt.bar(range(30), [ranking[i] for i in top_30_idx], color="steelblue")
    plt.xticks(range(30), [all_features[i] for i in top_30_idx], rotation=90, fontsize=8)
    plt.ylabel("Boruta Rank (1 = Top Confirmed)")
    plt.title("Top 30 Features by Boruta Ranking (Full Dataset Subsample)")
    plt.tight_layout()
    boruta_plot_path = RESULTS_DIR / "boruta_ranking.png"
    plt.savefig(boruta_plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved Boruta ranking plot to {boruta_plot_path}")

    # 3. SHAP Refinement per level (strictly derived from data)
    sample_size = 600
    sample_indices = np.random.RandomState(RANDOM_SEED).choice(len(X_sub), sample_size, replace=False)
    X_shap_sample = X_sub[sample_indices]

    # L1: Binary model SHAP
    logger.info("Computing SHAP values for L1 (Binary)...")
    rf_l1 = RandomForestClassifier(n_estimators=40, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    rf_l1.fit(X_sub, y_l1)
    explainer_l1 = shap.TreeExplainer(rf_l1)
    shap_vals_l1 = explainer_l1.shap_values(X_shap_sample)
    if isinstance(shap_vals_l1, list):
        mean_shap_l1 = np.abs(shap_vals_l1[1]).mean(axis=0)
    else:
        mean_shap_l1 = np.abs(shap_vals_l1).mean(axis=0)
    l1_ranked_idx = np.argsort(mean_shap_l1)[::-1]
    l1_features = [all_features[i] for i in l1_ranked_idx[:N_FEATURES_L1]]

    # L2: 3-class model SHAP
    logger.info("Computing SHAP values for L2 (3-class)...")
    rf_l2 = RandomForestClassifier(n_estimators=40, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1)
    rf_l2.fit(X_sub, y_l2)
    explainer_l2 = shap.TreeExplainer(rf_l2)
    shap_vals_l2 = explainer_l2.shap_values(X_shap_sample)
    if isinstance(shap_vals_l2, list):
        mean_shap_l2 = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals_l2], axis=0)
    else:
        mean_shap_l2 = np.abs(shap_vals_l2).mean(axis=(0, 2)) if shap_vals_l2.ndim == 3 else np.abs(shap_vals_l2).mean(axis=0)
    l2_ranked_idx = np.argsort(mean_shap_l2)[::-1]
    l2_features = [all_features[i] for i in l2_ranked_idx[:N_FEATURES_L2]]

    # L3: 6-class model SHAP
    logger.info("Computing SHAP values for L3 (6-class)...")
    rf_l3 = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=RANDOM_SEED, n_jobs=-1)
    rf_l3.fit(X_sub, y_l3)
    explainer_l3 = shap.TreeExplainer(rf_l3)
    shap_vals_l3 = explainer_l3.shap_values(X_shap_sample)
    if isinstance(shap_vals_l3, list):
        mean_shap_l3 = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals_l3], axis=0)
    else:
        mean_shap_l3 = np.abs(shap_vals_l3).mean(axis=(0, 2)) if shap_vals_l3.ndim == 3 else np.abs(shap_vals_l3).mean(axis=0)
    l3_ranked_idx = np.argsort(mean_shap_l3)[::-1]
    # Derive top 18 strictly from data ranking
    l3_features = [all_features[i] for i in l3_ranked_idx[:N_FEATURES_L3]]

    # Save SHAP Summary Bar Plot for L3
    plt.figure(figsize=(10, 8))
    top_20_shap_idx = l3_ranked_idx[:20]
    plt.barh(range(20), mean_shap_l3[top_20_shap_idx][::-1], color="darkorange")
    plt.yticks(range(20), [all_features[i] for i in top_20_shap_idx][::-1], fontsize=9)
    plt.xlabel("Mean |SHAP Value| across all 6 classes")
    plt.title("Top 20 Features Ranked by SHAP Importance (6-Class)")
    plt.tight_layout()
    shap_plot_path = RESULTS_DIR / "shap_summary.png"
    plt.savefig(shap_plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved SHAP summary plot to {shap_plot_path}")

    # Check relation between sets
    set_l1 = set(l1_features)
    set_l2 = set(l2_features)
    set_l3 = set(l3_features)

    is_identical_1_2 = set_l1 == set_l2
    is_nested_1_in_3 = set_l1.issubset(set_l3)
    is_nested_2_in_3 = set_l2.issubset(set_l3)

    logger.info("Feature Set Relationship Analysis:")
    logger.info(f"  L1 features ({len(l1_features)}): {l1_features}")
    logger.info(f"  L2 features ({len(l2_features)}): {l2_features}")
    logger.info(f"  L3 features ({len(l3_features)}): {l3_features}")
    logger.info(f"  L1 and L2 identical: {is_identical_1_2} (Overlap: {len(set_l1 & set_l2)}/{len(set_l1)})")
    logger.info(f"  L1 nested in L3: {is_nested_1_in_3} (Overlap: {len(set_l1 & set_l3)}/{len(set_l1)})")
    logger.info(f"  L2 nested in L3: {is_nested_2_in_3} (Overlap: {len(set_l2 & set_l3)}/{len(set_l2)})")

    feature_sets = {
        "all": all_features,
        "L1": l1_features,
        "L2": l2_features,
        "L3": l3_features,
        "boruta_ranked": ranked_features,
        "shap_ranked_L1": [all_features[i] for i in l1_ranked_idx],
        "shap_ranked_L2": [all_features[i] for i in l2_ranked_idx],
        "shap_ranked_L3": [all_features[i] for i in l3_ranked_idx],
        "relationship": {
            "L1_L2_identical": bool(is_identical_1_2),
            "L1_in_L3_nested": bool(is_nested_1_in_3),
            "L2_in_L3_nested": bool(is_nested_2_in_3),
            "overlap_L1_L2": len(set_l1 & set_l2),
            "overlap_L1_L3": len(set_l1 & set_l3),
            "overlap_L2_L3": len(set_l2 & set_l3),
        },
    }

    feature_sets_path = MODELS_DIR / "feature_sets.json"
    with open(feature_sets_path, "w", encoding="utf-8") as f:
        json.dump(feature_sets, f, indent=4)

    logger.info(f"Saved feature sets to {feature_sets_path}")
    return feature_sets


if __name__ == "__main__":
    run_feature_selection()
