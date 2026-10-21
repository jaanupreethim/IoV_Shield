"""Diagnostic module to analyze classifier confidence, entropy, missed attack patterns,

and leave-pattern-out generalization limit.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from config import (
    CACHE_DIR,
    L1_CLASSES,
    L2_CLASSES,
    L3_CLASSES,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.classifiers import HierarchicalIDSClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_confidence_and_entropy(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes max-probability confidence and Shannon entropy (base 2)."""
    conf = np.max(probs, axis=1)
    eps = 1e-12
    entropy = -np.sum(probs * np.log2(probs + eps), axis=1)
    return conf, entropy


def summarize_distribution(arr: np.ndarray) -> Dict[str, float]:
    """Returns descriptive statistics and quantiles."""
    if len(arr) == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "q90": 0.0, "q99": 0.0}
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q75": float(np.percentile(arr, 75)),
        "q90": float(np.percentile(arr, 90)),
        "q99": float(np.percentile(arr, 99)),
    }


def run_confidence_entropy_diagnostics(test_df: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> Dict[str, Any]:
    """Analyzes confidence and entropy distributions for L1 and L2 across:

    (a) True benign correctly classified
    (b) Attacks correctly detected
    (c) Attacks missed (labelled benign)
    Computes AUC of confidence for separating (c) from (a).
    """
    logger.info("Running Confidence and Entropy Diagnostics for L1 and L2...")

    # Load frozen L1 and L2 models
    l1_clf: HierarchicalIDSClassifier = joblib.load(MODELS_DIR / "L1_vehicle.joblib")
    l2_clf: HierarchicalIDSClassifier = joblib.load(MODELS_DIR / "L2_rsu.joblib")

    X_test_l1 = test_df[feature_sets["L1"]].values
    X_test_l2 = test_df[feature_sets["L2"]].values

    y_true_binary = (test_df["level1"] == "Attack").values  # True if Attack
    y_true_l2 = test_df["level2"].values
    y_true_l3 = test_df["level3"].values

    probs_l1 = l1_clf.predict_proba(X_test_l1)
    preds_l1 = l1_clf.predict(X_test_l1)
    conf_l1, ent_l1 = compute_confidence_and_entropy(probs_l1)

    probs_l2 = l2_clf.predict_proba(X_test_l2)
    preds_l2 = l2_clf.predict(X_test_l2)
    conf_l2, ent_l2 = compute_confidence_and_entropy(probs_l2)

    results = {}

    for model_name, preds, conf, ent, probs, classes in [
        ("L1_vehicle", preds_l1, conf_l1, ent_l1, probs_l1, L1_CLASSES),
        ("L2_rsu", preds_l2, conf_l2, ent_l2, probs_l2, L2_CLASSES),
    ]:
        logger.info(f"Processing diagnostics for {model_name}...")
        # (a) True benign correctly classified
        mask_a = (~y_true_binary) & (preds == "Benign")
        # (b) Attacks correctly detected
        mask_b = (y_true_binary) & (preds != "Benign")
        # (c) Attacks missed (labelled Benign)
        mask_c = (y_true_binary) & (preds == "Benign")

        logger.info(f"{model_name}: (a) True Benign={np.sum(mask_a):,}, (b) Attacks Detected={np.sum(mask_b):,}, (c) Attacks Missed={np.sum(mask_c):,}")

        conf_a, ent_a = conf[mask_a], ent[mask_a]
        conf_b, ent_b = conf[mask_b], ent[mask_b]
        conf_c, ent_c = conf[mask_c], ent[mask_c]

        # AUC for separating (c) from (a):
        # We want to see if low confidence or high entropy identifies missed attacks among all samples predicted as Benign!
        # Mask of all samples predicted as Benign:
        mask_pred_benign = (preds == "Benign")
        y_is_missed_attack = y_true_binary[mask_pred_benign].astype(int)  # 1 if missed attack, 0 if true benign
        scores_conf_inverse = 1.0 - conf[mask_pred_benign]  # higher uncertainty -> more likely to be missed attack
        scores_entropy = ent[mask_pred_benign]

        if len(np.unique(y_is_missed_attack)) == 2:
            auc_conf = float(roc_auc_score(y_is_missed_attack, scores_conf_inverse))
            auc_ent = float(roc_auc_score(y_is_missed_attack, scores_entropy))
        else:
            auc_conf, auc_ent = 0.5, 0.5

        # Plot Histograms
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].hist(conf_a, bins=30, alpha=0.5, label=f"(a) True Benign ({len(conf_a):,})", color="green", density=True)
        axes[0].hist(conf_b, bins=30, alpha=0.5, label=f"(b) Detected Attack ({len(conf_b):,})", color="orange", density=True)
        axes[0].hist(conf_c, bins=30, alpha=0.7, label=f"(c) Missed Attack ({len(conf_c):,})", color="red", density=True)
        axes[0].set_title(f"{model_name}: Max-Probability Confidence Distribution\nAUC (Separating Missed vs True Benign) = {auc_conf:.4f}")
        axes[0].set_xlabel("Confidence (Max Proba)")
        axes[0].set_ylabel("Density")
        axes[0].legend()

        axes[1].hist(ent_a, bins=30, alpha=0.5, label=f"(a) True Benign", color="green", density=True)
        axes[1].hist(ent_b, bins=30, alpha=0.5, label=f"(b) Detected Attack", color="orange", density=True)
        axes[1].hist(ent_c, bins=30, alpha=0.7, label=f"(c) Missed Attack", color="red", density=True)
        axes[1].set_title(f"{model_name}: Shannon Entropy Distribution\nAUC (Entropy) = {auc_ent:.4f}")
        axes[1].set_xlabel("Entropy (bits)")
        axes[1].set_ylabel("Density")
        axes[1].legend()

        plt.tight_layout()
        hist_path = RESULTS_DIR / f"{model_name}_conf_entropy_dist.png"
        plt.savefig(hist_path, dpi=200)
        plt.close()
        logger.info(f"Saved histogram plot to {hist_path}")

        results[model_name] = {
            "counts": {"true_benign_a": int(np.sum(mask_a)), "attacks_detected_b": int(np.sum(mask_b)), "attacks_missed_c": int(np.sum(mask_c))},
            "confidence_dist": {
                "a_true_benign": summarize_distribution(conf_a),
                "b_detected_attack": summarize_distribution(conf_b),
                "c_missed_attack": summarize_distribution(conf_c),
            },
            "entropy_dist": {
                "a_true_benign": summarize_distribution(ent_a),
                "b_detected_attack": summarize_distribution(ent_b),
                "c_missed_attack": summarize_distribution(ent_c),
            },
            "auc_separating_c_from_a": {
                "auc_uncertainty_score": auc_conf,
                "auc_entropy_score": auc_ent,
            },
        }

    return results


def run_missed_patterns_analysis(test_df: pd.DataFrame, feature_sets: Dict[str, List[str]], all_features: List[str]) -> Dict[str, Any]:
    """Analyzes which attack classes and specific bit patterns are missed at L1 and L2."""
    logger.info("Analyzing Miss Rate per Attack Class and Unique Patterns Missed...")

    l1_clf: HierarchicalIDSClassifier = joblib.load(MODELS_DIR / "L1_vehicle.joblib")
    l2_clf: HierarchicalIDSClassifier = joblib.load(MODELS_DIR / "L2_rsu.joblib")

    X_test_l1 = test_df[feature_sets["L1"]].values
    X_test_l2 = test_df[feature_sets["L2"]].values

    preds_l1 = l1_clf.predict(X_test_l1)
    preds_l2 = l2_clf.predict(X_test_l2)

    attack_classes = ["DoS", "Gas", "RPM", "Speed", "SW"]
    miss_analysis = {}

    for model_name, preds in [("L1_vehicle", preds_l1), ("L2_rsu", preds_l2)]:
        miss_analysis[model_name] = {}
        for cls in attack_classes:
            sub = test_df[test_df["level3"] == cls]
            sub_preds = preds[test_df["level3"] == cls]

            # Missed is labelled Benign
            miss_mask = sub_preds == "Benign"
            n_total = len(sub)
            n_missed = int(np.sum(miss_mask))
            miss_rate = (n_missed / n_total * 100.0) if n_total > 0 else 0.0

            # Unique bit patterns in missed packets
            if n_missed > 0:
                missed_rows = sub.iloc[miss_mask]
                unique_missed_patterns = len(missed_rows.drop_duplicates(subset=all_features))
            else:
                unique_missed_patterns = 0

            total_unique_patterns = len(sub.drop_duplicates(subset=all_features))

            miss_analysis[model_name][cls] = {
                "total_packets": n_total,
                "missed_packets": n_missed,
                "miss_rate_percent": miss_rate,
                "unique_patterns_in_class": total_unique_patterns,
                "unique_patterns_missed": unique_missed_patterns,
            }

    return miss_analysis


def run_leave_pattern_out_check(train_df: pd.DataFrame, test_df: pd.DataFrame, all_features: List[str]) -> Dict[str, Any]:
    """Tests generalization limit by holding out entire unique bit patterns from training

    and evaluating recall on those held-out patterns.
    """
    logger.info("Running Leave-Pattern-Out Generalization Check for DoS and RPM...")

    results = {}

    for target_class, n_holdout in [("DoS", 5), ("RPM", 3)]:
        logger.info(f"Leave-pattern-out for {target_class}: holding out {n_holdout} unique patterns...")
        # Get all unique patterns for target class in train
        train_class = train_df[train_df["level3"] == target_class]
        unique_patterns_df = train_class.drop_duplicates(subset=all_features).reset_index(drop=True)
        n_patterns_total = len(unique_patterns_df)

        # Select holdout patterns
        np.random.seed(RANDOM_SEED)
        holdout_idx = np.random.choice(n_patterns_total, size=n_holdout, replace=False)
        holdout_patterns_df = unique_patterns_df.iloc[holdout_idx]
        train_patterns_df = unique_patterns_df.drop(index=holdout_idx)

        holdout_tuples = set(map(tuple, holdout_patterns_df[all_features].values))

        # Filter train set: remove holdout patterns from train
        class_tuples_train = [tuple(x) for x in train_class[all_features].values]
        keep_mask = np.array([t not in holdout_tuples for t in class_tuples_train])
        filtered_class_train = train_class.iloc[keep_mask]

        # Combine with benign and other classes to form training set
        other_train = train_df[train_df["level3"] != target_class].sample(n=50000, random_state=RANDOM_SEED)
        lpo_train = pd.concat([other_train, filtered_class_train], ignore_index=True)

        # Train a Random Forest (153 features, 50 trees)
        y_train = (lpo_train["level3"] == target_class).astype(int).values
        rf = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(lpo_train[all_features].values, y_train)

        # Test on the held-out patterns
        # Find test set rows that match the holdout patterns
        test_class = test_df[test_df["level3"] == target_class]
        class_tuples_test = [tuple(x) for x in test_class[all_features].values]
        test_holdout_mask = np.array([t in holdout_tuples for t in class_tuples_test])
        test_seen_mask = ~test_holdout_mask

        n_test_holdout = int(np.sum(test_holdout_mask))
        n_test_seen = int(np.sum(test_seen_mask))

        if n_test_holdout > 0:
            preds_holdout = rf.predict(test_class.iloc[test_holdout_mask][all_features].values)
            recall_holdout = float(np.mean(preds_holdout == 1))
        else:
            recall_holdout = 0.0

        if n_test_seen > 0:
            preds_seen = rf.predict(test_class.iloc[test_seen_mask][all_features].values)
            recall_seen = float(np.mean(preds_seen == 1))
        else:
            recall_seen = 0.0

        results[target_class] = {
            "total_unique_patterns": n_patterns_total,
            "patterns_held_out": n_holdout,
            "test_rows_matching_held_out": n_test_holdout,
            "test_rows_matching_trained": n_test_seen,
            "recall_on_held_out_patterns": recall_holdout,
            "recall_on_seen_patterns": recall_seen,
        }
        logger.info(f"{target_class} LPO: Recall on seen patterns={recall_seen*100:.2f}%, Recall on HELD-OUT patterns={recall_holdout*100:.2f}%")

    return results


def run_all_diagnostics():
    """Executes the full diagnostic suite."""
    test_path = CACHE_DIR / "test.parquet"
    train_path = CACHE_DIR / "train.parquet"
    with open(MODELS_DIR / "feature_sets.json", "r", encoding="utf-8") as f:
        feature_sets = json.load(f)
    with open(MODELS_DIR / "all_features.json", "r", encoding="utf-8") as f:
        all_features = json.load(f)

    test_df = pd.read_parquet(test_path)
    train_df = pd.read_parquet(train_path)

    # 1. Confidence & Entropy
    conf_ent_results = run_confidence_entropy_diagnostics(test_df, feature_sets)

    # 2. Missed Patterns
    miss_results = run_missed_patterns_analysis(test_df, feature_sets, all_features)

    # 3. Leave-Pattern-Out
    lpo_results = run_leave_pattern_out_check(train_df, test_df, all_features)

    full_diagnostics = {
        "confidence_entropy_diagnostics": conf_ent_results,
        "missed_patterns_analysis": miss_results,
        "leave_pattern_out_generalization": lpo_results,
    }

    diag_path = RESULTS_DIR / "diagnostics_report.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(full_diagnostics, f, indent=4)

    # Print Summary Tables
    print("\n" + "=" * 95)
    print("DIAGNOSTIC REPORT 1: CONFIDENCE & ENTROPY DISTRIBUTIONS (L1 and L2)")
    print("=" * 95)
    for model_name, d in conf_ent_results.items():
        print(f"MODEL: {model_name}")
        print(f"  AUC (Uncertainty separating Missed Attacks vs True Benign): {d['auc_separating_c_from_a']['auc_uncertainty_score']:.4f}")
        print(f"  AUC (Entropy separating Missed Attacks vs True Benign):     {d['auc_separating_c_from_a']['auc_entropy_score']:.4f}")
        print("  Confidence Quantiles:")
        print(f"    (a) True Benign    (N={d['counts']['true_benign_a']:,}): Mean={d['confidence_dist']['a_true_benign']['mean']:.4f}, Median={d['confidence_dist']['a_true_benign']['median']:.4f}, Q25={d['confidence_dist']['a_true_benign']['q25']:.4f}, Q75={d['confidence_dist']['a_true_benign']['q75']:.4f}")
        print(f"    (b) Detected Attack(N={d['counts']['attacks_detected_b']:,}): Mean={d['confidence_dist']['b_detected_attack']['mean']:.4f}, Median={d['confidence_dist']['b_detected_attack']['median']:.4f}, Q25={d['confidence_dist']['b_detected_attack']['q25']:.4f}, Q75={d['confidence_dist']['b_detected_attack']['q75']:.4f}")
        print(f"    (c) Missed Attack  (N={d['counts']['attacks_missed_c']:,}): Mean={d['confidence_dist']['c_missed_attack']['mean']:.4f}, Median={d['confidence_dist']['c_missed_attack']['median']:.4f}, Q25={d['confidence_dist']['c_missed_attack']['q25']:.4f}, Q75={d['confidence_dist']['c_missed_attack']['q75']:.4f}")
        print("  Entropy Quantiles:")
        print(f"    (a) True Benign:     Mean={d['entropy_dist']['a_true_benign']['mean']:.4f}, Median={d['entropy_dist']['a_true_benign']['median']:.4f}")
        print(f"    (b) Detected Attack: Mean={d['entropy_dist']['b_detected_attack']['mean']:.4f}, Median={d['entropy_dist']['b_detected_attack']['median']:.4f}")
        print(f"    (c) Missed Attack:   Mean={d['entropy_dist']['c_missed_attack']['mean']:.4f}, Median={d['entropy_dist']['c_missed_attack']['median']:.4f}")
        print("-" * 95)

    print("\n" + "=" * 95)
    print("DIAGNOSTIC REPORT 2: MISSED ATTACK PATTERNS BREAKDOWN")
    print("=" * 95)
    print(f"{'Model':<12} {'Attack Class':<10} {'Total Test':>12} {'Missed Pkts':>12} {'Miss Rate':>12} {'Total Pats':>12} {'Missed Pats':>14}")
    print("-" * 95)
    for model_name, classes in miss_results.items():
        for cls, m in classes.items():
            print(f"{model_name:<12} {cls:<10} {m['total_packets']:>12,d} {m['missed_packets']:>12,d} {m['miss_rate_percent']:>11.2f}% {m['unique_patterns_in_class']:>12d} {m['unique_patterns_missed']:>14d}")
        print("-" * 95)

    print("\n" + "=" * 95)
    print("DIAGNOSTIC REPORT 3: LEAVE-PATTERN-OUT GENERALIZATION CHECK")
    print("=" * 95)
    for target_class, l in lpo_results.items():
        print(f"Target Class: {target_class} (Total Patterns: {l['total_unique_patterns']}, Held-out: {l['patterns_held_out']})")
        print(f"  Test rows matching trained patterns: {l['test_rows_matching_trained']:,} -> Recall: {l['recall_on_seen_patterns']*100:.2f}%")
        print(f"  Test rows matching HELD-OUT patterns: {l['test_rows_matching_held_out']:,} -> Recall: {l['recall_on_held_out_patterns']*100:.2f}%")
        print(f"  Generalization Drop: {(l['recall_on_seen_patterns'] - l['recall_on_held_out_patterns'])*100:.2f}%")
        print("-" * 95)
    print("=" * 95 + "\n")

    return full_diagnostics


if __name__ == "__main__":
    run_all_diagnostics()
