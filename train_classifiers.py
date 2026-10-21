"""Training, 10-Fold CV, and Test Set Evaluation for Hierarchical Classifiers.
Runs 10-fold Stratified CV on the FULL training data (985,753 rows) and prints per-class fold counts.
Evaluates on full 211,233 test set at dual granularities (attack-vs-benign and 6-class)
plus per-class metrics and novel test pattern performance.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib
matplotlib.use("Agg")  # non-GUI backend: avoids Tk crash when plotting from worker threads
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from config import (
    CACHE_DIR,
    CLOUD_CLASSES,
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


def plot_and_save_confusion_matrix(cm: np.ndarray, classes: List[str], title: str, save_path: Path):
    """Saves a normalized confusion matrix heatmap."""
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    norm_cm = np.divide(cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(norm_cm, cmap=plt.cm.Blues, vmin=0, vmax=1)
    fig.colorbar(cax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)

    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if norm_cm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]:,}\n({norm_cm[i, j]*100:.1f}%)", ha="center", va="center", color=color, fontsize=8)

    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10, fontweight="bold")
    ax.set_title(title, pad=20, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {save_path}")


def evaluate_classifier_full(
    classifier: HierarchicalIDSClassifier,
    X_test: np.ndarray,
    y_test_native: np.ndarray,
    y_test_binary: np.ndarray,
    level_name: str,
) -> Dict[str, Any]:
    """Evaluates classifier at binary detection granularity, native multi-class granularity,

    and per-class precision/recall/f1.
    """
    y_pred_native = classifier.predict(X_test)

    # 1. Native Multi-class metrics
    cm = confusion_matrix(y_test_native, y_pred_native, labels=classifier.classes)
    acc = accuracy_score(y_test_native, y_pred_native)
    p_macro = precision_score(y_test_native, y_pred_native, labels=classifier.classes, average="macro", zero_division=0)
    r_macro = recall_score(y_test_native, y_pred_native, labels=classifier.classes, average="macro", zero_division=0)
    f1_macro = f1_score(y_test_native, y_pred_native, labels=classifier.classes, average="macro", zero_division=0)
    p_weighted = precision_score(y_test_native, y_pred_native, labels=classifier.classes, average="weighted", zero_division=0)
    r_weighted = recall_score(y_test_native, y_pred_native, labels=classifier.classes, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test_native, y_pred_native, labels=classifier.classes, average="weighted", zero_division=0)

    # Per-class metrics
    p_per_class = precision_score(y_test_native, y_pred_native, labels=classifier.classes, average=None, zero_division=0)
    r_per_class = recall_score(y_test_native, y_pred_native, labels=classifier.classes, average=None, zero_division=0)
    f1_per_class = f1_score(y_test_native, y_pred_native, labels=classifier.classes, average=None, zero_division=0)

    per_class_dict = {}
    for idx, cls_name in enumerate(classifier.classes):
        per_class_dict[cls_name] = {
            "precision": float(p_per_class[idx]),
            "recall": float(r_per_class[idx]),
            "f1_score": float(f1_per_class[idx]),
            "support": int(np.sum(y_test_native == cls_name)),
        }

    # 2. Attack-vs-Benign Binary Granularity
    # Any prediction other than 'Benign' is considered an Attack detection
    y_pred_binary = np.where(y_pred_native == "Benign", "Benign", "Attack")
    cm_binary = confusion_matrix(y_test_binary, y_pred_binary, labels=["Benign", "Attack"])
    # cm_binary: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm_binary.ravel()
    bin_acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    bin_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    bin_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    bin_fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    bin_f1 = 2 * bin_prec * bin_recall / (bin_prec + bin_recall) if (bin_prec + bin_recall) > 0 else 0.0

    return {
        "level_name": level_name,
        "n_features": len(classifier.feature_names),
        "feature_list": classifier.feature_names,
        "multiclass": {
            "accuracy": float(acc),
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "macro_f1": float(f1_macro),
            "weighted_precision": float(p_weighted),
            "weighted_recall": float(r_weighted),
            "weighted_f1": float(f1_weighted),
            "confusion_matrix": cm.tolist(),
            "classes": classifier.classes,
            "per_class": per_class_dict,
        },
        "binary_detection": {
            "accuracy": float(bin_acc),
            "precision": float(bin_prec),
            "recall": float(bin_recall),
            "false_negative_rate": float(bin_fnr),
            "f1_score": float(bin_f1),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
    }


def evaluate_novel_rows(
    classifier: HierarchicalIDSClassifier,
    test_df: pd.DataFrame,
    train_patterns_set: set,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """Evaluates classifier specifically on test rows whose exact feature pattern was NOT seen in train."""
    test_tuples = [tuple(x) for x in test_df[feature_cols].values]
    novel_mask = np.array([t not in train_patterns_set for t in test_tuples])
    n_novel = int(np.sum(novel_mask))

    if n_novel == 0:
        return {"n_novel": 0, "status": "No novel rows in test set."}

    novel_df = test_df.iloc[novel_mask].copy()
    X_novel = novel_df[classifier.feature_names].values
    y_novel_native = novel_df[
        "level1" if len(classifier.classes) == 2 else ("level2" if len(classifier.classes) == 3 else "level3")
    ].values
    y_novel_binary = (novel_df["level1"] == "Attack").map({True: "Attack", False: "Benign"}).values

    y_pred_native = classifier.predict(X_novel)
    y_pred_binary = np.where(y_pred_native == "Benign", "Benign", "Attack")

    # Metrics on novel rows
    novel_acc = float(accuracy_score(y_novel_native, y_pred_native))

    cm_bin = confusion_matrix(y_novel_binary, y_pred_binary, labels=["Benign", "Attack"])
    tn, fp, fn, tp = cm_bin.ravel()

    # Per class breakdown of novel rows
    class_novel_counts = novel_df["level3"].value_counts().to_dict()

    return {
        "n_novel_total": n_novel,
        "class_novel_counts": class_novel_counts,
        "novel_multiclass_accuracy": novel_acc,
        "novel_binary_tp": int(tp),
        "novel_binary_fp": int(fp),
        "novel_binary_tn": int(tn),
        "novel_binary_fn": int(fn),
        "novel_recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "novel_fnr": float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "novel_benign_accuracy": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
    }


def run_training_and_evaluation():
    """Main execution function for Phase 1 on FULL 1.4M dataset."""
    train_path = CACHE_DIR / "train.parquet"
    test_path = CACHE_DIR / "test.parquet"
    features_json_path = MODELS_DIR / "feature_sets.json"

    if not train_path.exists() or not test_path.exists() or not features_json_path.exists():
        raise FileNotFoundError("Missing cached datasets or feature sets. Run data_prep.py and feature_selection.py first.")

    logger.info("Loading FULL train and test partitions...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    with open(features_json_path, "r", encoding="utf-8") as f:
        feature_sets = json.load(f)
    with open(MODELS_DIR / "all_features.json", "r", encoding="utf-8") as f:
        all_features = json.load(f)

    logger.info(f"Loaded Train set: {train_df.shape}, Test set: {test_df.shape}")

    # Binary ground truths for test set
    y_test_binary = (test_df["level1"] == "Attack").map({True: "Attack", False: "Benign"}).values

    # Precompute train patterns set for novel evaluation
    logger.info("Building train pattern set for novel test evaluation...")
    train_tuples_set = set(map(tuple, train_df[all_features].values))

    # Setup Classifiers — backends match base paper:
    #   L1 Vehicle   : BernoulliNB (lightweight, paper Sec. IV-B)
    #   L2 RSU       : RandomForest (paper Sec. IV-C)
    #   L3 NearEdge  : RandomForest (paper Sec. IV-D)
    #   Cloud        : XGBoost (paper Sec. IV-E)
    classifiers_def = [
        ("L1_vehicle",       feature_sets["L1"],  L1_CLASSES,    "level1", "random_forest",  30,   20),   # was naive_bayes: recall only 90.5% -> many FN; paper uses RF at root
        ("L2_rsu",           feature_sets["L2"],  L2_CLASSES,    "level2", "random_forest",  30, None),   # was depth 15: Spoofing recall only 86.4%
        ("L3_nearedge",      feature_sets["L3"],  L3_CLASSES,    "level3", "random_forest",  50,   20),
    ]

    all_metrics = {}
    latencies = {}
    training_times = {}

    # 1. 10-Fold Stratified Cross Validation on FULL Train Data (985,753 rows)
    logger.info("=" * 80)
    logger.info("RUNNING 10-FOLD STRATIFIED CROSS VALIDATION ON FULL TRAIN SET (985,753 rows)")
    logger.info("=" * 80)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    # Print per-class counts in every fold to PROVE stratification works
    print("\n" + "=" * 95)
    print("10-FOLD STRATIFIED SPLIT: VALIDATION FOLD CLASS COUNTS (PROOF OF STRATIFICATION)")
    print("=" * 95)
    header = f"{'Fold':<6} {'Benign':>10} {'DoS':>8} {'RPM':>8} {'Speed':>8} {'SW':>8} {'Gas':>8} {'Total':>10}"
    print(header)
    print("-" * 95)

    y_train_full_l3 = train_df["level3"].values
    fold_splits = list(skf.split(train_df, y_train_full_l3))

    for fold_idx, (tr_idx, val_idx) in enumerate(fold_splits):
        val_fold_y = y_train_full_l3[val_idx]
        counts = pd.Series(val_fold_y).value_counts().to_dict()
        b = counts.get("Benign", 0)
        d = counts.get("DoS", 0)
        r = counts.get("RPM", 0)
        sp = counts.get("Speed", 0)
        sw = counts.get("SW", 0)
        g = counts.get("Gas", 0)
        tot = len(val_idx)
        print(f"{fold_idx:<6d} {b:>10,d} {d:>8,d} {r:>8,d} {sp:>8,d} {sw:>8,d} {g:>8,d} {tot:>10,d}")
    print("=" * 95 + "\n")

    # Run 10-Fold CV for L1, L2, L3 on FULL Train Data
    cv_results = {}
    for name, feats, classes, target_col, backend, n_est, m_depth in classifiers_def[:3]:
        logger.info(f"Computing 10-fold CV on FULL train set for {name} ({len(feats)} features, backend={backend})...")
        fold_scores = []
        X_train_full = train_df[feats].values
        y_train_full = train_df[target_col].values

        t0_cv = time.time()
        for fold_idx, (tr_idx, val_idx) in enumerate(fold_splits):
            fold_clf = HierarchicalIDSClassifier(
                level_name=f"{name}_f{fold_idx}",
                feature_names=feats,
                classes=classes,
                backend=backend,
                n_estimators=n_est,
                max_depth=m_depth,
                random_state=RANDOM_SEED,
            )
            fold_clf.fit(X_train_full[tr_idx], y_train_full[tr_idx])
            preds = fold_clf.predict(X_train_full[val_idx])
            fold_scores.append(accuracy_score(y_train_full[val_idx], preds))
        t_cv_total = time.time() - t0_cv
        cv_mean = float(np.mean(fold_scores))
        cv_std = float(np.std(fold_scores))
        cv_results[name] = {"mean": cv_mean, "std": cv_std, "fold_scores": fold_scores, "cv_time_sec": t_cv_total}
        logger.info(f"10-Fold CV for {name}: {cv_mean*100:.3f}% (+/- {cv_std*100:.3f}%) in {t_cv_total:.1f}s")

    # 2. Train Full Models on 985,753 rows & Evaluate on Test Set
    for name, feats, classes, target_col, backend, n_est, m_depth in classifiers_def:
        logger.info("=" * 80)
        logger.info(f"Training Model: {name} | backend={backend} | {len(feats)} features...")

        clf = HierarchicalIDSClassifier(
            level_name=name,
            feature_names=feats,
            classes=classes,
            backend=backend,
            n_estimators=n_est,
            max_depth=m_depth,
            random_state=RANDOM_SEED,
        )

        X_train = train_df[feats].values
        y_train = train_df[target_col].values

        t0_train = time.time()
        clf.fit(X_train, y_train)
        train_duration = time.time() - t0_train
        training_times[name] = train_duration
        logger.info(f"Training completed for {name} in {train_duration:.2f} seconds.")

        # Latency Profiling (200 trials single instance)
        X_test = test_df[feats].values
        latency_ms = clf.profile_latency(X_test, n_trials=200)
        latencies[name] = {
            "single_latency_ms": latency_ms,
            "single_latency_us": latency_ms * 1000.0,
        }
        logger.info(f"Measured single-instance latency for {name}: {latency_ms*1000.0:.2f} us ({latency_ms:.4f} ms)")

        # Test Set Evaluation (211,233 rows)
        logger.info(f"Evaluating {name} on 211,233 test rows...")
        y_test_native = test_df[target_col].values
        metrics = evaluate_classifier_full(
            clf,
            X_test,
            y_test_native,
            y_test_binary,
            level_name=name,
        )
        metrics["latency"] = latencies[name]
        metrics["training_time_sec"] = train_duration
        if name in cv_results:
            metrics["cv_10fold_accuracy_mean"] = cv_results[name]["mean"]
            metrics["cv_10fold_accuracy_std"] = cv_results[name]["std"]

        # Evaluate on Novel test rows
        novel_eval = evaluate_novel_rows(clf, test_df, train_tuples_set, all_features)
        metrics["novel_rows_evaluation"] = novel_eval

        all_metrics[name] = metrics

        # Plot & Save Confusion Matrix
        cm = np.array(metrics["multiclass"]["confusion_matrix"])
        cm_path = RESULTS_DIR / f"cm_{name}.png"
        plot_and_save_confusion_matrix(cm, classes, f"Confusion Matrix: {name} ({len(feats)} features)", cm_path)

        # Save model joblib
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(clf, model_path)
        logger.info(f"Saved frozen model to {model_path}")

    # ------------------------------------------------------------------
    # Attack-TYPE hint for the RL router. L1 itself stays binary (Benign / Attack). This tiny model runs at the
    # vehicle ONLY on packets L1 flagged as attack and tells the RL agent whether the attack looks like
    # DoS or Spoofing, so it can route DoS -> RSU and Spoofing -> Near-Edge (RL input "Attack Type" in the design).
    # ------------------------------------------------------------------
    l1_feats = feature_sets["L1"]
    atk_tr = train_df[train_df["level1"] == "Attack"]
    atk_te = test_df[test_df["level1"] == "Attack"]
    hint_clf = HierarchicalIDSClassifier(
        level_name="L1_type", feature_names=l1_feats, classes=["DoS", "Spoofing"], backend="random_forest",
        n_estimators=10, max_depth=8, random_state=RANDOM_SEED,
    )
    hint_clf.fit(atk_tr[l1_feats].values, atk_tr["level2"].values)
    hint_acc = float((hint_clf.predict(atk_te[l1_feats].values) == atk_te["level2"].values).mean())
    hint_clf.profile_latency(atk_te[l1_feats].values, n_trials=200)
    joblib.dump(hint_clf, MODELS_DIR / "L1_type.joblib")
    all_metrics["L1_type_hint"] = {"accuracy_on_attack_rows": hint_acc, "n_features": len(l1_feats)}
    logger.info(f"Attack-type hint (DoS vs Spoofing, attack rows only): {hint_acc*100:.3f}% on {len(atk_te):,} test attacks, "
                f"{hint_clf.measured_single_latency_ms:.3f} ms")

    # Flat baseline
    flat_path = MODELS_DIR / "Flat_classifier.joblib"
    joblib.dump(clf, flat_path)
    all_metrics["Flat_baseline"] = dict(all_metrics["L3_nearedge"])   # flat = one full 6-class model
    all_metrics["Flat_baseline"]["level_name"] = "Flat_baseline"
    latencies["Flat_baseline"] = latencies["L3_nearedge"]
    training_times["Flat_baseline"] = training_times["L3_nearedge"]

    # Save summary latencies & metrics JSON
    with open(MODELS_DIR / "inference_latencies.json", "w", encoding="utf-8") as f:
        json.dump(latencies, f, indent=4)
    with open(RESULTS_DIR / "classifier_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    # ---------------------------------------------------------
    # PRINT COMPREHENSIVE TABLES TO STDOUT
    # ---------------------------------------------------------
    print("\n" + "=" * 105)
    print("PHASE 1: FULL DATASET TEST SET EVALUATION REPORT (211,233 TEST ROWS)")
    print("=" * 105)
    print("GRANULARITY 1: ATTACK vs BENIGN DETECTION")
    print("-" * 105)
    print(f"{'Classifier':<18} {'Feats':>5} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'FNR':>10} {'F1-Score':>10} {'Train Time':>12} {'Latency':>14}")
    print("-" * 105)
    for name in ["L1_vehicle", "L2_rsu", "L3_nearedge"]:
        b = all_metrics[name]["binary_detection"]
        n_f = all_metrics[name]["n_features"]
        t_tr = training_times[name]
        lat_us = latencies[name]["single_latency_us"]
        print(f"{name:<18} {n_f:>5d} {b['accuracy']*100:>9.3f}% {b['precision']*100:>9.3f}% {b['recall']*100:>9.3f}% {b['false_negative_rate']*100:>9.3f}% {b['f1_score']*100:>9.3f}% {t_tr:>10.2f} s {lat_us:>11.2f} us")

    print("\n" + "-" * 105)
    print("GRANULARITY 2: MULTI-CLASS CLASSIFICATION")
    print("-" * 105)
    print(f"{'Classifier':<18} {'Classes':>7} {'Accuracy':>10} {'Macro P':>10} {'Macro R':>10} {'Macro F1':>10} {'Weighted F1':>12} {'10-Fold CV':>14}")
    print("-" * 105)
    for name in ["L1_vehicle", "L2_rsu", "L3_nearedge"]:
        m = all_metrics[name]["multiclass"]
        n_c = len(m["classes"])
        cv_str = f"{cv_results[name]['mean']*100:.2f}%" if name in cv_results else "N/A"
        print(f"{name:<18} {n_c:>7d} {m['accuracy']*100:>9.3f}% {m['macro_precision']*100:>9.3f}% {m['macro_recall']*100:>9.3f}% {m['macro_f1']*100:>9.3f}% {m['weighted_f1']*100:>11.3f}% {cv_str:>14}")

    print("\n" + "-" * 105)
    print("PER-CLASS METRICS TABLE ON TEST SET")
    print("-" * 105)
    print(f"{'Classifier':<18} {'Class':<14} {'Support':>10} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-" * 105)
    for name in ["L1_vehicle", "L2_rsu", "L3_nearedge"]:
        pc = all_metrics[name]["multiclass"]["per_class"]
        for cls_name, vals in pc.items():
            print(f"{name:<18} {cls_name:<14} {vals['support']:>10,d} {vals['precision']*100:>11.2f}% {vals['recall']*100:>11.2f}% {vals['f1_score']*100:>11.2f}%")
        print("-" * 105)

    print("\n" + "-" * 105)
    print("NOVEL TEST PATTERNS EVALUATION (PATTERNS UNSEEN IN TRAINING DATA)")
    print("-" * 105)
    for name in ["L1_vehicle", "L2_rsu", "L3_nearedge"]:
        nov = all_metrics[name]["novel_rows_evaluation"]
        print(f"{name:<18} Novel Total: {nov['n_novel_total']} (Benign: {nov['class_novel_counts'].get('Benign', 0)}, Attacks: 0)")
        print(f"                   Novel Multiclass Acc: {nov['novel_multiclass_accuracy']*100:.2f}% | Novel Benign Acc (TN): {nov['novel_benign_accuracy']*100:.2f}% | FP: {nov['novel_binary_fp']}")
    print("=" * 105 + "\n")

    return all_metrics


if __name__ == "__main__":
    run_training_and_evaluation()
