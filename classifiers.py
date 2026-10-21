"""Classifier wrappers and latency profiling module.

Provides a unified HierarchicalIDSClassifier that supports multiple backends:
  - L1 (Vehicle)  : BernoulliNB  (lightweight, paper-aligned)
  - L2 (RSU)      : RandomForest (paper-aligned)
  - L3 (NearEdge) : RandomForest (paper-aligned)
  - Cloud         : XGBoost      (paper-aligned)

All classifiers expose the same interface:
  predict(), predict_proba(), predict_full(), get_padded_proba()
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

from config import (
    CLOUD_CLASSES,
    L1_CLASSES,
    L2_CLASSES,
    L3_CLASSES,
    MODELS_DIR,
    PROB_VECTOR_DIM,
    RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _build_model(backend: str, n_estimators: int, max_depth: Optional[int], random_state: int):
    """Factory: returns the correct sklearn/xgboost model object."""
    if backend == "naive_bayes":
        return BernoulliNB(alpha=1.0)
    elif backend == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth if max_depth else 6,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
        except ImportError:
            logger.warning("XGBoost not installed — falling back to RandomForest for Cloud.")
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
    else:  # default: random_forest
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )


class HierarchicalIDSClassifier:
    """Unified IDS classifier supporting multiple ML backends.

    Backend choices:
      "naive_bayes"   → sklearn BernoulliNB  (L1 / Vehicle)
      "random_forest" → sklearn RF           (L2 RSU, L3 NearEdge)
      "xgboost"       → XGBClassifier        (Cloud)

    All backends expose identical interfaces:
      .classes_  → sorted array of class names
      .predict() → label array
      .predict_proba() → probability matrix
    """

    def __init__(
        self,
        level_name: str,
        feature_names: List[str],
        classes: List[str],
        backend: str = "random_forest",
        n_estimators: int = 50,
        max_depth: Optional[int] = None,
        random_state: int = RANDOM_SEED,
    ):
        self.level_name = level_name
        self.feature_names = feature_names
        self.classes = classes
        self.backend = backend
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.model = _build_model(backend, n_estimators, max_depth, random_state)
        self.is_fitted = False
        self.measured_single_latency_ms: float = 0.0
        # Set after fit: sorted class array from the underlying model
        self._model_classes: List[str] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HierarchicalIDSClassifier":
        """Fits the underlying model."""
        if self.backend == "xgboost":
            # XGBoost needs integer labels internally
            self._label_to_int = {c: i for i, c in enumerate(sorted(set(y)))}
            self._int_to_label = {i: c for c, i in self._label_to_int.items()}
            y_int = np.array([self._label_to_int[c] for c in y])
            self.model.fit(X, y_int)
            # Store sorted class names (XGBClassifier.classes_ is read-only)
            self._model_classes = sorted(self._label_to_int.keys())
        else:
            self.model.fit(X, y)
            self._model_classes = list(self.model.classes_)

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted string labels."""
        if self.backend == "xgboost":
            raw = self.model.predict(X)
            return np.array([self._int_to_label[int(v)] for v in raw])
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class-probability matrix (N × n_classes)."""
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    # Fixed-dim probability padding
    # ------------------------------------------------------------------
    def get_padded_from_raw_probs(self, probs: np.ndarray) -> np.ndarray:
        """Maps model-output probs to canonical 6-dim vector [Benign,DoS,Gas,RPM,Speed,SW]."""
        N = probs.shape[0]
        padded = np.zeros((N, PROB_VECTOR_DIM), dtype=np.float32)
        model_classes = self._model_classes or self.classes
        for i, cls_name in enumerate(model_classes):
            if cls_name == "Benign":
                padded[:, 0] = probs[:, i]
            elif cls_name == "Attack":
                # Binary L1: spread attack probability equally over slots 1-5
                padded[:, 1:] = probs[:, i : i + 1] / 5.0
            elif cls_name == "DoS":
                padded[:, 1] = probs[:, i]
            elif cls_name == "Spoofing":
                padded[:, 2:6] = probs[:, i : i + 1] / 4.0
            elif cls_name == "Gas":
                padded[:, 2] = probs[:, i]
            elif cls_name == "RPM":
                padded[:, 3] = probs[:, i]
            elif cls_name == "Speed":
                padded[:, 4] = probs[:, i]
            elif cls_name == "SW":
                padded[:, 5] = probs[:, i]
        return padded

    def get_padded_proba(self, X: np.ndarray) -> np.ndarray:
        return self.get_padded_from_raw_probs(self.predict_proba(X))

    def predict_full(self, X: np.ndarray) -> Tuple[str, np.ndarray, float, float, np.ndarray]:
        """Single forward pass → (pred_label, raw_probs, max_conf, entropy, padded_probs)."""
        raw_probs = self.predict_proba(X)
        probs_1d = raw_probs[0]
        max_idx = int(np.argmax(probs_1d))
        model_classes = self._model_classes or self.classes
        pred_label = model_classes[max_idx]
        conf = float(probs_1d[max_idx])
        eps = 1e-12
        entropy = float(-np.sum(probs_1d * np.log2(probs_1d + eps)))
        padded = self.get_padded_from_raw_probs(raw_probs)[0]
        return pred_label, probs_1d, conf, entropy, padded

    # ------------------------------------------------------------------
    # Latency profiling
    # ------------------------------------------------------------------
    def profile_latency(self, X_sample: np.ndarray, n_trials: int = 100) -> float:
        """Profiles single-instance inference latency in milliseconds."""
        single_row = X_sample[0:1]
        for _ in range(10):
            self.predict(single_row)
        start = time.perf_counter()
        for _ in range(n_trials):
            self.predict(single_row)
        avg_ms = (time.perf_counter() - start) / n_trials * 1000.0
        self.measured_single_latency_ms = avg_ms
        return avg_ms
