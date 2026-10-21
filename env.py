"""Gym-style Reinforcement Learning Environment for Dynamic Hierarchical IDS Packet Routing.
Implements sequential routing decision process per packet with action masking,
state representation, and multi-objective rewards.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    ACTION_ACCEPT,
    ACTION_FORWARD_CLOUD,
    ACTION_FORWARD_NEAREDGE,
    ACTION_FORWARD_RSU,
    MAX_HOPS,
    NODE_NAMES,
    NODE_TO_IDX,
    NUM_ACTIONS,
    PENALTY_EXCESSIVE_HOPS,
    PENALTY_HOP_COEFF,
    PENALTY_LATENCY_COEFF,
    PENALTY_OVERLOAD,
    PROB_VECTOR_DIM,
    RANDOM_SEED,
    REWARD_ATTACK_L1,
    REWARD_ATTACK_L2,
    REWARD_ATTACK_L3_CLOUD,
    REWARD_BENIGN_CORRECT,
    REWARD_FALSE_NEGATIVE,
    REWARD_WRONG_ATTACK,
    RSU_EARLY_EXIT_BONUS,
)
from src.classifiers import HierarchicalIDSClassifier
from src.network_model import IoVNetworkSimulation


class PacketRoutingEnv:
    """RL Environment where an agent routes incoming CAN packets across vehicle,

    RSU and near-edge classifiers based on uncertainty and network load (cloud only trains/logs/updates).
    """

    def __init__(
        self,
        dataset_df: pd.DataFrame,
        feature_sets: Dict[str, List[str]],
        classifiers: Dict[str, HierarchicalIDSClassifier],
        random_seed: int = RANDOM_SEED,
    ):
        self.dataset_df = dataset_df.reset_index(drop=True)
        self.feature_sets = feature_sets
        self.classifiers = classifiers
        self.rng = np.random.RandomState(random_seed)
        self.net_sim = IoVNetworkSimulation(random_seed=random_seed)

        self.current_row_idx: int = 0
        self.current_packet: Optional[pd.Series] = None
        self.current_node: str = "vehicle"
        self.current_hops: int = 0
        self.elapsed_latency_ms: float = 0.0
        self.current_padded_probs: np.ndarray = np.zeros(PROB_VECTOR_DIM, dtype=np.float32)
        self.current_conf: float = 0.0
        self.current_entropy: float = 0.0
        self.current_pred_label: str = "Benign"

        self.state_dim = PROB_VECTOR_DIM + 1 + 1 + 4 + 4 + 4 + 1  # 21 dimensions
        self.action_dim = NUM_ACTIONS  # 4 actions

        # Pre-extract numpy matrices for blazing fast indexing
        self.feat_matrices = {
            "L1": self.dataset_df[self.feature_sets["L1"]].to_numpy(dtype=np.float32),
            "L2": self.dataset_df[self.feature_sets["L2"]].to_numpy(dtype=np.float32),
            "L3": self.dataset_df[self.feature_sets["L3"]].to_numpy(dtype=np.float32),
        }
        self.level1_arr = self.dataset_df["level1"].to_numpy()
        self.level2_arr = self.dataset_df["level2"].to_numpy()
        self.level3_arr = self.dataset_df["level3"].to_numpy()

    def reset(self, row_idx: Optional[int] = None) -> np.ndarray:
        """Starts a new episode for a single packet starting at the vehicle."""
        if row_idx is None:
            self.current_row_idx = self.rng.randint(0, len(self.dataset_df))
        else:
            self.current_row_idx = row_idx % len(self.dataset_df)

        self.current_packet = self.dataset_df.iloc[self.current_row_idx]
        self.current_node = "vehicle"
        self.current_hops = 0
        self.elapsed_latency_ms = 0.0

        # Step background network queues
        self.net_sim.step(delta_time=0.1)

        # Execute L1 (vehicle) classifier initial prediction
        x_l1 = self.feat_matrices["L1"][self.current_row_idx : self.current_row_idx + 1]
        l1_clf = self.classifiers["L1_vehicle"]
        (
            self.current_pred_label,
            _,
            self.current_conf,
            self.current_entropy,
            self.current_padded_probs,
        ) = l1_clf.predict_full(x_l1)

        # Vehicle base processing latency
        self.elapsed_latency_ms += l1_clf.measured_single_latency_ms

        # Attack-type hint for the router: L1 stays binary, but when it flags an attack a tiny model says whether
        # it looks like DoS or Spoofing. The hint only reshapes the probability vector the RL agent observes.
        hint = self.classifiers.get("L1_type")
        if hint is not None and self.current_pred_label != "Benign":
            p_attack = float(self.current_padded_probs[1:].sum())
            hint_padded = hint.get_padded_from_raw_probs(hint.predict_proba(x_l1))[0]
            shaped = self.current_padded_probs.copy()
            shaped[1:] = p_attack * hint_padded[1:]
            self.current_padded_probs = shaped.astype(np.float32)
            self.elapsed_latency_ms += hint.measured_single_latency_ms

        return self._build_state()

    def _build_state(self) -> np.ndarray:
        """Constructs canonical observation vector.

        Uses ONLY classifier outputs (probs, conf, entropy), node status, and link status.
        Never raw features, never ground-truth labels.
        """
        # 1. Padded probabilities (6)
        probs = self.current_padded_probs
        # 2. Confidence (1) & Entropy (1)
        conf = np.array([self.current_conf], dtype=np.float32)
        ent = np.array([min(1.0, self.current_entropy / 2.58)], dtype=np.float32)  # max entropy for 6 classes is log2(6) ~ 2.58
        # 3. Current node one-hot (4)
        node_onehot = np.zeros(4, dtype=np.float32)
        node_onehot[NODE_TO_IDX[self.current_node]] = 1.0
        # 4. Node queue loads (4)
        node_loads = self.net_sim.get_node_loads()
        # 5. Outgoing link latencies (4)
        link_lats = self.net_sim.get_outgoing_latencies(self.current_node)
        # 6. Hops normalized (1)
        hops_norm = np.array([self.current_hops / float(MAX_HOPS)], dtype=np.float32)

        state = np.concatenate([probs, conf, ent, node_onehot, node_loads, link_lats, hops_norm])
        return state.astype(np.float32)

    def get_action_mask(self) -> np.ndarray:
        """Returns boolean mask of valid actions from current node."""
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        # ACCEPT is always valid
        mask[ACTION_ACCEPT] = True

        if self.current_hops >= MAX_HOPS:
            # Must accept if max hops reached
            return mask

        # The cloud does NOT classify (it trains models, logs and pushes updates), so ACTION_FORWARD_CLOUD
        # is never valid. Routing is vehicle -> {RSU | Near-Edge}; an RSU may escalate to Near-Edge.
        if self.current_node == "vehicle":
            mask[ACTION_FORWARD_RSU] = True
            mask[ACTION_FORWARD_NEAREDGE] = True
        elif self.current_node == "rsu":
            mask[ACTION_FORWARD_NEAREDGE] = True
        # nearedge: terminal (only ACCEPT)

        return mask

    # ------------------------------------------------------------------
    # Depth-neutral reward helpers
    # ------------------------------------------------------------------
    _SPOOF_TYPES = {"Gas", "RPM", "Speed", "SW"}

    @staticmethod
    def _is_correct(node: str, pred: str, true_l1: str, true_l2: str, true_l3: str) -> bool:
        """Correctness at the granularity the node can actually output.
        vehicle -> L1 label, rsu -> L2 label, nearedge/cloud -> L3 label."""
        if node == "vehicle":
            return pred == true_l1
        if node == "rsu":
            return pred == true_l2
        return pred == true_l3

    def _terminal_reward(self, node: str, pred: str, true_l1: str, true_l2: str, true_l3: str) -> float:
        """Reward for finishing at `node` with prediction `pred`.

        KEY FIX: a fully-resolved answer earns the SAME top reward wherever it is produced
        (e.g. DoS accepted at the RSU is already complete). Previously the top reward was only
        paid at near-edge/cloud, which taught the agent to push every attack to the deepest,
        slowest node and made RL slower than the fixed hierarchy.
        """
        is_attack = (true_l1 == "Attack")
        if not is_attack:
            return REWARD_BENIGN_CORRECT if pred == "Benign" else REWARD_WRONG_ATTACK
        if pred == "Benign":
            return REWARD_FALSE_NEGATIVE
        if pred == true_l3:                       # fully resolved (DoS at RSU, any subtype at near-edge)
            if node == "rsu":                     # resolved at the lowest sufficient tier -> early-exit bonus
                return REWARD_ATTACK_L3_CLOUD + RSU_EARLY_EXIT_BONUS
            return REWARD_ATTACK_L3_CLOUD
        if node == "vehicle":                     # L1 only knows "Attack"
            return REWARD_ATTACK_L1
        if node == "rsu":                         # coarse family right (Spoofing) but subtype unknown
            return REWARD_ATTACK_L2 if pred == true_l2 else REWARD_ATTACK_L1
        if pred in self._SPOOF_TYPES and true_l3 in self._SPOOF_TYPES:
            return REWARD_ATTACK_L2               # attack found, wrong spoof subtype
        return REWARD_WRONG_ATTACK

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Executes selected routing action, advances packet, and returns (s', r, done, info)."""
        valid_mask = self.get_action_mask()
        if not valid_mask[action]:
            # Invalid action penalty
            reward = -5.0
            done = True
            info = {"error": "invalid_action", "hops": self.current_hops, "latency": self.elapsed_latency_ms}
            return self._build_state(), reward, done, info

        true_l1 = self.level1_arr[self.current_row_idx]
        true_l2 = self.level2_arr[self.current_row_idx]
        true_l3 = self.level3_arr[self.current_row_idx]
        is_attack = (true_l1 == "Attack")

        # -----------------------------------------------
        # Action 0: ACCEPT current prediction & Terminate
        # -----------------------------------------------
        if action == ACTION_ACCEPT:
            done = True
            reward = self._terminal_reward(
                self.current_node, self.current_pred_label, true_l1, true_l2, true_l3
            )

            # Latency and hop costs
            reward -= (PENALTY_LATENCY_COEFF * self.elapsed_latency_ms)
            reward -= (PENALTY_HOP_COEFF * self.current_hops)

            info = {
                "final_node": self.current_node,
                "final_pred": self.current_pred_label,
                "true_l3": true_l3,
                "is_attack": is_attack,
                "hops": self.current_hops,
                "latency_ms": self.elapsed_latency_ms,
                "correct": self._is_correct(self.current_node, self.current_pred_label, true_l1, true_l2, true_l3),
                "false_negative": (is_attack and self.current_pred_label == "Benign"),
            }
            return self._build_state(), float(reward), done, info

        # -----------------------------------------------
        # Actions 1, 2, 3: FORWARD to Higher Node
        # -----------------------------------------------
        self.current_hops += 1
        if action == ACTION_FORWARD_RSU:
            next_node = "rsu"
            clf = self.classifiers["L2_rsu"]
            feats_key = "L2"
        elif action == ACTION_FORWARD_NEAREDGE:
            next_node = "nearedge"
            clf = self.classifiers["L3_nearedge"]
            feats_key = "L3"
        else:
            raise ValueError(f"Action {action} is not a valid classification target (cloud does not classify).")

        # PENALTY: Forwarding a Benign-classified packet wastes resources
        # If L1 says Benign with high confidence, agent should ACCEPT, not forward
        step_reward = 0.0
        if self.current_pred_label == "Benign" and self.current_conf > 0.7:
            step_reward -= 8.0  # Strong penalty for forwarding benign packets

        # Link latency and queueing
        link_lat = self.net_sim.get_link_latency_ms(self.current_node, next_node)
        self.elapsed_latency_ms += link_lat
        self.elapsed_latency_ms += clf.measured_single_latency_ms

        # Check overload penalty on target node
        if self.net_sim.nodes[next_node].load_ratio > 0.80:
            step_reward -= PENALTY_OVERLOAD

        # Enqueue packet at target node
        self.net_sim.nodes[next_node].enqueue(1)
        self.current_node = next_node

        # Run inference at new node
        x_sub = self.feat_matrices[feats_key][self.current_row_idx : self.current_row_idx + 1]
        (
            self.current_pred_label,
            _,
            self.current_conf,
            self.current_entropy,
            self.current_padded_probs,
        ) = clf.predict_full(x_sub)

        done = (self.current_node == "nearedge") or (self.current_hops >= MAX_HOPS)
        if done:
            # Auto-accept at the terminal Near-Edge node / max hops
            step_reward += self._terminal_reward(
                self.current_node, self.current_pred_label, true_l1, true_l2, true_l3
            )

            step_reward -= (PENALTY_LATENCY_COEFF * self.elapsed_latency_ms)
            step_reward -= (PENALTY_HOP_COEFF * self.current_hops)

        info = {
            "current_node": self.current_node,
            "pred_label": self.current_pred_label,
            "hops": self.current_hops,
            "latency_ms": self.elapsed_latency_ms,
        }
        if done:
            info.update({
                "final_node": self.current_node,
                "final_pred": self.current_pred_label,
                "correct": self._is_correct(self.current_node, self.current_pred_label, true_l1, true_l2, true_l3),
            })
        return self._build_state(), float(step_reward), done, info


_TIER_FEATS = {"L1_vehicle": "L1", "L2_rsu": "L2", "L3_nearedge": "L3", "L1_type": "L1"}


def prepare_classifiers_for_inference(classifiers, feature_sets, df, n_trials: int = 100):
    """Call this in train_rl.py, main.py and evaluate.py so training and testing see the SAME latencies.

    Single-row predict with n_jobs=-1 spawns a thread pool per call (tens of ms), which inflated every
    latency figure. Force n_jobs=1 and re-profile the real per-instance inference time.
    """
    import logging
    log = logging.getLogger(__name__)
    out = {}
    for name, fkey in _TIER_FEATS.items():
        if name not in classifiers:
            continue
        clf = classifiers[name]
        try:
            if "n_jobs" in clf.model.get_params():
                clf.model.set_params(n_jobs=1)
            X = df[feature_sets[fkey]].head(200).to_numpy(dtype=np.float32)
            out[name] = clf.profile_latency(X, n_trials=n_trials)
            log.info(f"Profiled {name}: {out[name]:.3f} ms / instance (n_jobs=1)")
        except Exception as ex:
            log.warning(f"Could not re-profile {name}: {ex}")
    return out
