"""Evaluation policies module.
Implements Flat, Fixed Hierarchy (base paper), and RL Dynamic Routing policies.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    ACTION_ACCEPT,
    ACTION_FORWARD_CLOUD,
    ACTION_FORWARD_NEAREDGE,
    ACTION_FORWARD_RSU,
    L3_CLASSES,
)
from src.env import PacketRoutingEnv
from src.rl_agent import DQNAgent


class PolicyEvaluator:
    """Evaluates Flat, Fixed, and RL policies on test packets."""

    def __init__(self, env: PacketRoutingEnv, rl_agent: Optional[DQNAgent] = None):
        self.env = env
        self.rl_agent = rl_agent

    def run_single_packet(self, policy_name: str, row_idx: int) -> Dict[str, Any]:
        """Routes a single packet through the network according to policy_name.

        Returns detailed packet-level trace.
        """
        state = self.env.reset(row_idx=row_idx)
        done = False
        route_nodes = [self.env.current_node]
        actions_taken = []
        hop_predictions = []
        hop_confidences = []

        while not done:
            hop_predictions.append(self.env.current_pred_label)
            hop_confidences.append(float(self.env.current_conf))

            # Policy Action Selection
            if policy_name == "Flat":
                # Flat (no hierarchy): the vehicle does no filtering and ships EVERY packet to the
                # single full 6-class classifier at the Near-Edge node.
                if self.env.current_node == "vehicle":
                    action = ACTION_FORWARD_NEAREDGE
                else:
                    action = ACTION_ACCEPT

            elif policy_name == "Fixed":
                # Fixed Hierarchy (Base Paper Uddin et al. 2025):
                # Vehicle: if Benign -> Accept; if Attack -> Forward to RSU
                # RSU: if DoS -> Accept; if Spoofing -> Forward to NearEdge
                # NearEdge: Accept
                if self.env.current_node == "vehicle":
                    if self.env.current_pred_label == "Benign":
                        action = ACTION_ACCEPT
                    else:
                        action = ACTION_FORWARD_RSU
                elif self.env.current_node == "rsu":
                    if self.env.current_pred_label in ("DoS", "Benign"):
                        action = ACTION_ACCEPT
                    else:
                        action = ACTION_FORWARD_NEAREDGE
                else:
                    action = ACTION_ACCEPT

            elif policy_name == "RL":
                if self.rl_agent is None:
                    raise ValueError("RL agent must be provided for RL policy evaluation.")
                mask = self.env.get_action_mask()
                action = self.rl_agent.select_action(state, mask, greedy=True)
            else:
                action = ACTION_ACCEPT

            actions_taken.append(action)
            next_state, reward, done, info = self.env.step(action)
            state = next_state
            if self.env.current_node != route_nodes[-1]:
                route_nodes.append(self.env.current_node)

        true_l1 = self.env.level1_arr[self.env.current_row_idx]
        true_l2 = self.env.level2_arr[self.env.current_row_idx]
        true_l3 = self.env.level3_arr[self.env.current_row_idx]
        is_attack = (true_l1 == "Attack")

        final_pred = info.get("final_pred", self.env.current_pred_label)
        final_node = info.get("final_node", self.env.current_node)
        total_lat = info.get("latency_ms", self.env.elapsed_latency_ms)
        total_hops = info.get("hops", self.env.current_hops)

        # Correctness check at final level
        # Correct at the granularity the final node can output (vehicle=L1, rsu=L2, edge/cloud=L3)
        correct = int(PacketRoutingEnv._is_correct(final_node, final_pred, true_l1, true_l2, true_l3))

        is_fn = int(is_attack and final_pred == "Benign")
        is_fp = int((not is_attack) and final_pred != "Benign")

        # Bandwidth estimation (CAN payload 8 bytes + IP/V2X header ~ 64 bytes per hop)
        bytes_transmitted = total_hops * 64

        return {
            "policy": policy_name,
            "row_idx": row_idx,
            "true_l1": true_l1,
            "true_l2": true_l2,
            "true_l3": true_l3,
            "is_attack": int(is_attack),
            "final_pred": final_pred,
            "final_node": final_node,
            "route_taken": " > ".join(route_nodes),
            "hops": total_hops,
            "latency_ms": total_lat,
            "correct": correct,
            "false_negative": is_fn,
            "false_positive": is_fp,
            "bandwidth_bytes": bytes_transmitted,
            "reward": float(reward),
        }
