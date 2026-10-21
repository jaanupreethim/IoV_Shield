"""Cloud tier of the hierarchical IoV IDS.

Matches the system-design diagrams: the cloud does NOT classify packets. Its jobs are
  1. Logging / reporting : every detection result reported by the vehicle, RSU and near-edge nodes
                           is written to logs/cloud_feedback_<timestamp>.csv
  2. Training            : (re)trains the L1 / L2 / L3 classifiers and the DQN routing agent
  3. Model updates       : periodically (every `update_interval` reported packets, or on demand)
                           retrains on  [replay sample of the original training data + newly reported,
                           cloud-labelled traffic]  and DEPLOYS the new models to the vehicle, RSU and
                           near-edge nodes (hot-swap) - nodes are therefore not frozen.

NOTE (demo honesty): in the simulation the "cloud-confirmed labels" of the reported traffic come from the
replayed dataset (ground truth). Do not read post-update accuracy on those same rows as generalisation;
the before/after accuracy printed by update() is measured on held-out rows that were NOT reported.
"""

import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from config import LOGS_DIR, MODELS_DIR, RANDOM_SEED
from src.classifiers import HierarchicalIDSClassifier
from src.env import PacketRoutingEnv, prepare_classifiers_for_inference

logger = logging.getLogger(__name__)

# classifier name -> (feature-set key, label column)
TIERS = {
    "L1_vehicle": ("L1", "level1"),
    "L2_rsu": ("L2", "level2"),
    "L3_nearedge": ("L3", "level3"),
}
_LEVEL_ARRAY = {"level1": "level1_arr", "level2": "level2_arr", "level3": "level3_arr"}

CSV_HEADER = ["timestamp", "step", "policy", "row_idx", "route", "final_node", "final_pred", "true_label",
              "correct", "false_negative", "false_positive", "latency_ms", "hops", "model_version"]


class CloudManager:
    def __init__(
        self,
        train_df: pd.DataFrame,
        feature_sets: Dict[str, List[str]],
        classifiers: Dict[str, HierarchicalIDSClassifier],
        env: PacketRoutingEnv,
        agent,
        update_interval: int = 1000,
        min_new_samples: int = 200,
        replay_per_class: int = 8000,
        rl_episodes: int = 600,
        seed: int = RANDOM_SEED,
    ):
        self.feature_sets = feature_sets
        self.classifiers = classifiers          # SAME dict object the env uses -> hot-swap deploys instantly
        self.env = env
        self.agent = agent
        self.update_interval = update_interval
        self.min_new_samples = min_new_samples
        self.rl_episodes = rl_episodes
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.version = 1
        self.pending = 0
        self._train_env: Optional[PacketRoutingEnv] = None

        # replay sample of the ORIGINAL training data (keeps old knowledge while learning new traffic)
        parts = [g.sample(n=min(len(g), replay_per_class), random_state=seed)
                 for _, g in train_df.groupby("level3")]
        self.replay = pd.concat(parts, ignore_index=True)
        logger.info(f"CLOUD: replay buffer ready ({len(self.replay):,} rows from the training partition).")

        # reported traffic
        self.stream_rows: List[int] = []
        self._stream_set = set()
        self.stats = {"received": 0, "attacks": 0, "fn": 0, "fp": 0, "wrong": 0, "lat_sum": 0.0}

        # cloud feedback log (one file per run, timestamped)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOGS_DIR / f"cloud_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._fh = open(self.log_path, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._fh)
        self._csv.writerow(CSV_HEADER)
        logger.info(f"CLOUD: logging reported detections to {self.log_path}")

    # ------------------------------------------------------------------
    # Logging / reporting
    # ------------------------------------------------------------------
    def receive(self, step: int, policy: str, row_idx: int, route: str, final_node: str, final_pred: str,
                true_label: str, correct: bool, fn: bool, fp: bool, latency_ms: float, hops: int, is_attack: bool):
        """A node reports its detection result to the cloud."""
        self._csv.writerow([datetime.now().isoformat(timespec="milliseconds"), step, policy, row_idx, route,
                            final_node, final_pred, true_label, int(correct), int(fn), int(fp),
                            f"{latency_ms:.3f}", hops, self.version])
        s = self.stats
        s["received"] += 1
        s["attacks"] += int(is_attack)
        s["fn"] += int(fn)
        s["fp"] += int(fp)
        s["wrong"] += int(not correct)
        s["lat_sum"] += latency_ms
        self.pending += 1
        if row_idx not in self._stream_set:
            self._stream_set.add(row_idx)
            self.stream_rows.append(row_idx)
        if s["received"] % 50 == 0:
            self._fh.flush()

    def due(self) -> bool:
        return self.pending >= self.update_interval and len(self.stream_rows) >= self.min_new_samples

    def report(self) -> str:
        s = self.stats
        n = max(1, s["received"])
        return (f"CLOUD REPORT | model v{self.version} | reported={s['received']} attacks={s['attacks']} "
                f"FN={s['fn']} FP={s['fp']} wrong={s['wrong']} avg_latency={s['lat_sum']/n:.2f}ms "
                f"| pending_for_next_update={self.pending}/{self.update_interval}")

    # ------------------------------------------------------------------
    # Periodic retraining + deployment
    # ------------------------------------------------------------------
    def update(self, progress: Optional[Callable[[str], None]] = None, reason: str = "scheduled") -> str:
        """Retrain L1/L2/L3 + fine-tune the RL agent in the cloud, then deploy to the nodes."""
        def say(msg: str):
            logger.info(msg)
            if progress:
                progress(msg)

        t0 = time.time()
        ver = self.version + 1
        say(f"CLOUD: model update v{ver} started ({reason}) with {len(self.stream_rows)} reported samples")

        stream_df = self.env.dataset_df.iloc[self.stream_rows]
        data = pd.concat([self.replay, stream_df], ignore_index=True)

        # held-out rows (NOT reported to the cloud) to measure before/after honestly
        mask = np.ones(len(self.env.dataset_df), dtype=bool)
        mask[self.stream_rows] = False
        pool = np.where(mask)[0]
        eval_idx = self.rng.choice(pool, size=min(4000, len(pool)), replace=False)

        new_models: Dict[str, HierarchicalIDSClassifier] = {}
        for name, (fkey, col) in TIERS.items():
            old = self.classifiers[name]
            X = data[self.feature_sets[fkey]].to_numpy(dtype=np.float32)
            y = data[col].to_numpy()
            clf = HierarchicalIDSClassifier(
                level_name=old.level_name, feature_names=old.feature_names, classes=old.classes,
                backend=old.backend, n_estimators=old.n_estimators, max_depth=old.max_depth,
                random_state=old.random_state + ver,
            )
            clf.fit(X, y)
            new_models[name] = clf
            say(f"CLOUD: retrained {name} on {len(data):,} rows")

        prepare_classifiers_for_inference(new_models, self.feature_sets, self.env.dataset_df)

        # VALIDATION GATE: compare old vs new on held-out rows (not reported to the cloud).
        # A new model is deployed only if it is not worse than the running one (tolerance 0.2 points).
        TOL = 0.2
        gate_lines, deploy = [], {}
        for name, (fkey, col) in TIERS.items():
            Xe = self.env.feat_matrices[fkey][eval_idx]
            ye = getattr(self.env, _LEVEL_ARRAY[col])[eval_idx]
            a_old = accuracy_score(ye, self.classifiers[name].predict(Xe)) * 100
            a_new = accuracy_score(ye, new_models[name].predict(Xe)) * 100
            ok = a_new + TOL >= a_old
            if ok:
                deploy[name] = new_models[name]
            gate_lines.append(f"{name}: {a_old:.2f}% -> {a_new:.2f}% ({'DEPLOY' if ok else 'KEEP OLD'})")

        # DEPLOY: hot-swap into the dict the env reads from (vehicle / RSU / near-edge now use the new models)
        if deploy:
            self.classifiers.update(deploy)
            upd_dir = MODELS_DIR / "updates"
            upd_dir.mkdir(parents=True, exist_ok=True)
            for name, clf in deploy.items():
                joblib.dump(clf, upd_dir / f"{name}_v{ver}.joblib")
        upd_dir = MODELS_DIR / "updates"
        upd_dir.mkdir(parents=True, exist_ok=True)
        say("CLOUD: held-out validation | " + " | ".join(gate_lines))

        # RL agent fine-tuning on fresh experience (new classifiers + current traffic mix)
        r_first, r_last = self._finetune_agent(self.rl_episodes)
        self.agent.save(upd_dir / f"rl_policy_v{ver}.pt")
        say(f"CLOUD: RL policy fine-tuned ({self.rl_episodes} episodes, avg reward {r_first:.1f} -> {r_last:.1f}) "
            f"and deployed")

        self.version = ver
        self.pending = 0
        summary = (f"CLOUD: update v{ver} complete - deployed {len(deploy)}/3 classifiers + RL policy "
                   f"({time.time()-t0:.1f}s)")
        say(summary)
        return summary

    def _get_train_env(self) -> PacketRoutingEnv:
        if self._train_env is None:
            self._train_env = PacketRoutingEnv(
                dataset_df=self.env.dataset_df, feature_sets=self.feature_sets,
                classifiers=self.classifiers, random_seed=self.seed + 7,
            )
        return self._train_env

    def _finetune_agent(self, episodes: int):
        env = self._get_train_env()
        env.net_sim.reset()
        agent = self.agent
        stream = np.array(self.stream_rows)
        old_lr = [g["lr"] for g in agent.optimizer.param_groups]
        for g in agent.optimizer.param_groups:
            g["lr"] = 2e-4                        # gentle fine-tune, avoid forgetting
        agent.epsilon = 0.15
        rewards = []
        for ep in range(episodes):
            env.net_sim.burst_mode = ((ep // 50) % 4 == 3)     # keep ~25% burst regime, in blocks
            row = int(self.rng.choice(stream)) if (len(stream) and self.rng.rand() < 0.7) else None
            state = env.reset(row_idx=row)
            done, ep_r = False, 0.0
            while not done:
                mask = env.get_action_mask()
                a = agent.select_action(state, mask, greedy=False)
                ns, r, done, _ = env.step(a)
                agent.memory.push(state, a, r, ns, done, env.get_action_mask())
                agent.update()
                state, ep_r = ns, ep_r + r
            rewards.append(ep_r)
        for g, lr in zip(agent.optimizer.param_groups, old_lr):
            g["lr"] = lr
        agent.epsilon = agent.epsilon_min
        k = max(1, min(100, len(rewards) // 4))
        return float(np.mean(rewards[:k])), float(np.mean(rewards[-k:]))

    def close(self):
        logger.info(self.report())
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
