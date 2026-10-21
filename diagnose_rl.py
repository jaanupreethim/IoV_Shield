"""Shows WHERE the trained RL agent sends each attack type (first routing decision), normal vs burst.
Run after train_rl.py:   python diagnose_rl.py
Expected: DoS/spoofing -> RSU first in normal traffic; near-edge first during a burst (RSU congested)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import torch

from config import CACHE_DIR, MODELS_DIR, RANDOM_SEED
from src.env import PacketRoutingEnv, prepare_classifiers_for_inference
from src.rl_agent import DQNAgent


def main(n_per_class: int = 300):
    test_df = pd.read_parquet(CACHE_DIR / "test.parquet")
    with open(MODELS_DIR / "feature_sets.json", "r", encoding="utf-8") as f:
        fs = json.load(f)
    clfs = {k: joblib.load(MODELS_DIR / f"{k}.joblib") for k in ("L1_vehicle", "L2_rsu", "L3_nearedge")}
    if (MODELS_DIR / "L1_type.joblib").exists():
        clfs["L1_type"] = joblib.load(MODELS_DIR / "L1_type.joblib")
    prepare_classifiers_for_inference(clfs, fs, test_df)
    env = PacketRoutingEnv(test_df, fs, clfs, random_seed=RANDOM_SEED)
    agent = DQNAgent(env.state_dim, env.action_dim)
    agent.load(MODELS_DIR / "rl_policy.pt")
    rng = np.random.RandomState(0)

    print(f"\n{'true class':<10}{'traffic':<8}{'->RSU':>8}{'->NearEdge':>12}{'ACCEPT':>8}{'  mean Q(RSU)-Q(Edge)':>24}")
    print("-" * 72)
    for true in ["DoS", "Gas", "RPM", "Speed", "SW"]:
        idx = np.where(env.level3_arr == true)[0]
        for burst in (False, True):
            env.net_sim.reset()
            env.net_sim.burst_mode = burst
            for _ in range(200):               # let queues settle for this regime
                env.net_sim.step(0.1)
            acts, dq = [], []
            for i in rng.choice(idx, size=min(n_per_class, len(idx)), replace=False):
                s = env.reset(row_idx=int(i))
                if env.current_pred_label == "Benign":      # L1 missed it -> not a routing decision
                    continue
                with torch.no_grad():
                    q = agent.q_net(torch.FloatTensor(s).unsqueeze(0)).squeeze(0).numpy()
                acts.append(agent.select_action(s, env.get_action_mask(), greedy=True))
                dq.append(q[1] - q[2])
            a = np.array(acts)
            if len(a) == 0:
                continue
            print(f"{true:<10}{'burst' if burst else 'normal':<8}{100*np.mean(a==1):>7.0f}%{100*np.mean(a==2):>11.0f}%"
                  f"{100*np.mean(a==0):>7.0f}%{np.mean(dq):>22.2f}")
    print("\nQ(RSU)-Q(Edge) > 0 means the agent prefers the RSU first.\n")


if __name__ == "__main__":
    main()
