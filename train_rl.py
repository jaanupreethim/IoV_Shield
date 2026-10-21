"""Headless training script for the DQN Dynamic Routing Agent.
Trains on TRAIN+VAL data under steady-state and DoS burst regimes.
Saves trained policy to models/rl_policy.pt and learning curves to results/.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib
matplotlib.use("Agg")  # non-GUI backend: avoids Tk crash when plotting from worker threads
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CACHE_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.env import PacketRoutingEnv, prepare_classifiers_for_inference
from src.rl_agent import DQNAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_dqn_policy(total_episodes: int = 12000):
    """Executes fast headless training of the DQN policy."""
    logger.info("Initializing DQN training on Train + Val datasets...")

    # Train the RL agent on VAL only. The frozen classifiers were fitted on TRAIN, so on train rows their
    # confidences are over-optimistic and the agent would learn to trust them too much. Test stays untouched.
    val_df = pd.read_parquet(CACHE_DIR / "val.parquet")
    combined_train_val = val_df
    logger.info(f"RL training dataset (validation partition): {len(combined_train_val):,} rows.")

    with open(MODELS_DIR / "feature_sets.json", "r", encoding="utf-8") as f:
        feature_sets = json.load(f)

    # Load frozen classifiers
    classifiers = {
        "L1_vehicle": joblib.load(MODELS_DIR / "L1_vehicle.joblib"),
        "L2_rsu": joblib.load(MODELS_DIR / "L2_rsu.joblib"),
        "L3_nearedge": joblib.load(MODELS_DIR / "L3_nearedge.joblib"),
    }
    _hint = MODELS_DIR / "L1_type.joblib"
    if _hint.exists():
        classifiers["L1_type"] = joblib.load(_hint)
    else:
        logger.warning("L1_type.joblib not found - RL will NOT see the attack-type hint. Run train_classifiers.py.")

    prepare_classifiers_for_inference(classifiers, feature_sets, combined_train_val)

    env = PacketRoutingEnv(
        dataset_df=combined_train_val,
        feature_sets=feature_sets,
        classifiers=classifiers,
        random_seed=RANDOM_SEED,
    )

    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9997,
        buffer_capacity=50000,
        batch_size=64,
        target_update_freq=100,
        random_seed=RANDOM_SEED,
    )

    attack_idx = np.where(env.level1_arr == "Attack")[0]
    ep_rng = np.random.RandomState(RANDOM_SEED + 1)
    log_records = []
    moving_reward = 0.0

    t0 = time.time()
    logger.info(f"Starting headless DQN training loop ({total_episodes} episodes)...")

    for ep in range(1, total_episodes + 1):
        # DoS burst regime for 25% of training (in blocks)
        # Burst regime in BLOCKS of 150 episodes (25% of the time) so queues actually build up and the
        # agent can observe congestion. Toggling every 4th episode let queues drain in between.
        env.net_sim.burst_mode = ((ep // 150) % 4 == 3)

        # Oversample attacks: only ~13% of the data is attack, so uniform sampling gave ~1,500 attack episodes in
        # 12,000 - far too few to learn the RSU-vs-NearEdge trade-off. 50% attacks normally, 80% during bursts.
        p_attack = 0.8 if env.net_sim.burst_mode else 0.5
        row = int(ep_rng.choice(attack_idx)) if ep_rng.rand() < p_attack else None
        state = env.reset(row_idx=row)
        done = False
        ep_reward = 0.0
        ep_loss = 0.0
        loss_steps = 0

        while not done:
            mask = env.get_action_mask()
            action = agent.select_action(state, mask, greedy=False)

            next_state, reward, done, info = env.step(action)
            next_mask = env.get_action_mask()

            agent.memory.push(state, action, reward, next_state, done, next_mask)
            loss = agent.update()
            if loss is not None:
                ep_loss += loss
                loss_steps += 1

            ep_reward += reward
            state = next_state

        avg_loss = ep_loss / max(1, loss_steps)
        moving_reward = 0.95 * moving_reward + 0.05 * ep_reward if ep > 1 else ep_reward

        is_fn = 1 if info.get("false_negative", False) else 0
        is_corr = 1 if info.get("correct", False) else 0

        log_records.append({
            "episode": ep,
            "reward": float(ep_reward),
            "moving_reward": float(moving_reward),
            "loss": float(avg_loss),
            "epsilon": float(agent.epsilon),
            "latency_ms": float(info.get("latency_ms", 0.0)),
            "hops": int(info.get("hops", 0)),
            "correct": is_corr,
            "false_negative": is_fn,
            "burst_active": int(env.net_sim.burst_mode),
        })

        if ep % 250 == 0 or ep == total_episodes:
            logger.info(f"Episode {ep:4d}/{total_episodes} | Avg Reward: {moving_reward:6.2f} | Eps: {agent.epsilon:.3f} | Latency: {info.get('latency_ms', 0):5.1f}ms | Hops: {info.get('hops', 0)}")

    total_time = time.time() - t0
    logger.info(f"DQN Training finished in {total_time:.2f} seconds.")

    # Save policy
    model_save_path = MODELS_DIR / "rl_policy.pt"
    agent.save(model_save_path)
    logger.info(f"Saved trained DQN policy to {model_save_path}")

    # Save training log CSV
    log_df = pd.DataFrame(log_records)
    csv_path = RESULTS_DIR / "training_log.csv"
    log_df.to_csv(csv_path, index=False)
    logger.info(f"Saved training log CSV to {csv_path}")

    # Plot & Save Learning Curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Reward
    axes[0, 0].plot(log_df["episode"], log_df["reward"], alpha=0.3, color="steelblue", label="Episode Reward")
    axes[0, 0].plot(log_df["episode"], log_df["moving_reward"], color="darkblue", linewidth=2, label="Moving Average")
    axes[0, 0].set_title("Reward Curve")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()

    # Latency
    roll_lat = log_df["latency_ms"].rolling(50, min_periods=1).mean()
    axes[0, 1].plot(log_df["episode"], roll_lat, color="coral", linewidth=1.8)
    axes[0, 1].set_title("Average Latency (50-ep Moving Avg)")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Latency (ms)")

    # Hops
    roll_hops = log_df["hops"].rolling(50, min_periods=1).mean()
    axes[1, 0].plot(log_df["episode"], roll_hops, color="forestgreen", linewidth=1.8)
    axes[1, 0].set_title("Average Hops (50-ep Moving Avg)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Hops")

    # Loss & Epsilon
    axes[1, 1].plot(log_df["episode"], log_df["loss"], color="purple", alpha=0.6, label="Loss")
    axes[1, 1].set_title("Training Loss")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Smooth L1 Loss")
    axes[1, 1].legend()

    plt.tight_layout()
    curve_path = RESULTS_DIR / "rl_learning_curves.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()
    logger.info(f"Saved learning curves plot to {curve_path}")

    return agent, log_df


if __name__ == "__main__":
    train_dqn_policy(total_episodes=12000)
