"""Main Interactive Demonstration Launcher for IoV Dynamic Packet Routing HIDS.

Features
  * Timestamped session log file (logs/simulation_YYYYMMDD_HHMMSS.log)
  * Live per-level metrics table (Precision / Recall / F1 / Accuracy) for our implementation
  * --benchmark : Flat vs Fixed (paper) vs RL on identical packets, no SUMO needed
  * SUMO + TraCI co-simulation with live Pygame dashboard
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import traci
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import (
    CACHE_DIR,
    CLOUD_MIN_NEW_SAMPLES,
    CLOUD_REPLAY_PER_CLASS,
    CLOUD_RL_FINETUNE_EPISODES,
    CLOUD_UPDATE_INTERVAL,
    LOGS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    SUMO_CONFIG_DIR,
    SUMO_HOME,
)
from src.baselines import PolicyEvaluator
from src.cloud import CloudManager
from src.dashboard import CompanionDashboard
from src.env import PacketRoutingEnv, prepare_classifiers_for_inference
from src.rl_agent import DQNAgent
from src.sumo_bridge import (
    INTERSECTION_NODES,
    RSU_TO_EDGE,
    SumoVisualizerBridge,
    get_nearest_rsu_id,
)


# ----------------------------------------------------------------------------
# Logging (console + timestamped file, new file every run)
# ----------------------------------------------------------------------------
def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    return log_path


LOG_PATH = setup_logging()
logger = logging.getLogger(__name__)

TIER_KEYS = {"L1_vehicle": "L1", "L2_rsu": "L2", "L3_nearedge": "L3"}   # cloud does not classify


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IoV Dynamic Routing HIDS Interactive Demo")
    p.add_argument("--headless", action="store_true", help="Run without SUMO GUI / Pygame display")
    p.add_argument("--steps", type=int, default=None, help="Maximum simulation steps (default: infinite)")
    p.add_argument("--policy", type=str, default="RL", choices=["Flat", "Fixed", "RL"], help="Initial routing policy")
    p.add_argument("--burst", action="store_true", help="Start with DoS burst active")
    p.add_argument("--delay", type=int, default=35, help="Simulation step delay in ms")
    p.add_argument("--sample-rate", type=int, default=2, help="Emit packet every N simulation steps")
    p.add_argument("--cloud-interval", type=int, default=CLOUD_UPDATE_INTERVAL,
                   help="Reported packets between automatic cloud model updates (0 = only on [U])")
    p.add_argument("--benchmark", action="store_true", help="Compare Flat/Fixed/RL on identical packets and exit")
    p.add_argument("--benchmark-packets", type=int, default=5000, help="Packets per policy in benchmark mode")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Asset loading / preparation
# ----------------------------------------------------------------------------
def load_assets():
    test_df = pd.read_parquet(CACHE_DIR / "test.parquet")
    logger.info(f"Loaded test dataset: {len(test_df):,} rows.")
    with open(MODELS_DIR / "feature_sets.json", "r", encoding="utf-8") as f:
        feature_sets = json.load(f)
    classifiers = {name: joblib.load(MODELS_DIR / f"{name}.joblib") for name in TIER_KEYS}
    hint_path = MODELS_DIR / "L1_type.joblib"
    if hint_path.exists():
        classifiers["L1_type"] = joblib.load(hint_path)
    else:
        logger.warning("L1_type.joblib not found - RL will NOT see the attack-type hint. Run train_classifiers.py.")
    logger.info("Loaded classifiers deployed to the nodes (L1 vehicle, L2 RSU, L3 near-edge).")
    return test_df, feature_sets, classifiers


def print_classifier_metrics(classifiers, feature_sets, test_df):
    """Computes Pre/Recall/F1/Acc (macro) on the test set per level and prints them once
    (console + log file)."""
    levels = [
        ("Root (L1 Vehicle)", "L1_vehicle", "L1", "level1"),
        ("Second (L2 RSU)", "L2_rsu", "L2", "level2"),
        ("Third (L3 NearEdge)", "L3_nearedge", "L3", "level3"),
    ]
    out = ["", "=" * 96,
           f"  PROPOSED SYSTEM - PERFORMANCE PER LEVEL (test set: {len(test_df):,} rows, macro-averaged, %)",
           "=" * 96,
           f"  {'Level':<22}{'#Feat':>6}{'Backend':>14}{'Pre':>9}{'Recall':>9}{'F1':>9}{'Acc':>9}",
           "-" * 96]
    for label, name, fkey, ycol in levels:
        try:
            clf = classifiers[name]
            X = test_df[feature_sets[fkey]].to_numpy(dtype=np.float32)
            y = test_df[ycol].to_numpy()
            pred = clf.predict(X)
            pre, rec, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
            acc = accuracy_score(y, pred)
            out.append(f"  {label:<22}{len(feature_sets[fkey]):>6}{clf.backend:>14}"
                       f"{pre*100:>9.2f}{rec*100:>9.2f}{f1*100:>9.2f}{acc*100:>9.2f}")
        except Exception as e:
            out.append(f"  {label:<22} could not evaluate: {e}")
    if "L1_type" in classifiers:
        try:
            atk = test_df[test_df["level1"] == "Attack"]
            hp = classifiers["L1_type"].predict(atk[feature_sets["L1"]].to_numpy(dtype=np.float32))
            pre, rec, f1, _ = precision_recall_fscore_support(atk["level2"].to_numpy(), hp, average="macro", zero_division=0)
            out.append(f"  {'Attack-type hint (DoS/Spoof)':<22}{len(feature_sets['L1']):>6}{classifiers['L1_type'].backend:>14}"
                       f"{pre*100:>9.2f}{rec*100:>9.2f}{f1*100:>9.2f}{accuracy_score(atk['level2'].to_numpy(), hp)*100:>9.2f}")
        except Exception as ex:
            out.append(f"  attack-type hint could not be evaluated: {ex}")
    out += ["=" * 96,
            "  Policies: [1] Flat (no hierarchy)  [2] Fixed (paper hierarchy)  [3] RL (proposed)",
            "  Hotkeys: [B] Burst  [UP/DOWN] Rate  [R] Reset  [SPACE] Pause  [S] Screenshot  [U] Cloud update  [ESC] Exit",
            "=" * 96, ""]
    logger.info("Classifier metrics:" + "\n".join(out))   # logger prints to console AND the log file (only once)


def build_env_and_agent(test_df, feature_sets, classifiers):
    env = PacketRoutingEnv(dataset_df=test_df, feature_sets=feature_sets,
                           classifiers=classifiers, random_seed=RANDOM_SEED)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, random_seed=RANDOM_SEED)
    agent.load(MODELS_DIR / "rl_policy.pt")
    logger.info("Loaded trained RL policy from models/rl_policy.pt.")
    return env, agent


# ----------------------------------------------------------------------------
# Benchmark (no SUMO)
# ----------------------------------------------------------------------------
def run_benchmark(n_packets: int, env, agent):
    """Flat vs Fixed vs RL on identical packets: first half normal traffic, second half DoS burst."""
    evaluator = PolicyEvaluator(env, agent)
    idx = np.random.default_rng(RANDOM_SEED).integers(0, len(env.dataset_df), size=n_packets)
    half = n_packets // 2

    rows = []
    for policy in ("Flat", "Fixed", "RL"):
        env.net_sim.reset()
        recs = []
        for i, row in enumerate(idx):
            env.net_sim.burst_mode = i >= half
            r = evaluator.run_single_packet(policy, int(row))
            r["burst"] = int(i >= half)
            recs.append(r)
        df = pd.DataFrame(recs)
        benign = df[df["is_attack"] == 0]
        rows.append((policy, 100.0 * df["correct"].mean(),
                     df.loc[df["burst"] == 0, "latency_ms"].mean(), df.loc[df["burst"] == 1, "latency_ms"].mean(),
                     df["latency_ms"].quantile(0.95), df["hops"].mean(),
                     int(df["false_negative"].sum()), int(df["false_positive"].sum()),
                     100.0 * (benign["final_node"] != "vehicle").mean(),
                     {"normal": df.loc[df["burst"] == 0, "final_node"].value_counts().to_dict(),
                      "burst": df.loc[df["burst"] == 1, "final_node"].value_counts().to_dict()}))

    out = ["", "=" * 124, f"  POLICY BENCHMARK - {n_packets} identical packets per policy (first half normal, second half DoS burst)",
           "=" * 124,
           f"  {'Policy':<7}{'Correct%':>9}{'Lat normal':>12}{'Lat burst':>11}{'P95 ms':>9}{'Hops':>7}{'FN':>6}{'FP':>6}{'Benign esc%':>13}",
           "-" * 124]
    for p, acc, ln, lb, p95, mh, fn, fp, esc, dest in rows:
        out.append(f"  {p:<7}{acc:>9.2f}{ln:>12.2f}{lb:>11.2f}{p95:>9.2f}{mh:>7.2f}{fn:>6}{fp:>6}{esc:>13.2f}")
        out.append(f"          final nodes  normal: {dest['normal']}   burst: {dest['burst']}")
    out += ["=" * 124,
            "  Correct% is scored at each final node's granularity (vehicle=L1, RSU=L2, near-edge=L3). Latency in ms.",
            "  Benign esc% = share of benign packets that did NOT finish at the vehicle.", ""]
    logger.info("Benchmark results:" + "\n".join(out))


# ----------------------------------------------------------------------------
# Interactive SUMO co-simulation
# ----------------------------------------------------------------------------
def run_interactive_simulation(args: argparse.Namespace, test_df, env, agent):
    logger.info("=" * 70)
    logger.info("STARTING IOV DYNAMIC ROUTING HIDS INTERACTIVE SIMULATION")
    logger.info("=" * 70)
    logger.info(f"Session start: {datetime.now().isoformat(timespec='seconds')} | Log file: {LOG_PATH}")
    logger.info(f"Args: {vars(args)}")

    os.environ["SUMO_HOME"] = str(SUMO_HOME)
    sumo_bin_dir = Path(SUMO_HOME) / "bin"
    if sumo_bin_dir.exists() and str(sumo_bin_dir) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(sumo_bin_dir) + os.pathsep + os.environ.get("PATH", "")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    sumocfg_path = SUMO_CONFIG_DIR / "intersection.sumocfg"
    if not sumocfg_path.exists():
        raise FileNotFoundError(f"SUMO config file not found: {sumocfg_path}")

    env.net_sim.burst_mode = args.burst
    evaluator = PolicyEvaluator(env, agent)

    sumo_binary = "sumo" if args.headless else "sumo-gui"
    sumo_cmd = [sumo_binary, "-c", str(sumocfg_path), "--start", "--quit-on-end", "--delay", str(args.delay)]
    logger.info(f"Connecting TraCI to {sumo_binary}...")
    try:
        traci.start(sumo_cmd)
    except Exception as e:
        logger.warning(f"Failed to start {sumo_binary}: {e}. Retrying with headless sumo...")
        sumo_cmd[0] = "sumo"
        traci.start(sumo_cmd)

    bridge = SumoVisualizerBridge(use_intersection=True)
    bridge.setup_infrastructure()

    dashboard = CompanionDashboard()
    dashboard.current_mode = args.policy
    dashboard.attack_burst_active = args.burst
    dashboard.decision_logs.append(f">> Interactive session initialized in {args.policy} mode.")

    # Cloud tier: logging/reporting + periodic retraining + deployment (it does not classify)
    fs = env.feature_sets
    cols = sorted(set(fs["L1"]) | set(fs["L2"]) | set(fs["L3"])) + ["level1", "level2", "level3"]
    cloud_train_df = pd.read_parquet(CACHE_DIR / "train.parquet", columns=cols)
    cloud = CloudManager(
        cloud_train_df, fs, env.classifiers, env, agent,
        update_interval=args.cloud_interval if args.cloud_interval > 0 else 10**12,
        min_new_samples=CLOUD_MIN_NEW_SAMPLES, replay_per_class=CLOUD_REPLAY_PER_CLASS,
        rl_episodes=CLOUD_RL_FINETUNE_EPISODES,
    )
    del cloud_train_df
    dashboard.decision_logs.append(
        f">> Cloud online: auto-update every {args.cloud_interval} reported packets, [U] = update now.")

    step_count = packet_seq = test_idx = 0
    attack_indices = test_df.index[test_df["level1"] == "Attack"].to_numpy()
    last_mode = dashboard.current_mode
    lat_sum = hop_sum = 0.0
    rng = np.random.default_rng(RANDOM_SEED)

    try:
        while True:
            if args.steps is not None and step_count >= args.steps:
                logger.info(f"Reached specified step limit ({args.steps}). Stopping.")
                break

            events = dashboard.handle_events()
            if events.get("quit") or not dashboard.running:
                logger.info("User requested exit.")
                break

            if dashboard.current_mode != last_mode:
                logger.info(f"[step {step_count}] Policy switched: {last_mode} -> {dashboard.current_mode}")
                last_mode = dashboard.current_mode
            if "burst_toggle" in events:
                env.net_sim.burst_mode = events["burst_toggle"]
                logger.info(f"[step {step_count}] Attack burst mode: {env.net_sim.burst_mode}")
            if "rate_change" in events:
                logger.info(f"[step {step_count}] Packet rate multiplier: {events['rate_change']}x")
            if events.get("reset"):
                env.net_sim.reset()
                env.net_sim.burst_mode = dashboard.attack_burst_active   # reset() clears it; keep UI in sync
                lat_sum = hop_sum = 0.0                                  # restart running averages
                logger.info(f"[step {step_count}] Network queues and statistics reset.")

            if events.get("cloud_update") or cloud.due():
                reason = "manual [U]" if events.get("cloud_update") else f"scheduled, every {args.cloud_interval} reported packets"

                def _progress(msg: str):
                    dashboard.log_decision(">> " + msg[:115])
                    dashboard.render()

                logger.info(f"[step {step_count}] {cloud.report()}")
                dashboard.log_decision(">> CLOUD: retraining... simulation paused")
                dashboard.render()
                cloud.update(progress=_progress, reason=reason)

            if dashboard.paused:
                dashboard.render()
                time.sleep(0.05)
                continue

            traci.simulationStep()
            step_count += 1
            bridge.step_links()
            bridge.update_packet_animations()
            veh_ids = traci.vehicle.getIDList()

            emit_every = max(1, int(round(args.sample_rate / max(0.5, dashboard.packet_rate))))
            if step_count % emit_every == 0 and len(veh_ids) > 0:
                packet_seq += 1
                veh_id = veh_ids[packet_seq % len(veh_ids)]
                try:
                    veh_pos = traci.vehicle.getPosition(veh_id)
                except Exception:
                    veh_pos = None

                if veh_pos is not None:
                    if dashboard.attack_burst_active and rng.random() < 0.8:
                        row_idx = int(rng.choice(attack_indices))
                    else:
                        row_idx = test_idx % len(test_df)
                        test_idx += 1

                    active_policy = dashboard.current_mode
                    result = evaluator.run_single_packet(active_policy, row_idx)
                    dest_node_key = result["final_node"]
                    is_attack = bool(result["is_attack"])
                    is_fn = bool(result["false_negative"])
                    is_fp = bool(result["false_positive"])
                    final_label = result["final_pred"]

                    nearest_rsu = get_nearest_rsu_id(veh_pos)
                    rsu_pos = INTERSECTION_NODES[nearest_rsu]["pos"]
                    edge_id = RSU_TO_EDGE[nearest_rsu]
                    edge_pos = INTERSECTION_NODES[edge_id]["pos"]
                    cloud_pos = INTERSECTION_NODES["Cloud"]["pos"]

                    if dest_node_key == "vehicle":
                        waypoints = [veh_pos]
                    elif dest_node_key == "rsu":
                        waypoints = [veh_pos, rsu_pos]
                    else:  # nearedge (cloud never classifies)
                        waypoints = [veh_pos, rsu_pos, edge_pos]
                    decision_pos = waypoints[-1]
                    # Detected attacks are reported to the cloud over the backhaul (logging / reporting)
                    if final_label != "Benign" and dest_node_key != "vehicle":
                        if dest_node_key == "rsu":
                            waypoints = waypoints + [edge_pos]
                        waypoints = waypoints + [cloud_pos]

                    if len(waypoints) > 1:
                        bridge.spawn_multihop_packet(
                            packet_id=str(packet_seq), waypoints=waypoints, label_class=final_label,
                            is_attack=is_attack, is_fn=is_fn, steps_per_hop=5)
                        bridge.show_decision_label(decision_pos[0], decision_pos[1],
                                                   f"{active_policy}: {final_label}")
                    bridge.update_vehicle_color(veh_id, final_label)

                    rsu_q = env.net_sim.nodes["rsu"].queue_length
                    node_queues = {
                        "RSU_NE": rsu_q // 4, "RSU_NW": rsu_q // 4, "RSU_SE": rsu_q // 4, "RSU_SW": rsu_q // 4,
                        "Edge_N": env.net_sim.nodes["nearedge"].queue_length // 2,
                        "Edge_S": env.net_sim.nodes["nearedge"].queue_length // 2,
                        "Cloud": env.net_sim.nodes["cloud"].queue_length,
                    }
                    bridge.update_node_loadbars(node_queues)

                    st = dashboard.stats
                    st["total_packets"] += 1
                    if is_attack:
                        st["attacks_detected" if not is_fn else "false_negatives"] += 1
                    elif is_fp:
                        st["false_positives"] += 1

                    lat_sum += float(result["latency_ms"])
                    hop_sum += float(result["hops"])
                    st["avg_latency"] = lat_sum / st["total_packets"]
                    st["avg_hops"] = hop_sum / st["total_packets"]
                    st["queue_rsu"] = node_queues["RSU_NE"] * 4
                    st["queue_nearedge"] = node_queues["Edge_N"] + node_queues["Edge_S"]
                    st["queue_cloud"] = node_queues["Cloud"]
                    st["load_rsu"] = env.net_sim.nodes["rsu"].load_ratio * 100.0
                    st["load_nearedge"] = env.net_sim.nodes["nearedge"].load_ratio * 100.0
                    st["load_cloud"] = env.net_sim.nodes["cloud"].load_ratio * 100.0

                    cloud.receive(step_count, active_policy, row_idx, result["route_taken"], dest_node_key,
                                  final_label, result["true_l3"], bool(result["correct"]), is_fn, is_fp,
                                  float(result["latency_ms"]), int(result["hops"]), is_attack)

                    dashboard.log_decision(
                        f"[{step_count:04d}] {active_policy:<5} | Pkt#{packet_seq:04d} "
                        f"True:{result['true_l3']:<6} -> {dest_node_key:<8} | {result['latency_ms']:.1f}ms"
                        + (" | FN:" if is_fn else ""))
                    logger.info(
                        f"PKT {packet_seq:05d} | policy={active_policy} | veh={veh_id} | route={result['route_taken']} "
                        f"| true={result['true_l3']} | pred={final_label} | attack={is_attack} FN={is_fn} FP={is_fp} "
                        f"| correct={bool(result['correct'])} | hops={result['hops']} | latency={result['latency_ms']:.2f}ms")

            dashboard.update_metrics(dashboard.stats)
            dashboard.render()

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.error(f"Error during simulation: {e}", exc_info=True)
    finally:
        logger.info("Closing TraCI and cleaning up...")
        try:
            traci.close()
        except Exception:
            pass
        dashboard.close()
        try:
            cloud.close()
        except Exception:
            pass
        st = dashboard.stats
        logger.info("=" * 70)
        logger.info("SIMULATION FINISHED.")
        logger.info(f"Total Packets Processed: {st['total_packets']}")
        logger.info(f"Attacks Detected: {st['attacks_detected']} | False Negatives: {st['false_negatives']} "
                    f"| False Positives: {st['false_positives']}")
        logger.info(f"Avg Latency: {st['avg_latency']:.2f} ms | Avg Hops: {st['avg_hops']:.2f}")
        logger.info(f"Log saved to: {LOG_PATH}")
        logger.info("=" * 70)


if __name__ == "__main__":
    cli_args = parse_arguments()
    logger.info(f"Log file: {LOG_PATH}")
    test_df, feature_sets, classifiers = load_assets()
    print_classifier_metrics(classifiers, feature_sets, test_df)
    prepare_classifiers_for_inference(classifiers, feature_sets, test_df)
    env, agent = build_env_and_agent(test_df, feature_sets, classifiers)
    if cli_args.benchmark:
        run_benchmark(cli_args.benchmark_packets, env, agent)
    else:
        run_interactive_simulation(cli_args, test_df, env, agent)
