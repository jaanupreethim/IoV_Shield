"""Phase 2 Validation Script: SUMO Intersection Scenario and Live Visualisation.
Launches sumo-gui via TraCI with clean off-white intersection styling matching user diagram.
Streams test set CAN packets, animates links and moving packet POIs,
updates node loadbars, and saves screenshots of both windows.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import traci

from config import (
    CACHE_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SUMO_CONFIG_DIR,
    SUMO_HOME,
)
from src.dashboard import CompanionDashboard
from src.sumo_bridge import INTERSECTION_NODES, SumoVisualizerBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_phase2_simulation(total_steps: int = 140, headless: bool = False):
    """Executes Phase 2 intersection simulation run with visual validation."""
    os.environ["SUMO_HOME"] = str(SUMO_HOME)
    sumocfg_path = SUMO_CONFIG_DIR / "intersection.sumocfg"

    # Load test set partition and trained classifiers
    test_path = CACHE_DIR / "test.parquet"
    test_df = pd.read_parquet(test_path)
    logger.info(f"Loaded {len(test_df):,} test set rows for packet emission.")

    with open(MODELS_DIR / "feature_sets.json", "r", encoding="utf-8") as f:
        feature_sets = json.load(f)

    l1_clf = joblib.load(MODELS_DIR / "L1_vehicle.joblib")
    l2_clf = joblib.load(MODELS_DIR / "L2_rsu.joblib")
    l3_clf = joblib.load(MODELS_DIR / "L3_nearedge.joblib")

    # Start SUMO
    sumo_binary = "sumo" if headless else "sumo-gui"
    sumo_cmd = [
        sumo_binary,
        "-c",
        str(sumocfg_path),
        "--start",
        "--quit-on-end",
        "--delay",
        "35",
    ]

    logger.info(f"Starting {sumo_binary} via TraCI on intersection scenario...")
    traci.start(sumo_cmd)

    bridge = SumoVisualizerBridge(use_intersection=True)
    bridge.setup_infrastructure()

    dashboard = CompanionDashboard()
    dashboard.decision_logs.append(">> Intersection Simulation Started.")

    sim_time = 0.0
    packet_id_seq = 0
    test_idx = 0

    # Node simulated queues
    queue_counts = {"RSU": 2, "Edge": 4, "Cloud": 6}

    logger.info(f"Running simulation loop for {total_steps} steps...")

    try:
        for step in range(total_steps):
            traci.simulationStep()
            sim_time += 0.1

            events = dashboard.handle_events()
            if events.get("quit") or not dashboard.running:
                logger.info("User requested exit from dashboard.")
                break

            bridge.step_links()
            bridge.update_packet_animations()

            veh_ids = traci.vehicle.getIDList()

            # Emit packet every 2 steps from an active vehicle
            if step % 2 == 0 and len(veh_ids) > 0:
                packet_id_seq += 1
                veh_id = veh_ids[step % len(veh_ids)]
                veh_pos = traci.vehicle.getPosition(veh_id)

                row = test_df.iloc[test_idx % len(test_df)]
                test_idx += 1
                true_l1 = row["level1"]
                true_l2 = row["level2"]
                true_l3 = row["level3"]
                is_attack = (true_l1 == "Attack")

                # L1 Prediction at Vehicle
                x_l1 = row[feature_sets["L1"]].values.reshape(1, -1)
                pred_l1 = l1_clf.predict(x_l1)[0]
                probs_l1 = l1_clf.predict_proba(x_l1)[0]
                conf_l1 = float(np.max(probs_l1))

                is_fn = (is_attack and pred_l1 == "Benign")

                # Routing decision
                mode = dashboard.current_mode
                if mode == "Flat":
                    dest_node = "Cloud"
                    dest_pos = INTERSECTION_NODES["Cloud"]["pos"]
                    route_name = f"{veh_id} -> Cloud"
                    final_label = true_l3
                elif mode == "Fixed":
                    if pred_l1 == "Benign":
                        dest_node = "vehicle"
                        dest_pos = veh_pos
                        route_name = f"{veh_id} [Benign Exit]"
                        final_label = "Benign"
                    else:
                        dest_node = "RSU"
                        dest_pos = INTERSECTION_NODES["RSU"]["pos"]
                        route_name = f"{veh_id} -> RSU"
                        final_label = "Attack"
                else:  # RL Mode
                    if pred_l1 == "Benign" and conf_l1 > 0.90:
                        dest_node = "vehicle"
                        dest_pos = veh_pos
                        route_name = f"RL: {veh_id} [Benign Exit]"
                        final_label = "Benign"
                    elif queue_counts.get("RSU", 0) > 8:
                        dest_node = "Edge"
                        dest_pos = INTERSECTION_NODES["Edge"]["pos"]
                        route_name = f"RL: {veh_id} -> Edge (Bypass RSU)"
                        final_label = true_l3
                    else:
                        dest_node = "RSU"
                        dest_pos = INTERSECTION_NODES["RSU"]["pos"]
                        route_name = f"RL: {veh_id} -> RSU"
                        final_label = true_l2 if is_attack else "Benign"

                # Light up link & animate packet POI
                if dest_node != "vehicle":
                    link_color = (231, 76, 60) if is_attack else (46, 204, 113)
                    bridge.light_up_link(veh_pos, dest_pos, f"{veh_id}_{dest_node}", link_color)
                    bridge.spawn_packet_animation(
                        packet_id=str(packet_id_seq),
                        start_pos=veh_pos,
                        end_pos=dest_pos,
                        label_class=final_label,
                        is_attack=is_attack,
                        is_fn=is_fn,
                        total_steps=5,
                    )
                    bridge.show_decision_label(dest_pos[0], dest_pos[1], route_name)

                # Update vehicle color (yellow for normal, red for attacker)
                bridge.update_vehicle_color(veh_id, final_label)

                # Queue counts update
                queue_counts[dest_node] = queue_counts.get(dest_node, 0) + (2 if dashboard.attack_burst_active else 1)
                for qk in queue_counts:
                    queue_counts[qk] = max(0, queue_counts[qk] - 1)

                dashboard.stats["total_packets"] += 1
                if is_attack:
                    if not is_fn:
                        dashboard.stats["attacks_detected"] += 1
                    else:
                        dashboard.stats["false_negatives"] += 1

                dashboard.stats["avg_latency"] = 2.0 if dest_node == "RSU" else (15.0 if dest_node == "Edge" else 60.0)
                dashboard.stats["avg_hops"] = 1 if dest_node != "vehicle" else 0
                dashboard.stats["queue_rsu"] = queue_counts.get("RSU", 0)
                dashboard.stats["queue_nearedge"] = queue_counts.get("Edge", 0)
                dashboard.stats["queue_cloud"] = queue_counts.get("Cloud", 0)
                dashboard.stats["load_rsu"] = min(100.0, queue_counts.get("RSU", 0) / 100.0 * 100.0)
                dashboard.stats["load_nearedge"] = min(100.0, queue_counts.get("Edge", 0) / 250.0 * 100.0)
                dashboard.stats["load_cloud"] = min(100.0, queue_counts.get("Cloud", 0) / 1000.0 * 100.0)

                log_entry = f"[{sim_time:5.1f}s] Pkt#{packet_id_seq:04d} True:{true_l3:<5} Route:{route_name}"
                dashboard.log_decision(log_entry)

            bridge.update_node_loadbars(queue_counts)
            dashboard.update_metrics(dashboard.stats)
            dashboard.render()

            # Take screenshot at step 70 when traffic and packet transmission are active
            if step == 70:
                try:
                    sumo_ss_path = str(RESULTS_DIR / "sumo_gui_screenshot.png")
                    traci.gui.screenshot(traci.gui.DEFAULT_VIEW, sumo_ss_path, width=1280, height=720)
                    logger.info(f"Captured sumo-gui screenshot to {sumo_ss_path}")
                except Exception as e:
                    logger.warning(f"SUMO screenshot exception: {e}")

                try:
                    dash_ss_path = str(RESULTS_DIR / "dashboard_screenshot.png")
                    dashboard.save_screenshot(dash_ss_path)
                    logger.info(f"Captured dashboard screenshot to {dash_ss_path}")
                except Exception as e:
                    logger.warning(f"Dashboard screenshot exception: {e}")

    finally:
        try:
            dashboard.close()
        except Exception:
            pass
        try:
            traci.close()
            logger.info("TraCI closed cleanly.")
        except Exception:
            pass

    logger.info("Intersection simulation completed successfully!")


if __name__ == "__main__":
    run_phase2_simulation(total_steps=140, headless=False)
