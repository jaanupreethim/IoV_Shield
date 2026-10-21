"""Configuration parameters for RL-Based Hierarchical IoV IDS Simulation.
Contains paths, label schemas, hyperparameters, network parameters (cited vs assumed),
RL state definitions, and GUI visualization colors.
"""

import os
from pathlib import Path

# ==========================================
# Paths
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "binary"

# Pipeline variant. Default (unset) = original pipeline, folders: data/cache, models, results.
# PowerShell:  $env:IDS_VARIANT="dedup"   -> separate folders data/cache_dedup, models_dedup, results_dedup
# "dedup" removes duplicate CAN patterns BEFORE the train/val/test split, so no test pattern appears in training.
VARIANT = os.environ.get("IDS_VARIANT", "").strip().lower()
_SUFFIX = f"_{VARIANT}" if VARIANT else ""
DEDUP_BEFORE_SPLIT = (VARIANT == "dedup")

CACHE_DIR = PROJECT_ROOT / "data" / f"cache{_SUFFIX}"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / f"models{_SUFFIX}"
RESULTS_DIR = PROJECT_ROOT / f"results{_SUFFIX}"
LOGS_DIR = PROJECT_ROOT / "logs"
SUMO_CONFIG_DIR = PROJECT_ROOT / "sumo_config"

# SUMO HOME
SUMO_HOME = os.environ.get("SUMO_HOME", r"D:\sumo")

# Ensure required directories exist
for d in [CACHE_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, SUMO_CONFIG_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==========================================
# Random Seed
# ==========================================
RANDOM_SEED = 42

# ==========================================
# Classes & Hierarchical Schema
# ==========================================
L1_CLASSES = ["Benign", "Attack"]
L2_CLASSES = ["Benign", "DoS", "Spoofing"]
L3_CLASSES = ["Benign", "DoS", "Gas", "RPM", "Speed", "SW"]
CLOUD_CLASSES = L3_CLASSES

# Fixed length for canonical probability vector in RL state
PROB_VECTOR_DIM = len(L3_CLASSES)  # 6: [Benign, DoS, Gas, RPM, Speed, SW]

# Mapping from raw filename to hierarchical labels
FILENAME_LABEL_MAP = {
    "binary_benign.csv": ("Benign", "Benign", "Benign"),
    "binary_DoS.csv": ("Attack", "DoS", "DoS"),
    "binary_spoofing-GAS.csv": ("Attack", "Spoofing", "Gas"),
    "binary_spoofing-RPM.csv": ("Attack", "Spoofing", "RPM"),
    "binary_spoofing-SPEED.csv": ("Attack", "Spoofing", "Speed"),
    "binary_spoofing-STEERING_WHEEL.csv": ("Attack", "Spoofing", "SW"),
}

# Mapping from L3 ground truth to higher level truths
L3_TO_L2 = {
    "Benign": "Benign",
    "DoS": "DoS",
    "Gas": "Spoofing",
    "RPM": "Spoofing",
    "Speed": "Spoofing",
    "SW": "Spoofing",
}

L3_TO_L1 = {
    "Benign": "Benign",
    "DoS": "Attack",
    "Gas": "Attack",
    "RPM": "Attack",
    "Speed": "Attack",
    "SW": "Attack",
}

RAW_LABEL_COLS = ["label", "category", "specific_class"]

# Target number of selected features per level
N_FEATURES_L1 = 11
N_FEATURES_L2 = 11
N_FEATURES_L3 = 18

# Subsample size for Boruta and feature selection
BORUTA_SUBSAMPLE_SIZE = 150000

# ==============================================================================
# Network Topology & Latency Model (ms)
# All parameters annotated as [CITED] or [ASSUMED] for sensitivity analysis.
# ==============================================================================
LINK_BASE_LATENCIES = {
    # [CITED]: IEEE 802.11p / DSRC short-range V2I standard latency (1-3 ms, typical 2 ms; IEEE 802.11p-2010 / Uddin et al. 2025 Sec. I)
    ("vehicle", "rsu"): 2.0,
    # [CITED]: 3GPP Rel-16 5G NR C-V2X Uu interface edge user-plane latency (10-20 ms; 3GPP TR 38.885)
    ("vehicle", "nearedge"): 15.0,
    # [ASSUMED]: 5G cellular core + public Internet backhaul round-trip to centralized cloud (50-80 ms range)
    ("vehicle", "cloud"): 60.0,
    # [ASSUMED]: Local fiber-optic backhaul between roadside unit and edge computing facility (3-7 ms range)
    ("rsu", "nearedge"): 5.0,
    # [ASSUMED]: Metropolitan backhaul network connecting RSU to regional cloud center (35-55 ms range)
    ("rsu", "cloud"): 45.0,
    # [ASSUMED]: High-speed optical inter-datacenter backhaul from near-edge to cloud core (30-50 ms range)
    ("nearedge", "cloud"): 40.0,
}

# Node compute processing capacity multiplier relative to vehicle ECU
# [ASSUMED]: Plausible edge-to-cloud hardware scaling:
# Vehicle: Automotive MCU (e.g. ARM Cortex-R / Renesas RH850) -> 1.0x baseline
# RSU: Embedded edge computer (e.g. NVIDIA Jetson Orin / Intel Atom) -> 4.0x
# Near-Edge: Micro-datacenter rack server (e.g. Intel Xeon-D / AMD EPYC Embedded) -> 8.0x
# Cloud: High-throughput cloud cluster (e.g. AMD EPYC / NVIDIA A100 multi-core) -> 32.0x
NODE_PROCESSING_MULTIPLIER = {
    "vehicle": 1.0,
    "rsu": 4.0,
    "nearedge": 8.0,
    "cloud": 32.0,
}

# Maximum packet queue buffer length per node
# [ASSUMED]: Based on typical buffer sizing in vehicular edge computing architectures:
NODE_MAX_QUEUE = {
    "vehicle": 20,       # Constrained in-vehicle ECU buffer
    "rsu": 200,          # Roadside unit buffer
    "nearedge": 500,     # Near-edge server buffer
    "cloud": 2000,       # Centralized cloud queue
}

# Base service rate in packets per second (derived from measured inference times and hardware multipliers)
# Vehicle ECU: ~30 pkts/s (based on ~31ms measured RF inference time)
# RSU: ~120 pkts/s
# Near-edge: ~250 pkts/s
# Cloud: ~1000 pkts/s
NODE_BASE_SERVICE_RATE = {
    "vehicle": 30.0,
    "rsu": 120.0,
    "nearedge": 250.0,
    "cloud": 1000.0,
}

# Background (non-IDS) traffic as a FRACTION OF EACH NODE'S SERVICE RATE. [ASSUMED]
# The old absolute rates (2 pkt/s at RSU vs 120 pkt/s capacity) meant a DoS burst never built a queue,
# so dynamic routing had nothing to optimise. Burst multiplier (x4, network_model.py) pushes RSU to ~120% load.
NODE_BG_UTILIZATION = {
    "vehicle": 0.30,
    "rsu": 0.10,       # normal traffic: RSU / near-edge lightly loaded (a few ms of queueing)
    "nearedge": 0.10,
    "cloud": 0.05,
}

# Background load DURING AN ATTACK BURST (fraction of service rate). [ASSUMED]
# A burst loads the whole infrastructure, not just RSU/edge: RSU and near-edge run hot but stable
# (~80-90%), and the shared cloud also gets busier. Previously the RSU was driven to 120% (permanently full,
# never usable) and the cloud stayed idle (always the best choice), so RL sent every packet to the cloud.
NODE_BG_UTILIZATION_BURST = {
    "vehicle": 0.30,
    "rsu": 0.90,
    "nearedge": 0.80,
    "cloud": 0.50,
}

# Sensitivity Analysis parameter ranges for evaluation phase (Phase 4)
SENSITIVITY_RANGES = {
    "latency_scaling": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],  # 50% to 200% base latency
    "queue_scaling": [0.5, 1.0, 2.0],
    "burst_multiplier": [1.0, 2.0, 5.0, 10.0],            # DoS burst intensity factors
}

# ==============================================================================
# RL Environment & Reward Weights
# STRICT RULE: Observation space uses ONLY classifier outputs (probs, conf, entropy),
# node status (loads, queue lengths), and link status (latencies, hops).
# NEVER raw features and NEVER ground-truth labels!
# ==============================================================================
REWARD_BENIGN_CORRECT = 5.0           # Reward early benign exit at any layer
REWARD_ATTACK_L1 = 2.0                # L1 binary attack detection (low — generic)
REWARD_ATTACK_L2 = 8.0                # L2 broad attack categorization (DoS vs Spoofing)
REWARD_ATTACK_L3_CLOUD = 18.0         # L3 or Cloud exact attack subtype (DoS/Gas/RPM/Speed/SW)
REWARD_FALSE_NEGATIVE = -25.0         # Heavy penalty if attack is labelled Benign
REWARD_WRONG_ATTACK = -4.0            # Penalty for incorrect attack subtype
PENALTY_LATENCY_COEFF = 0.05          # Penalty per ms of total latency (0.01 was too weak: a 500 ms RSU queue cost only 5 reward vs +10 for full resolution)
PENALTY_HOP_COEFF = 0.15              # Penalty per network hop (reduced)
RSU_EARLY_EXIT_BONUS = 3.0            # extra reward when an attack is FULLY resolved at the RSU (DoS) instead of going deeper
PENALTY_OVERLOAD = 2.0                # Penalty for forwarding to node with queue > 80%
PENALTY_EXCESSIVE_HOPS = 0.5

MAX_HOPS = 4

# ==============================================================================
# Cloud tier (training / logging / periodic model updates - it does NOT classify)
# ==============================================================================
CLOUD_UPDATE_INTERVAL = 1000      # reported packets between automatic model updates (scaled-down "hourly" update)
CLOUD_MIN_NEW_SAMPLES = 200       # minimum distinct reported samples before an update is allowed
CLOUD_REPLAY_PER_CLASS = 30000    # rows per class kept from the original training data during retraining
CLOUD_RL_FINETUNE_EPISODES = 600  # RL fine-tune episodes per update

# Actions
ACTION_ACCEPT = 0
ACTION_FORWARD_RSU = 1
ACTION_FORWARD_NEAREDGE = 2
ACTION_FORWARD_CLOUD = 3
ACTION_NAMES = ["ACCEPT", "FORWARD_RSU", "FORWARD_NEAREDGE", "FORWARD_CLOUD"]
NUM_ACTIONS = 4

# Node Indices
NODE_NAMES = ["vehicle", "rsu", "nearedge", "cloud"]
NODE_TO_IDX = {name: idx for idx, name in enumerate(NODE_NAMES)}

# ==========================================
# Colors (RGB tuples & SUMO RGBA)
# ==========================================
COLORS_RGB = {
    "grey": (128, 128, 128),      # New / Unclassified
    "blue": (30, 144, 255),       # Processing at node
    "green": (46, 204, 113),      # Benign
    "orange": (243, 156, 18),     # L1 Suspected Attack
    "red": (231, 76, 60),         # DoS
    "purple": (155, 89, 182),     # Generic Spoofing (L2)
    "yellow": (241, 196, 15),     # Spoofing - Gas
    "pink": (255, 105, 180),      # Spoofing - SW
    "cyan": (26, 188, 156),       # Spoofing - Speed
    "brown": (160, 82, 45),       # Spoofing - RPM
    "black": (20, 20, 20),        # Misclassified / FN Flash
}

CLASS_TO_COLOR = {
    "Benign": COLORS_RGB["green"],
    "Attack": COLORS_RGB["orange"],
    "DoS": COLORS_RGB["red"],
    "Spoofing": COLORS_RGB["purple"],
    "Gas": COLORS_RGB["yellow"],
    "SW": COLORS_RGB["pink"],
    "Speed": COLORS_RGB["cyan"],
    "RPM": COLORS_RGB["brown"],
}
