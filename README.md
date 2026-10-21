# RL-Based Dynamic Packet Routing for a Hierarchical Intrusion Detection System in IoV

[![Python 3.10](https://img.shields.io/badge/python-3.10.11-blue.svg)](https://www.python.org/)
[![TraCI](https://img.shields.io/badge/SUMO-1.27.1-green.svg)](https://eclipse.dev/sumo/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIC--IoV2024-orange.svg)](https://www.unb.ca/cic/datasets/iov-dataset-2024.html)

**Final-Year Engineering Project & Research Thesis Implementation**  
Deep Q-Network (DQN) Dynamic Packet Routing for Hierarchical Intrusion Detection in the Internet of Vehicles, visualised live in Eclipse SUMO with a synchronized companion telemetry dashboard.

---

## 1. Executive Summary & Research Novelty

Hierarchical Intrusion Detection Systems (HIDS) for Connected and Autonomous Vehicles (CAVs) distribute anomaly detection across multiple computational tiers: Vehicle On-Board Units (ECUs), Roadside Units (RSUs), Near-Edge Computing Nodes, and the Central Cloud.

### The Research Gap (Base Paper Limitation)
In the base paper (*"A Scalable Hierarchical Intrusion Detection System for Internet of Vehicles"*, IEEE Internet of Things Journal, 2025):
- Packets are forced through a **rigid sequential hierarchy** ($V \to RSU \to NearEdge \to Cloud$).
- Under normal driving conditions, forwarding benign packets across unnecessary intermediate hops wastes scarce V2X bandwidth.
- Under **Denial-of-Service (DoS) burst floods**, intermediate RSU queues saturate rapidly, causing severe buffer bloat, latency spikes ($>100\text{ ms}$), and catastrophic packet drops.

### Our Proposed Solution: RL Dynamic Packet Routing
We introduce a **Deep Q-Network (DQN) routing agent** that dynamically selects the optimal next hop for every packet:
1. **Confidence-Aware Early Exit:** Packets classified as Benign at Level 1 (Vehicle) with high confidence are immediately accepted locally, bypassing all wireless hops.
2. **Dynamic Edge/Cloud Bypass:** When RSU queues experience congestion or attack floods, the agent dynamically switches transmission to direct 5G NR cellular bypass links ($V \to NearEdge$ or $V \to Cloud$).
3. **Observation Space Decoupling:** The RL agent observes *only* classifier output distributions (probabilities, max confidence, Shannon entropy), node queue load ratios, and link latencies. It **never** receives raw CAN payload bits or ground-truth labels, preventing data leakage.

---

## 2. Dataset Limitations & Empirical Findings (CIC-IoV2024)

A rigorous empirical audit of the **CIC-IoV2024** binary CAN bus dataset (1,408,219 records) revealed key foundational limitations:

1. **Only 41 Unique Attack Patterns in Full Dataset:**  
   Across all 1.4M rows, the five attack classes contain only **41 unique 153-bit binary patterns**:
   - `DoS`: 21 unique patterns (74,663 records)
   - `RPM`: 10 unique patterns (54,900 records)
   - `Speed`: 5 unique patterns (24,951 records)
   - `SW` (Steering Wheel): 3 unique patterns (19,977 records)
   - `Gas`: 2 unique patterns (9,991 records)
   - *(The `Benign` class contains 3,547 unique patterns across 1,223,737 records).*

2. **100% Train-Test Duplicate Twin Overlap for Attacks:**  
   Under standard stratified 70/15/15 splitting, **100.00% of test attack rows have an exact duplicate twin in the training set**. Zero novel attack patterns exist in the test split.

3. **Pattern Memorisation vs. Generalisation Ceiling:**  
   Near-perfect classifier test accuracy ($>99.9\%$) reflects exact bit-pattern memorization.
   In our **Leave-Pattern-Out (LPO)** experiment (holding out specific bit patterns during training):
   - `DoS`: generalizes across unseen patterns (100% recall).
   - `RPM`: recall plummets from **100.00%** to **24.86%** (a **75.14% performance drop**).
   - **Conclusion:** Novel-pattern attack generalization cannot be claimed from test split accuracy alone.

4. **153 Features vs. Paper's "152 Features":**  
   The raw dataset contains 153 binary feature columns ($17\text{ ID bits} + 136\text{ Data bits}$). The paper's claim of 152 features is documented as an unconfirmed discrepancy.

---

## 3. System Architecture & Tier Specifications

```
                       +-------------------------+
                       |       Cloud Tier        |
                       | (153 Feats, 6 Classes)  |
                       +-------------------------+
                                   ^     ^
                    Cellular Bypass|     |Core Network
                                   |     |
                       +-----------+     +-----------+
                       |    Near-Edge Node (MEC)     |
                       |   (18 Feats, 6 Classes)     |
                       +-----------------------------+
                                   ^     ^
                    Cellular Bypass|     |DSRC / C-V2X
                                   |     |
                       +-----------+     +-----------+
                       |      Roadside Unit (RSU)    |
                       | (11 Feats: Benign/DoS/Spoof)|
                       +-----------------------------+
                                      ^
                                      | DSRC / V2I
                                      |
                       +-----------------------------+
                       |    Vehicle On-Board Unit    |
                       | (11 Feats: Benign vs Attack)|
                       +-----------------------------+
```

### Network Link Latency Model `[CITED vs ASSUMED]`

| Link Hop | Interface / Protocol | Base Latency | Bandwidth | Reference Annotation |
| :--- | :--- | :--- | :--- | :--- |
| Vehicle $\to$ RSU | DSRC / IEEE 802.11p | $2.0\text{ ms}$ | $27\text{ Mbps}$ | `[CITED]` (Kenney 2011, Base Paper) |
| Vehicle $\to$ Near-Edge | 5G NR C-V2X (Direct) | $15.0\text{ ms}$ | $100\text{ Mbps}$ | `[CITED]` (3GPP Rel-16 V2X) |
| Vehicle $\to$ Cloud | 5G Cellular to Internet | $60.0\text{ ms}$ | $50\text{ Mbps}$ | `[CITED]` (Base Paper benchmark) |
| RSU $\to$ Near-Edge | Fiber Optic Backhaul | $5.0\text{ ms}$ | $1\text{ Gbps}$ | `[ASSUMED]` (Standard metro fiber) |
| Near-Edge $\to$ Cloud | Internet Core WAN | $45.0\text{ ms}$ | $10\text{ Gbps}$ | `[ASSUMED]` (Regional cloud gateway) |

---

## 4. Reinforcement Learning Dynamic Routing Formulation

- **State Space $\mathcal{S} \in \mathbb{R}^{21}$:**
  1. $s_{0..5}$: Canonical 6-dim padded probability vector $\mathbf{p} = [p_{\text{Benign}}, p_{\text{DoS}}, p_{\text{Gas}}, p_{\text{RPM}}, p_{\text{Speed}}, p_{\text{SW}}]$
  2. $s_{6}$: Maximum probability confidence $\max(\mathbf{p})$
  3. $s_{7}$: Normalized Shannon entropy $H(\mathbf{p}) / \log_2(6)$
  4. $s_{8..11}$: Current node location (one-hot: Vehicle, RSU, NearEdge, Cloud)
  5. $s_{12..15}$: Node queue load ratios $\frac{q_i}{C_i} \in [0, 1]$
  6. $s_{16..19}$: Outgoing link transmission latencies normalized
  7. $s_{20}$: Accumulated hop ratio $\frac{h}{h_{\max}}$

- **Action Space $\mathcal{A}$ (with Action Masking):**
  - $a = 0$: `ACCEPT` (Terminate routing and accept current prediction)
  - $a = 1$: `FORWARD_RSU` (Valid only from Vehicle)
  - $a = 2$: `FORWARD_NEAREDGE` (Valid from Vehicle or RSU)
  - $a = 3$: `FORWARD_CLOUD` (Valid from Vehicle, RSU, or NearEdge)

- **Reward Function:**
  $$R = R_{\text{correct}} - \lambda_{\text{lat}} \cdot T_{\text{elapsed}} - \lambda_{\text{hop}} \cdot h - \mathbb{I}_{\text{FN}} \cdot P_{\text{FN}} - \mathbb{I}_{\text{overload}} \cdot P_{\text{overload}}$$
  - Early correct benign termination bonus: $+5.0$
  - Severe false negative penalty: $-25.0$
  - Latency penalty: $-0.05 \times T_{\text{ms}}$
  - Hop penalty: $-1.0 \times h$
  - Buffer overload penalty: $-3.0$

---

## 5. Visual Simulation & SUMO Scenario

The SUMO intersection simulation (`sumo_config/intersection.sumocfg`) features:
- **Clean Aesthetic:** Light cream background (`#F4F1EA`), dark asphalt roadways (`#2C3E50`), and crisp white lane markings matching real-world urban layouts.
- **Hierarchical Infrastructure Nodes:** Positioned in the upper-right quadrant ($RSU$ in blue, $NearEdge$ in emerald green, $Cloud$ in purple).
- **Moving Packet POIs:** CAN packets dynamically animate as colored spheres moving along transmission links ($46, 204, 113$ for Benign, $231, 76, 60$ for Attack).
- **Live Node Loadbars:** Real-time queue occupancy rendered directly above each infrastructure tower.
- **Companion Telemetry Dashboard:** Non-blocking Pygame HUD displaying live rolling latency, queue utilization, detection KPI cards, and decision logs.

### Hotkeys in Live Simulation
| Hotkey | Function |
| :--- | :--- |
| `1` | Switch routing policy to **Flat Centralized Cloud** |
| `2` | Switch routing policy to **Fixed Hierarchy (Uddin et al. 2025)** |
| `3` | Switch routing policy to **RL Dynamic Routing (Proposed)** |
| `B` | Toggle **DoS Attack Burst Injection** (floods CAN bus with attack traffic) |
| `R` | **Reset** network queues, loadbars, and rolling statistics |
| `SPACE` | **Pause / Resume** simulation |
| `S` | Save high-resolution screenshot |
| `ESC` / `Q` | Graceful shutdown |

---

## 6. Project Directory Structure

```
d:/proj_finalyear/
├── config.py                 # Central configurations, latencies, seed, reward weights
├── train_classifiers.py      # Feature selection (Boruta+SHAP) and 10-fold CV classifier training
├── train_rl.py               # Headless Deep Q-Network (DQN) policy training
├── evaluate.py               # Comparative benchmark (Flat vs Fixed vs RL), HTML & LaTeX generator
├── main.py                   # Live interactive SUMO + Dashboard simulation launcher
├── build_notebook.py         # Generator script for simulation_demo.ipynb
├── simulation_demo.ipynb     # Self-contained Jupyter Notebook demonstration
├── requirements.txt          # Pinned exact package dependencies
├── data/
│   ├── raw/                  # Raw CIC-IoV2024 CSV dataset
│   └── cache/                # Stratified 70/15/15 parquet partitions (train, val, test)
├── models/
│   ├── feature_sets.json     # Selected feature names per hierarchy tier
│   ├── L1_vehicle.joblib     # Frozen Level 1 Vehicle classifier
│   ├── L2_rsu.joblib         # Frozen Level 2 RSU classifier
│   ├── L3_nearedge.joblib    # Frozen Level 3 Near-Edge classifier
│   ├── Cloud_classifier.joblib # Frozen Cloud classifier
│   └── rl_policy.pt          # Trained PyTorch DQN weights
├── results/
│   ├── metrics.csv           # Comparative benchmark summary
│   ├── thesis_table.tex      # Publication-ready LaTeX table
│   ├── report.html           # Standalone interactive HTML report
│   ├── policy_comparison_bars.png
│   ├── latency_vs_burst.png
│   ├── route_usage_pie.png
│   ├── policy_confusion_matrices.png
│   ├── latency_hops_histograms.png
│   └── rl_learning_curves.png
├── src/
│   ├── classifiers.py        # Hierarchical classifier wrapper & padded probability mapping
│   ├── network_model.py      # Discrete-event network simulation & queue dynamics
│   ├── env.py                # Gymnasium-style packet routing environment with action masking
│   ├── rl_agent.py           # PyTorch DQN with target network and experience replay
│   ├── baselines.py          # Flat and Fixed routing baseline implementations
│   ├── dashboard.py          # Companion Pygame telemetry dashboard
│   └── sumo_bridge.py        # TraCI visualizer, moving POIs, loadbars, and link animations
└── sumo_config/              # Clean 4-way intersection SUMO network, routes, and viewport
```

---

## 7. How to Run the Project

### Prerequisites
- Python 3.10.11 standalone environment: `D:\proj_finalyear\.venv`
- Eclipse SUMO 1.27.1 installed at `D:\sumo`

### 1. Interactive Live Simulation
```powershell
.\.venv\Scripts\python.exe main.py
```
*Run with `--headless` if running in an automated environment without a display window.*

### 2. Comprehensive Evaluation & Thesis Report Generation
```powershell
.\.venv\Scripts\python.exe evaluate.py
```
*Generates `results/metrics.csv`, `results/thesis_table.tex`, `results/report.html`, and all 6 publication figures.*

### 3. Jupyter Notebook Demonstration
```powershell
.\.venv\Scripts\jupyter.exe notebook simulation_demo.ipynb
```

### 4. Retrain Models (Optional)
```powershell
# Retrain Hierarchical Classifiers (Boruta + SHAP + 10-fold Stratified CV)
.\.venv\Scripts\python.exe train_classifiers.py

# Retrain Headless DQN Routing Policy (2,500 episodes)
.\.venv\Scripts\python.exe train_rl.py
```
