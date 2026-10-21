"""TraCI bridge module for SUMO visualization.
Renders clean 4-way intersection with 4 RSUs (corners), 2 Near-Edge nodes,
and 1 Cloud node. Shows animated multi-hop packet travel along each link.
Vehicles turn RED on attack, GREEN on benign.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import traci

from config import (
    CLASS_TO_COLOR,
    COLORS_RGB,
    NODE_MAX_QUEUE,
)

# =============================================================================
# Infrastructure Layout
# Intersection center at (180, 180) after netconvert offset.
# 4 RSUs placed at 4 approach roads (corners of the intersection).
# 2 Near-Edge servers serve the North and South RSU pairs.
# 1 Cloud server at upper-right.
# =============================================================================
INTERSECTION_NODES = {
    # --- 4 RSUs at the 4 road approaches ---
    "RSU_NE": {"pos": (220.0, 220.0), "type": "rsu", "color": (52, 152, 219, 255),  "label": "RSU-NE"},
    "RSU_NW": {"pos": (140.0, 220.0), "type": "rsu", "color": (52, 152, 219, 255),  "label": "RSU-NW"},
    "RSU_SE": {"pos": (220.0, 140.0), "type": "rsu", "color": (52, 152, 219, 255),  "label": "RSU-SE"},
    "RSU_SW": {"pos": (140.0, 140.0), "type": "rsu", "color": (52, 152, 219, 255),  "label": "RSU-SW"},
    # --- 2 Near-Edge (MEC) nodes ---
    "Edge_N":  {"pos": (260.0, 250.0), "type": "nearedge", "color": (39, 174, 96, 255),   "label": "Edge-N"},
    "Edge_S":  {"pos": (260.0, 110.0), "type": "nearedge", "color": (39, 174, 96, 255),   "label": "Edge-S"},
    # --- 1 Cloud server ---
    "Cloud":   {"pos": (300.0, 180.0), "type": "cloud",    "color": (142, 68, 173, 255),  "label": "Cloud"},
}

# Nearest RSU for each road quadrant (by vehicle position relative to center)
def get_nearest_rsu_id(veh_pos: Tuple[float, float]) -> str:
    """Returns nearest RSU node key based on vehicle position vs intersection center."""
    cx, cy = 180.0, 180.0
    x, y = veh_pos
    if x >= cx and y >= cy:
        return "RSU_NE"
    elif x < cx and y >= cy:
        return "RSU_NW"
    elif x >= cx and y < cy:
        return "RSU_SE"
    else:
        return "RSU_SW"

# Which Near-Edge serves which RSU
RSU_TO_EDGE = {
    "RSU_NE": "Edge_N",
    "RSU_NW": "Edge_N",
    "RSU_SE": "Edge_S",
    "RSU_SW": "Edge_S",
}


def create_dashed_line_segments(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    dash_len: float = 3.5,
    gap_len: float = 2.5,
) -> List[List[Tuple[float, float]]]:
    """Generates segments for a dashed line between two points."""
    x1, y1 = p1
    x2, y2 = p2
    total_dist = math.hypot(x2 - x1, y2 - y1)
    if total_dist == 0:
        return []
    dx = (x2 - x1) / total_dist
    dy = (y2 - y1) / total_dist
    segments = []
    curr = 0.0
    while curr < total_dist:
        end = min(curr + dash_len, total_dist)
        s_start = (x1 + dx * curr, y1 + dy * curr)
        s_end = (x1 + dx * end, y1 + dy * end)
        segments.append([s_start, s_end])
        curr += dash_len + gap_len
    return segments


class SumoVisualizerBridge:
    """Manages clean research visualization inside sumo-gui via TraCI.

    Features:
    - 4 RSU nodes (blue squares) at 4 road approaches
    - 2 Near-Edge nodes (green) serving North and South RSU pairs
    - 1 Cloud node (purple)
    - Animated multi-hop packets: V→RSU→Edge→Cloud
    - Vehicle color: GREEN=Benign, RED=Attack
    - Live node queue load-bars
    """

    def __init__(self, use_intersection: bool = True):
        self.use_intersection = use_intersection
        self.nodes_config = INTERSECTION_NODES
        # active_packets: poi_id -> list of waypoints [{start,end,step,total,color}, ...]
        self.active_packets: Dict[str, List[Dict[str, Any]]] = {}
        self.lit_links: Dict[str, int] = {}
        self.packet_counter: int = 0
        self.decision_poi_id: str = "poi_decision_text"

    # ------------------------------------------------------------------
    # Infrastructure Setup
    # ------------------------------------------------------------------
    def setup_infrastructure(self):
        """Spawns all RSU, Edge, Cloud nodes, backhaul links, and sets viewport."""
        # 1. Draw all infrastructure nodes
        for node_id, data in self.nodes_config.items():
            x, y = data["pos"]
            color = data["color"]
            label = data["label"]
            try:
                traci.poi.add(
                    poiID=f"node_{node_id}",
                    x=x, y=y,
                    color=color,
                    poiType=label,
                    layer=50,
                    width=7.0 if "RSU" in node_id else 9.0,
                    height=7.0 if "RSU" in node_id else 9.0,
                )
            except Exception:
                pass

            # Load-bar indicator (vertical bar to right of node)
            bar_shape = [(x + 9.0, y - 3.0), (x + 9.0, y + 3.0),
                         (x + 11.0, y + 3.0), (x + 11.0, y - 3.0)]
            try:
                traci.polygon.add(
                    polygonID=f"loadbar_{node_id}",
                    shape=bar_shape,
                    color=(46, 204, 113, 220),
                    fill=True,
                    layer=55,
                )
            except Exception:
                pass

        # 2. Backhaul links: RSU → serving Edge → Cloud (dashed grey)
        for rsu_id, edge_id in RSU_TO_EDGE.items():
            rsu_pos  = self.nodes_config[rsu_id]["pos"]
            edge_pos = self.nodes_config[edge_id]["pos"]
            for idx, seg in enumerate(create_dashed_line_segments(rsu_pos, edge_pos)):
                try:
                    traci.polygon.add(
                        polygonID=f"bh_{rsu_id}_{edge_id}_{idx}",
                        shape=seg,
                        color=(120, 120, 120, 150),
                        fill=False,
                        layer=20,
                        lineWidth=1.5,
                    )
                except Exception:
                    pass

        cloud_pos = self.nodes_config["Cloud"]["pos"]
        for edge_id in ["Edge_N", "Edge_S"]:
            edge_pos = self.nodes_config[edge_id]["pos"]
            for idx, seg in enumerate(create_dashed_line_segments(edge_pos, cloud_pos)):
                try:
                    traci.polygon.add(
                        polygonID=f"bh_{edge_id}_Cloud_{idx}",
                        shape=seg,
                        color=(120, 120, 120, 150),
                        fill=False,
                        layer=20,
                        lineWidth=1.5,
                    )
                except Exception:
                    pass

        # 3. Viewport – wide enough to show all 4 RSUs + Edge + Cloud
        try:
            view_id = traci.gui.DEFAULT_VIEW
            traci.gui.setOffset(view_id, 210.0, 190.0)
            traci.gui.setZoom(view_id, 130.0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Node Load-bars
    # ------------------------------------------------------------------
    def update_node_loadbars(self, queue_lengths: Dict[str, int]):
        """Updates load-bar height and colour for all nodes."""
        for node_id, data in self.nodes_config.items():
            x, y = data["pos"]
            node_type = data["type"]
            q_len = queue_lengths.get(node_id, 0)
            max_q = NODE_MAX_QUEUE.get(node_type, 100)
            ratio = min(1.0, q_len / max(1, max_q))
            h = max(2.0, ratio * 18.0)
            col = (
                (46, 204, 113, 230) if ratio < 0.4 else
                (241, 196, 15, 240) if ratio < 0.75 else
                (231, 76, 60, 255)
            )
            bar_shape = [
                (x + 9.0, y - 3.0),
                (x + 9.0, y - 3.0 + h),
                (x + 11.0, y - 3.0 + h),
                (x + 11.0, y - 3.0),
            ]
            try:
                traci.polygon.setShape(f"loadbar_{node_id}", bar_shape)
                traci.polygon.setColor(f"loadbar_{node_id}", col)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Link lighting (one-step flash)
    # ------------------------------------------------------------------
    def light_up_link(
        self,
        src_pos: Tuple[float, float],
        dst_pos: Tuple[float, float],
        link_id: str,
        color_rgb: Tuple[int, int, int],
        frames: int = 5,
    ):
        """Illuminates a communication link for a few frames."""
        poly_id = f"lit_{link_id}"
        rgba = (color_rgb[0], color_rgb[1], color_rgb[2], 220)
        try:
            traci.polygon.add(
                polygonID=poly_id,
                shape=[src_pos, dst_pos],
                color=rgba,
                fill=False,
                layer=35,
                lineWidth=2.5,
            )
            self.lit_links[poly_id] = frames
        except Exception:
            try:
                traci.polygon.setColor(poly_id, rgba)
                self.lit_links[poly_id] = frames
            except Exception:
                pass

    def step_links(self):
        """Clears expired illuminated transmission lines."""
        expired = [k for k, v in self.lit_links.items() if v <= 1]
        for poly_id in expired:
            try:
                traci.polygon.remove(poly_id)
            except Exception:
                pass
            self.lit_links.pop(poly_id, None)
        for poly_id in self.lit_links:
            if poly_id not in [e for e in expired]:
                self.lit_links[poly_id] -= 1

    # ------------------------------------------------------------------
    # Multi-hop Packet Animation
    # ------------------------------------------------------------------
    def spawn_multihop_packet(
        self,
        packet_id: str,
        waypoints: List[Tuple[float, float]],  # [veh_pos, rsu_pos, edge_pos, cloud_pos]
        label_class: str,
        is_attack: bool,
        is_fn: bool = False,
        steps_per_hop: int = 6,
    ):
        """Spawns a packet POI that travels through each waypoint sequentially.

        waypoints = ordered list of (x,y) positions for each hop destination.
        """
        if is_fn:
            col = (20, 20, 20, 255)   # Black for False Negative
        else:
            col_tuple = CLASS_TO_COLOR.get(label_class, COLORS_RGB["grey"])
            col = (col_tuple[0], col_tuple[1], col_tuple[2], 255)

        poi_id = f"pkt_{packet_id}"
        start = waypoints[0]
        try:
            traci.poi.add(
                poiID=poi_id,
                x=start[0], y=start[1],
                color=col,
                poiType=label_class,
                layer=60,
                width=3.5,
                height=3.5,
            )
        except Exception:
            pass

        # Build segment queue: (src, dst) for each hop
        hops = []
        for i in range(len(waypoints) - 1):
            hops.append({
                "start": waypoints[i],
                "end": waypoints[i + 1],
                "step": 0,
                "total_steps": steps_per_hop,
                "color": col,
            })

        self.active_packets[poi_id] = hops

        # Light up all hop links immediately
        link_colors = [
            (52, 152, 219),   # V→RSU: blue
            (39, 174, 96),    # RSU→Edge: green
            (142, 68, 173),   # Edge→Cloud: purple
        ]
        for i, hop in enumerate(hops):
            color = link_colors[i] if i < len(link_colors) else (200, 200, 200)
            if is_attack:
                color = (231, 76, 60)   # Override all links red if attack
            self.light_up_link(
                hop["start"], hop["end"],
                f"{packet_id}_hop{i}",
                color,
                frames=steps_per_hop * (i + 1),
            )

    def spawn_packet_animation(
        self,
        packet_id: str,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        label_class: str,
        is_attack: bool,
        is_fn: bool = False,
        total_steps: int = 6,
    ):
        """Backward-compatible single-hop animation (used when no RSU waypoint available)."""
        self.spawn_multihop_packet(
            packet_id=packet_id,
            waypoints=[start_pos, end_pos],
            label_class=label_class,
            is_attack=is_attack,
            is_fn=is_fn,
            steps_per_hop=total_steps,
        )

    def update_packet_animations(self):
        """Steps all packets along their multi-hop paths."""
        completed_packets = []
        for poi_id, hops in self.active_packets.items():
            if not hops:
                completed_packets.append(poi_id)
                continue

            current_hop = hops[0]
            current_hop["step"] += 1
            progress = min(1.0, current_hop["step"] / current_hop["total_steps"])

            x0, y0 = current_hop["start"]
            x1, y1 = current_hop["end"]
            curr_x = x0 + progress * (x1 - x0)
            curr_y = y0 + progress * (y1 - y0)

            try:
                traci.poi.setPosition(poi_id, curr_x, curr_y)
            except Exception:
                pass

            if progress >= 1.0:
                hops.pop(0)     # advance to next hop
                if not hops:
                    completed_packets.append(poi_id)

        for poi_id in completed_packets:
            try:
                traci.poi.remove(poi_id)
            except Exception:
                pass
            self.active_packets.pop(poi_id, None)

    # ------------------------------------------------------------------
    # Vehicle Color
    # ------------------------------------------------------------------
    def update_vehicle_color(self, veh_id: str, latest_label: str):
        """Colors vehicle: GREEN=Benign, RED=any attack type."""
        if latest_label in ("Benign",):
            col = (46, 204, 113, 255)   # Bright green
        else:
            col = (231, 76, 60, 255)    # Bright red
        try:
            traci.vehicle.setColor(veh_id, col)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Decision label
    # ------------------------------------------------------------------
    def show_decision_label(self, x: float, y: float, text: str):
        """Displays temporary decision label near a node."""
        # Only remove if it exists - removing a missing PoI makes SUMO print "Could not remove PoI".
        if getattr(self, "_decision_poi_exists", False):
            try:
                traci.poi.remove(self.decision_poi_id)
            except Exception:
                pass
            self._decision_poi_exists = False
        try:
            self._decision_poi_exists = True
            traci.poi.add(
                poiID=self.decision_poi_id,
                x=x, y=y + 10.0,
                color=(30, 30, 30, 255),
                poiType=text,
                layer=70,
                width=4.0,
                height=4.0,
            )
        except Exception:
            self._decision_poi_exists = False
