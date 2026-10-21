"""Non-blocking Pygame companion dashboard window.
Displays live counters, rolling real-time charts, color legend, scrolling decision log,
and receives interactive keyboard controls.
"""

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pygame

from config import (
    CLASS_TO_COLOR,
    COLORS_RGB,
    RESULTS_DIR,
)


class CompanionDashboard:
    """Real-time Pygame companion UI rendered alongside sumo-gui."""

    def __init__(self, width: int = 1000, height: int = 700):
        pygame.init()
        pygame.display.set_caption("IoV Dynamic Routing HIDS - Companion Dashboard")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.font_header = pygame.font.SysFont("Segoe UI", 14, bold=True)
        self.font_text = pygame.font.SysFont("Segoe UI", 12)
        self.font_mono = pygame.font.SysFont("Consolas", 11)

        # State & Mode
        self.current_mode = "RL"  # "Flat", "Fixed", "RL"
        self.packet_rate = 1.0  # multiplier
        self.attack_burst_active = False
        self.paused = False
        self.running = True

        # Metrics history for rolling graphs (last 100 points)
        self.hist_latency: deque = deque(maxlen=100)
        self.hist_load_rsu: deque = deque(maxlen=100)
        self.hist_load_nearedge: deque = deque(maxlen=100)
        self.hist_load_cloud: deque = deque(maxlen=100)
        self.hist_accuracy: deque = deque(maxlen=100)

        # Scrolling log (up to 12 items)
        self.decision_logs: deque = deque(maxlen=12)

        # Accumulated Stats
        self.stats = {
            "total_packets": 0,
            "attacks_detected": 0,
            "false_negatives": 0,
            "false_positives": 0,
            "avg_latency": 0.0,
            "avg_hops": 0.0,
            "queue_rsu": 0,
            "queue_nearedge": 0,
            "queue_cloud": 0,
            "load_rsu": 0.0,
            "load_nearedge": 0.0,
            "load_cloud": 0.0,
        }

    def handle_events(self) -> Dict[str, Any]:
        """Polls Pygame events non-blockingly and returns user control actions."""
        action_triggered = {}
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                action_triggered["quit"] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    action_triggered["quit"] = True
                elif event.key == pygame.K_1:
                    self.current_mode = "Flat"
                    action_triggered["mode_change"] = "Flat"
                elif event.key == pygame.K_2:
                    self.current_mode = "Fixed"
                    action_triggered["mode_change"] = "Fixed"
                elif event.key == pygame.K_3:
                    self.current_mode = "RL"
                    action_triggered["mode_change"] = "RL"
                elif event.key == pygame.K_b:
                    self.attack_burst_active = not self.attack_burst_active
                    action_triggered["burst_toggle"] = self.attack_burst_active
                elif event.key == pygame.K_UP:
                    self.packet_rate = min(5.0, round(self.packet_rate + 0.5, 1))
                    action_triggered["rate_change"] = self.packet_rate
                elif event.key == pygame.K_DOWN:
                    self.packet_rate = max(0.5, round(self.packet_rate - 0.5, 1))
                    action_triggered["rate_change"] = self.packet_rate
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    action_triggered["pause_toggle"] = self.paused
                elif event.key == pygame.K_r:
                    self.reset_stats()
                    action_triggered["reset"] = True
                elif event.key == pygame.K_u:
                    action_triggered["cloud_update"] = True
                elif event.key == pygame.K_s:
                    self.save_screenshot()
                    action_triggered["screenshot"] = True

        return action_triggered

    def reset_stats(self):
        """Resets counters and histories."""
        for k in ["total_packets", "attacks_detected", "false_negatives", "false_positives"]:
            self.stats[k] = 0
        self.stats["avg_latency"] = 0.0
        self.stats["avg_hops"] = 0.0
        self.hist_latency.clear()
        self.hist_load_rsu.clear()
        self.hist_load_nearedge.clear()
        self.hist_load_cloud.clear()
        self.hist_accuracy.clear()
        self.decision_logs.append(">> Statistics reset.")

    def log_decision(self, msg: str):
        """Appends a new decision log line."""
        self.decision_logs.append(msg)

    def update_metrics(self, new_stats: Dict[str, Any]):
        """Updates internal KPI statistics and rolling histories."""
        self.stats.update(new_stats)
        self.hist_latency.append(self.stats.get("avg_latency", 0.0))
        self.hist_load_rsu.append(self.stats.get("load_rsu", 0.0))
        self.hist_load_nearedge.append(self.stats.get("load_nearedge", 0.0))
        self.hist_load_cloud.append(self.stats.get("load_cloud", 0.0))

        tot = self.stats["total_packets"]
        fn = self.stats["false_negatives"]
        fp = self.stats["false_positives"]
        acc = max(0.0, (tot - fn - fp) / max(1, tot) * 100.0)
        self.hist_accuracy.append(acc)

    def render(self):
        """Renders the dashboard frame."""
        self.screen.fill((24, 28, 36))  # Dark professional theme

        # Header Banner
        header_surf = self.font_title.render("Hierarchical IoV IDS: Live Routing Dashboard", True, (240, 240, 240))
        self.screen.blit(header_surf, (20, 15))

        mode_col = (46, 204, 113) if self.current_mode == "RL" else ((52, 152, 219) if self.current_mode == "Fixed" else (243, 156, 18))
        mode_surf = self.font_header.render(f"ACTIVE POLICY: {self.current_mode} Mode", True, mode_col)
        self.screen.blit(mode_surf, (550, 18))

        rate_surf = self.font_text.render(f"Rate: {self.packet_rate:.1f}x | Burst: {'ACTIVE' if self.attack_burst_active else 'OFF'}", True, (231, 76, 60) if self.attack_burst_active else (200, 200, 200))
        self.screen.blit(rate_surf, (780, 18))

        # Divider line
        pygame.draw.line(self.screen, (50, 56, 70), (20, 48), (980, 48), 2)

        # 1. Left Column: KPI Cards (x=20 to 300)
        self._render_kpi_cards(x=20, y=60)

        # 2. Middle Column: Rolling Graphs (x=330 to 670)
        self._render_rolling_graphs(x=330, y=60)

        # 3. Right Column: Color Legend & Controls (x=700 to 980)
        self._render_legend_and_controls(x=700, y=60)

        # 4. Bottom Panel: Scrolling Routing Decision Log (y=470 to 680)
        self._render_decision_log(x=20, y=470, w=960, h=210)

        pygame.display.flip()
        self.clock.tick(30)

    def _render_kpi_cards(self, x: int, y: int):
        """Draws key metric stat boxes."""
        cards = [
            ("Packets Processed", f"{self.stats['total_packets']:,}", (220, 220, 220)),
            ("Attacks Detected", f"{self.stats['attacks_detected']:,}", (231, 76, 60)),
            ("False Negatives (FN)", f"{self.stats['false_negatives']:,}", (255, 75, 75)),
            ("False Positives (FP)", f"{self.stats['false_positives']:,}", (241, 196, 15)),
            ("Average Latency", f"{self.stats['avg_latency']:.2f} ms", (52, 152, 219)),
            ("Average Hops", f"{self.stats['avg_hops']:.2f}", (155, 89, 182)),
            ("RSU Queue Load", f"{self.stats['load_rsu']:.1f}% ({self.stats['queue_rsu']})", (46, 204, 113) if self.stats['load_rsu'] < 70 else (231, 76, 60)),
            ("Near-Edge Load", f"{self.stats['load_nearedge']:.1f}% ({self.stats['queue_nearedge']})", (46, 204, 113) if self.stats['load_nearedge'] < 70 else (231, 76, 60)),
            ("Cloud Node Load", f"{self.stats['load_cloud']:.1f}% ({self.stats['queue_cloud']})", (46, 204, 113) if self.stats['load_cloud'] < 70 else (231, 76, 60)),
        ]

        curr_y = y
        for label, val, color in cards:
            pygame.draw.rect(self.screen, (32, 38, 48), (x, curr_y, 290, 40), border_radius=4)
            lbl_surf = self.font_text.render(label, True, (170, 175, 185))
            val_surf = self.font_header.render(val, True, color)
            self.screen.blit(lbl_surf, (x + 10, curr_y + 4))
            self.screen.blit(val_surf, (x + 10, curr_y + 20))
            curr_y += 45

    def _render_rolling_graphs(self, x: int, y: int):
        """Draws rolling line plots for latency and node loads."""
        # 1. Latency Chart (h=180)
        pygame.draw.rect(self.screen, (32, 38, 48), (x, y, 350, 180), border_radius=4)
        t_surf = self.font_header.render("End-to-End Latency (ms)", True, (220, 220, 220))
        self.screen.blit(t_surf, (x + 10, y + 8))
        self._draw_line_chart(self.hist_latency, x + 15, y + 35, 320, 130, max_val=100.0, line_col=(52, 152, 219))

        # 2. Node Load Comparison Chart (h=190)
        y2 = y + 195
        pygame.draw.rect(self.screen, (32, 38, 48), (x, y2, 350, 195), border_radius=4)
        t2_surf = self.font_header.render("Node Queue Utilization (%)", True, (220, 220, 220))
        self.screen.blit(t2_surf, (x + 10, y2 + 8))
        self._draw_line_chart(self.hist_load_rsu, x + 15, y2 + 35, 320, 140, max_val=100.0, line_col=(46, 204, 113))
        self._draw_line_chart(self.hist_load_nearedge, x + 15, y2 + 35, 320, 140, max_val=100.0, line_col=(155, 89, 182))
        self._draw_line_chart(self.hist_load_cloud, x + 15, y2 + 35, 320, 140, max_val=100.0, line_col=(52, 152, 219))

    def _draw_line_chart(self, data: deque, x: int, y: int, w: int, h: int, max_val: float, line_col: Tuple[int, int, int]):
        """Helper to draw a rolling line plot."""
        if len(data) < 2:
            return
        points = []
        pts_list = list(data)
        n = len(pts_list)
        for i, val in enumerate(pts_list):
            px = x + int(i / max(1, n - 1) * w)
            norm = min(1.0, max(0.0, val / max_val))
            py = y + h - int(norm * h)
            points.append((px, py))
        pygame.draw.lines(self.screen, line_col, False, points, 2)

    def _render_legend_and_controls(self, x: int, y: int):
        """Draws packet classification color legend and keyboard shortcuts."""
        # Legend Panel
        pygame.draw.rect(self.screen, (32, 38, 48), (x, y, 280, 200), border_radius=4)
        leg_title = self.font_header.render("Packet State Colors", True, (220, 220, 220))
        self.screen.blit(leg_title, (x + 10, y + 8))

        legend_items = [
            ("Benign", (46, 204, 113)),
            ("Suspected (L1)", (243, 156, 18)),
            ("DoS Attack", (231, 76, 60)),
            ("Spoofing (L2)", (155, 89, 182)),
            ("Gas Spoofing", (241, 196, 15)),
            ("Speed Spoofing", (26, 188, 156)),
            ("RPM Spoofing", (160, 82, 45)),
            ("False Negative Flash", (20, 20, 20)),
        ]

        ly = y + 32
        for name, col in legend_items:
            pygame.draw.circle(self.screen, col, (x + 18, ly + 6), 5)
            if col == (20, 20, 20):
                pygame.draw.circle(self.screen, (220, 220, 220), (x + 18, ly + 6), 6, 1)  # border
            item_surf = self.font_text.render(name, True, (200, 200, 200))
            self.screen.blit(item_surf, (x + 32, ly))
            ly += 20

        # Controls Panel
        y_ctrl = y + 215
        pygame.draw.rect(self.screen, (32, 38, 48), (x, y_ctrl, 280, 190), border_radius=4)
        c_title = self.font_header.render("Keyboard Controls", True, (220, 220, 220))
        self.screen.blit(c_title, (x + 10, y_ctrl + 8))

        controls = [
            ("1 / 2 / 3", "Switch mode: Flat / Fixed / RL"),
            ("UP / DOWN", "Adjust packet emission rate"),
            ("B", "Toggle DoS attack burst"),
            ("SPACE", "Pause / Resume simulation"),
            ("R", "Reset statistics counters"),
            ("S", "Capture dashboard screenshot"),
            ("U", "Cloud: retrain + deploy models"),
            ("ESC", "Quit simulation safely"),
        ]

        cy = y_ctrl + 32
        for key_text, desc_text in controls:
            k_surf = self.font_mono.render(f"[{key_text}]", True, (241, 196, 15))
            d_surf = self.font_text.render(desc_text, True, (190, 195, 205))
            self.screen.blit(k_surf, (x + 10, cy))
            self.screen.blit(d_surf, (x + 95, cy))
            cy += 18

    def _render_decision_log(self, x: int, y: int, w: int, h: int):
        """Draws the scrolling terminal log of routing decisions."""
        pygame.draw.rect(self.screen, (20, 24, 30), (x, y, w, h), border_radius=4)
        pygame.draw.rect(self.screen, (50, 56, 70), (x, y, w, h), 1, border_radius=4)

        t_surf = self.font_header.render("Live Routing Decisions & Event Log", True, (220, 220, 220))
        self.screen.blit(t_surf, (x + 12, y + 8))

        log_y = y + 32
        for line in self.decision_logs:
            col = (46, 204, 113) if "RL:" in line else ((243, 156, 18) if "BURST" in line else ((255, 75, 75) if "FN:" in line else (180, 185, 195)))
            line_surf = self.font_mono.render(line, True, col)
            self.screen.blit(line_surf, (x + 15, log_y))
            log_y += 14

    def save_screenshot(self, filename: Optional[str] = None):
        """Saves current dashboard frame as PNG."""
        if filename is None:
            filename = str(RESULTS_DIR / "dashboard_screenshot.png")
        pygame.image.save(self.screen, filename)
        self.decision_logs.append(f">> Screenshot saved to {filename}")

    def close(self):
        """Closes Pygame gracefully."""
        pygame.quit()
