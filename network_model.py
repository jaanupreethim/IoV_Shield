"""Discrete-time network queuing and latency simulation module.
Models node compute capacities, FIFO queues, service rates,
transmission latencies, and traffic burst dynamics.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import (
    LINK_BASE_LATENCIES,
    NODE_BASE_SERVICE_RATE,
    NODE_BG_UTILIZATION,
    NODE_MAX_QUEUE,
    NODE_NAMES,
    NODE_PROCESSING_MULTIPLIER,
    NODE_TO_IDX,
    RANDOM_SEED,
)


class NetworkNode:
    """Represents a computing node (Vehicle ECU, RSU, Near-Edge, or Cloud)

    with finite queue buffer and processing throughput.
    """

    def __init__(self, name: str):
        self.name = name
        self.max_queue = NODE_MAX_QUEUE.get(name, 100)
        self.base_service_rate = NODE_BASE_SERVICE_RATE.get(name, 50.0)  # pkts/s
        self.queue_length = 0
        self.total_processed = 0
        self.total_dropped = 0
        self.hardware_multiplier = NODE_PROCESSING_MULTIPLIER.get(name, 1.0)

    @property
    def load_ratio(self) -> float:
        """Returns current queue utilization in [0.0, 1.0]."""
        return min(1.0, self.queue_length / max(1, self.max_queue))

    def enqueue(self, num_packets: int = 1) -> int:
        """Attempts to enqueue packets; drops excess if queue full."""
        available_space = self.max_queue - self.queue_length
        admitted = min(num_packets, max(0, available_space))
        dropped = num_packets - admitted
        self.queue_length += admitted
        self.total_dropped += dropped
        return admitted

    def step(self, delta_time: float = 0.1) -> int:
        """Processes packets according to service rate over delta_time."""
        packets_to_process = int(self.base_service_rate * delta_time)
        processed = min(self.queue_length, packets_to_process)
        self.queue_length -= processed
        self.total_processed += processed
        return processed

    def get_queuing_delay_ms(self) -> float:
        """Estimated queuing delay in milliseconds."""
        if self.base_service_rate <= 0:
            return 0.0
        return (self.queue_length / self.base_service_rate) * 1000.0


class IoVNetworkSimulation:
    """Manages the full IoV network topology, links, queues, and latency model."""

    def __init__(self, random_seed: int = RANDOM_SEED):
        self.rng = np.random.RandomState(random_seed)
        self.nodes = {name: NetworkNode(name) for name in NODE_NAMES}
        self.base_latencies = dict(LINK_BASE_LATENCIES)
        self.burst_mode: bool = False
        self.burst_multiplier: float = 4.0

    def get_link_latency_ms(self, src: str, dst: str) -> float:
        """Computes end-to-end link latency including propagation, queuing, and jitter."""
        if src == dst:
            return 0.0

        key = (src, dst) if (src, dst) in self.base_latencies else (dst, src)
        base = self.base_latencies.get(key, 25.0)

        # Target node queuing delay
        target_node = self.nodes.get(dst)
        q_delay = target_node.get_queuing_delay_ms() if target_node else 0.0

        # Random wireless jitter (5% to 15%)
        jitter = base * self.rng.uniform(0.05, 0.15)
        total_lat = base + q_delay + jitter
        return float(total_lat)

    def step(self, delta_time: float = 0.1):
        """Advances node processing queues and injects background load."""
        for node in self.nodes.values():
            node.step(delta_time)

            # Background traffic injection
            # background load scales with node capacity so a burst can actually saturate a node
            bg_rate = NODE_BG_UTILIZATION.get(node.name, 0.1) * node.base_service_rate
            if self.burst_mode and node.name in ["rsu", "nearedge"]:
                bg_rate = int(bg_rate * self.burst_multiplier)

            injected = self.rng.poisson(lam=max(0.1, bg_rate * delta_time))
            node.enqueue(injected)

    def get_node_loads(self) -> np.ndarray:
        """Returns array of queue utilization ratios for all nodes."""
        return np.array([self.nodes[name].load_ratio for name in NODE_NAMES], dtype=np.float32)

    def get_outgoing_latencies(self, current_node: str) -> np.ndarray:
        """Returns array of outgoing latencies from current_node to other nodes."""
        latencies = []
        for target in NODE_NAMES:
            if target == current_node:
                latencies.append(0.0)
            else:
                latencies.append(self.get_link_latency_ms(current_node, target) / 100.0)  # normalized
        return np.array(latencies, dtype=np.float32)

    def reset(self):
        """Resets all queues and nodes to initial state."""
        for node in self.nodes.values():
            node.queue_length = 0
            node.total_processed = 0
            node.total_dropped = 0
        self.burst_mode = False
