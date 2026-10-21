"""Plain DQN Agent with Target Network in PyTorch.
Implements uniform replay buffer, epsilon-greedy exploration,
action masking, and target network synchronization.
"""

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import RANDOM_SEED


class QNetwork(nn.Module):
    """Feedforward Q-Network estimating Q(s, a) for all valid routing actions."""

    def __init__(self, state_dim: int = 21, action_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Uniform experience replay buffer storing (s, a, r, s', done, mask)."""

    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, next_mask: np.ndarray):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones, next_masks = zip(*[self.buffer[i] for i in indices])
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
            torch.BoolTensor(np.array(next_masks)),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Standard DQN Agent with target network and action masking."""

    def __init__(
        self,
        state_dim: int = 21,
        action_dim: int = 4,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.997,
        buffer_capacity: int = 20000,
        batch_size: int = 64,
        target_update_freq: int = 150,
        random_seed: int = RANDOM_SEED,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_step_count = 0

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state: np.ndarray, action_mask: np.ndarray, greedy: bool = False) -> int:
        """Selects an action using epsilon-greedy policy with strict action masking."""
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return 0  # fallback to ACCEPT

        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_actions))

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor).squeeze(0).numpy()

        # Mask invalid actions
        masked_q = np.full_like(q_values, -1e8)
        masked_q[valid_actions] = q_values[valid_actions]

        return int(np.argmax(masked_q))

    def update(self) -> Optional[float]:
        """Samples mini-batch and performs one gradient descent step."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(self.batch_size)

        # Current Q-values: Q(s, a)
        curr_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + gamma * max_a' Q_target(s', a') with action masking
        with torch.no_grad():
            next_q_all = self.target_net(next_states)
            # Mask invalid actions on next state
            next_q_all[~next_masks] = -1e8
            max_next_q, _ = torch.max(next_q_all, dim=1)
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = self.criterion(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(loss.item())

    def save(self, filepath: Path):
        """Serializes model weights."""
        torch.save({
            "q_net_state_dict": self.q_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "epsilon": self.epsilon,
        }, filepath)

    def load(self, filepath: Path):
        """Loads trained weights."""
        checkpoint = torch.load(filepath, map_location="cpu")
        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.q_net.eval()
