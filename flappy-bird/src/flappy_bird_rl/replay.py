"""Replay buffer utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


class ReplayBuffer:
    """Uniform replay buffer for off-policy DQN training."""

    def __init__(self, capacity: int, observation_dim: int) -> None:
        self.capacity = int(capacity)
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.observations[index] = observation
        self.next_observations[index] = next_observation
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.dones[index] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, Any]:
        indices = rng.integers(0, self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "next_observations": self.next_observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
        }
