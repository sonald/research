"""Replay buffer utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


class ReplayBuffer:
    """Uniform replay buffer for off-policy DQN training."""

    def __init__(self, capacity: int, board_shape: tuple[int, ...], meta_dim: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.board = np.zeros((capacity, *board_shape), dtype=np.float32)
        self.meta = np.zeros((capacity, meta_dim), dtype=np.float32)
        self.action_mask = np.zeros((capacity, action_dim), dtype=bool)
        self.next_board = np.zeros((capacity, *board_shape), dtype=np.float32)
        self.next_meta = np.zeros((capacity, meta_dim), dtype=np.float32)
        self.next_action_mask = np.zeros((capacity, action_dim), dtype=bool)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation: dict[str, np.ndarray],
        action: int,
        reward: float,
        next_observation: dict[str, np.ndarray],
        done: bool,
    ) -> None:
        index = self.position
        self.board[index] = observation["board"]
        self.meta[index] = observation["meta"]
        self.action_mask[index] = observation["action_mask"]
        self.next_board[index] = next_observation["board"]
        self.next_meta[index] = next_observation["meta"]
        self.next_action_mask[index] = next_observation["action_mask"]
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.dones[index] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, Any]:
        indices = rng.integers(0, self.size, size=batch_size)
        return {
            "board": self.board[indices],
            "meta": self.meta[indices],
            "action_mask": self.action_mask[indices],
            "next_board": self.next_board[indices],
            "next_meta": self.next_meta[indices],
            "next_action_mask": self.next_action_mask[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
        }
