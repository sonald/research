"""Neural network modules and tensor helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


def observation_to_tensors(observation: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    board = torch.as_tensor(np.asarray(observation["board"]), dtype=torch.float32, device=device)
    meta = torch.as_tensor(np.asarray(observation["meta"]), dtype=torch.float32, device=device)
    action_mask = torch.as_tensor(np.asarray(observation["action_mask"]), dtype=torch.bool, device=device)
    if board.ndim == 3:
        board = board.unsqueeze(0)
    if meta.ndim == 1:
        meta = meta.unsqueeze(0)
    if action_mask.ndim == 1:
        action_mask = action_mask.unsqueeze(0)
    return board, meta, action_mask


def mask_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    negative_inf = torch.finfo(logits.dtype).min
    return torch.where(action_mask, logits, torch.full_like(logits, negative_inf))


def masked_argmax(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return mask_logits(logits, action_mask).argmax(dim=-1)


def masked_max(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return mask_logits(logits, action_mask).max(dim=-1).values


@dataclass
class NetworkOutput:
    logits: torch.Tensor
    value: torch.Tensor | None = None


class TetrisEncoder(nn.Module):
    """Encode board channels and meta features into a shared hidden vector."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 5 * 3 + 30, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, board: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        board_features = self.conv(board)
        combined = torch.cat([board_features, meta], dim=-1)
        return self.head(combined)


class QNetwork(nn.Module):
    """DQN backbone with a shared encoder and a Q-value head."""

    def __init__(self, hidden_dim: int = 256, action_dim: int = 8) -> None:
        super().__init__()
        self.encoder = TetrisEncoder(hidden_dim=hidden_dim)
        self.q_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, board: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(board, meta)
        return self.q_head(hidden)


class ActorCritic(nn.Module):
    """Shared encoder actor-critic network for A2C."""

    def __init__(self, hidden_dim: int = 256, action_dim: int = 8) -> None:
        super().__init__()
        self.encoder = TetrisEncoder(hidden_dim=hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, board: torch.Tensor, meta: torch.Tensor) -> NetworkOutput:
        hidden = self.encoder(board, meta)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return NetworkOutput(logits=logits, value=value)
