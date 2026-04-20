"""Neural network modules and tensor helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


def observation_to_tensor(observation: Any, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(observation), dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


@dataclass
class NetworkOutput:
    logits: torch.Tensor
    value: torch.Tensor | None = None


class FlappyEncoder(nn.Module):
    """Encode the compact state vector into a hidden feature."""

    def __init__(self, state_dim: int = 8, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 8, hidden_dim: int = 256, action_dim: int = 2) -> None:
        super().__init__()
        self.encoder = FlappyEncoder(state_dim=state_dim, hidden_dim=hidden_dim)
        self.q_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(observation)
        return self.q_head(hidden)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 8, hidden_dim: int = 256, action_dim: int = 2) -> None:
        super().__init__()
        self.encoder = FlappyEncoder(state_dim=state_dim, hidden_dim=hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor) -> NetworkOutput:
        hidden = self.encoder(observation)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return NetworkOutput(logits=logits, value=value)
