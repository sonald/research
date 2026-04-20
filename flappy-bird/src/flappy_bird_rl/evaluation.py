"""Checkpoint loading and evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from .checkpoint import load_checkpoint
from .config import ExperimentConfig, experiment_config_from_dict
from .env import build_env
from .models import ActorCritic, QNetwork, observation_to_tensor


def _config_from_payload(payload: dict[str, Any]) -> ExperimentConfig:
    return experiment_config_from_dict(payload["config"])


def load_policy(path: str | Path, device: torch.device) -> tuple[str, ExperimentConfig, torch.nn.Module]:
    payload = load_checkpoint(path, device=device)
    config = _config_from_payload(payload)
    algo = str(payload["algo"])

    if algo == "dqn":
        model = QNetwork(hidden_dim=config.dqn.hidden_dim if config.dqn else 256).to(device)
    elif algo == "a2c":
        model = ActorCritic(hidden_dim=config.a2c.hidden_dim if config.a2c else 256).to(device)
    else:
        raise ValueError(f"Unsupported algorithm in checkpoint: {algo}")

    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return algo, config, model


def policy_action(
    algo: str,
    model: torch.nn.Module,
    observation: np.ndarray,
    device: torch.device,
    *,
    stochastic: bool = False,
) -> int:
    tensor = observation_to_tensor(observation, device)
    with torch.no_grad():
        if algo == "dqn":
            q_values = model(tensor)
            return int(q_values.argmax(dim=-1).item())

        output = model(tensor)
        if stochastic:
            distribution = Categorical(logits=output.logits)
            return int(distribution.sample().item())
        return int(output.logits.argmax(dim=-1).item())


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    episodes: int = 5,
    device: str | torch.device = "cpu",
    render_mode: str | None = None,
    stochastic_policy: bool = False,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    algo, config, model = load_policy(checkpoint_path, device=torch_device)
    env = build_env(config.env, render_mode=render_mode)

    episode_returns: list[float] = []
    episode_scores: list[int] = []
    episode_lengths: list[int] = []

    for episode_index in range(episodes):
        observation, _ = env.reset(seed=config.train.seed + 10_000 + episode_index)
        done = False
        total_reward = 0.0
        length = 0
        score = 0
        while not done:
            action = policy_action(algo, model, observation, torch_device, stochastic=stochastic_policy)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            length += 1
            score = int(info["score"])
            done = terminated or truncated
        episode_returns.append(float(total_reward))
        episode_scores.append(score)
        episode_lengths.append(length)

    env.close()
    return {
        "algo": algo,
        "episodes": episodes,
        "mean_return": float(np.mean(episode_returns)),
        "mean_score": float(np.mean(episode_scores)),
        "mean_length": float(np.mean(episode_lengths)),
        "returns": episode_returns,
        "scores": episode_scores,
        "lengths": episode_lengths,
    }
