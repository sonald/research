"""Reference DQN trainer for the Flappy Bird baseline."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .checkpoint import save_checkpoint
from .config import DQNConfig, EnvConfig, ExperimentConfig, TrainConfig, config_to_dict
from .env import build_env
from .models import QNetwork, observation_to_tensor
from .replay import ReplayBuffer


class DQNTrainer:
    """Basic DQN implementation for the compact Flappy state."""

    def __init__(
        self,
        env_config: EnvConfig,
        train_config: TrainConfig,
        algo_config: DQNConfig,
        *,
        device: torch.device,
    ) -> None:
        self.env_config = env_config
        self.train_config = train_config
        self.algo_config = algo_config
        self.device = device
        self.env = build_env(env_config)
        self.rng = np.random.default_rng(train_config.seed)

        self.q_network = QNetwork(hidden_dim=algo_config.hidden_dim).to(device)
        self.target_network = deepcopy(self.q_network).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=algo_config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(capacity=algo_config.replay_size, observation_dim=8)
        self.global_step = 0
        self.best_mean_return = float("-inf")
        self.completed_episode_count = 0

    def _epsilon(self, step: int) -> float:
        progress = min(step / max(1, self.algo_config.epsilon_decay_steps), 1.0)
        return self.algo_config.epsilon_start + progress * (
            self.algo_config.epsilon_end - self.algo_config.epsilon_start
        )

    def select_action(self, observation: np.ndarray, *, greedy: bool = False) -> int:
        epsilon = 0.0 if greedy else self._epsilon(self.global_step)
        if not greedy and self.rng.random() < epsilon:
            return int(self.rng.integers(0, 2))

        tensor = observation_to_tensor(observation, self.device)
        with torch.no_grad():
            q_values = self.q_network(tensor)
        return int(q_values.argmax(dim=-1).item())

    def _update(self) -> dict[str, float]:
        batch = self.replay.sample(self.algo_config.batch_size, rng=self.rng)

        observations = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(batch["next_observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        q_values = self.q_network(observations)
        chosen_q = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_values = self.target_network(next_observations)
            next_q = next_q_values.max(dim=-1).values
            targets = rewards + self.algo_config.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(chosen_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.algo_config.gradient_clip_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "q_mean": float(chosen_q.mean().item()),
            "target_mean": float(targets.mean().item()),
        }

    def _save(self, name: str, recent_returns: list[float]) -> Path:
        experiment_config = ExperimentConfig(env=self.env_config, train=self.train_config, dqn=self.algo_config)
        return save_checkpoint(
            Path(self.train_config.output_dir) / name,
            algo="dqn",
            model=self.q_network,
            optimizer=self.optimizer,
            config=config_to_dict(experiment_config),
            step=self.global_step,
            extra={"recent_returns": list(recent_returns)},
        )

    def _episode_seed(self) -> int:
        return self.train_config.seed + self.completed_episode_count

    def train(self) -> dict[str, Any]:
        observation, _ = self.env.reset(seed=self._episode_seed())
        episode_return = 0.0
        episode_length = 0
        completed_returns: deque[float] = deque(maxlen=50)
        completed_lengths: deque[int] = deque(maxlen=50)
        latest_metrics: dict[str, float] = {}

        output_dir = Path(self.train_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for step in range(1, self.train_config.total_steps + 1):
            self.global_step = step
            action = self.select_action(observation, greedy=False)
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.replay.add(observation, action, reward, next_observation, done)
            observation = next_observation
            episode_return += reward
            episode_length += 1

            if len(self.replay) >= max(self.algo_config.batch_size, self.algo_config.warmup_steps):
                if step % self.algo_config.train_frequency == 0:
                    latest_metrics = self._update()

            if step % self.algo_config.target_update_interval == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            if done:
                completed_returns.append(float(episode_return))
                completed_lengths.append(int(episode_length))
                self.completed_episode_count += 1
                observation, _ = self.env.reset(seed=self._episode_seed())
                episode_return = 0.0
                episode_length = 0

            if step % self.train_config.log_interval == 0:
                mean_return = float(np.mean(completed_returns)) if completed_returns else 0.0
                mean_length = float(np.mean(completed_lengths)) if completed_lengths else 0.0
                epsilon = self._epsilon(step)
                print(
                    f"[DQN] step={step} epsilon={epsilon:.3f} replay={len(self.replay)} "
                    f"episodes={len(completed_returns)} mean_return={mean_return:.3f} "
                    f"mean_length={mean_length:.1f} loss={latest_metrics.get('loss', 0.0):.4f}"
                )
                if completed_returns and mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self._save("best.pt", list(completed_returns))

            if step % self.train_config.checkpoint_interval == 0:
                self._save("last.pt", list(completed_returns))

        last_path = self._save("last.pt", list(completed_returns))
        return {
            "last_checkpoint": str(last_path),
            "mean_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
            "episodes": len(completed_returns),
        }
