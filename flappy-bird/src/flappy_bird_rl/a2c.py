"""Reference A2C trainer for the Flappy Bird baseline."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from .checkpoint import save_checkpoint
from .config import A2CConfig, EnvConfig, ExperimentConfig, TrainConfig, config_to_dict
from .env import build_env
from .gym_compat import SyncVectorEnv
from .models import ActorCritic, observation_to_tensor


class A2CTrainer:
    """Synchronous multi-environment A2C implementation."""

    def __init__(
        self,
        env_config: EnvConfig,
        train_config: TrainConfig,
        algo_config: A2CConfig,
        *,
        device: torch.device,
    ) -> None:
        self.env_config = env_config
        self.train_config = train_config
        self.algo_config = algo_config
        self.device = device
        self.global_step = 0
        self.best_mean_return = float("-inf")

        self.envs = SyncVectorEnv([self._make_env_fn(seed_offset=i) for i in range(algo_config.num_envs)])
        self.model = ActorCritic(hidden_dim=algo_config.hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=algo_config.learning_rate)

    def _make_env_fn(self, seed_offset: int) -> Any:
        def _factory() -> Any:
            env = build_env(self.env_config)
            env.reset(seed=self.train_config.seed + seed_offset)
            return env

        return _factory

    def _save(self, name: str, recent_returns: list[float]) -> Path:
        experiment_config = ExperimentConfig(env=self.env_config, train=self.train_config, a2c=self.algo_config)
        return save_checkpoint(
            Path(self.train_config.output_dir) / name,
            algo="a2c",
            model=self.model,
            optimizer=self.optimizer,
            config=config_to_dict(experiment_config),
            step=self.global_step,
            extra={"recent_returns": list(recent_returns)},
        )

    def train(self) -> dict[str, Any]:
        observation, _ = self.envs.reset(seed=self.train_config.seed)
        output_dir = Path(self.train_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        episode_returns = np.zeros(self.algo_config.num_envs, dtype=np.float32)
        episode_lengths = np.zeros(self.algo_config.num_envs, dtype=np.int32)
        completed_returns: deque[float] = deque(maxlen=50)
        completed_lengths: deque[int] = deque(maxlen=50)

        while self.global_step < self.train_config.total_steps:
            rollout_observations: list[np.ndarray] = []
            rollout_actions: list[np.ndarray] = []
            rollout_rewards: list[np.ndarray] = []
            rollout_dones: list[np.ndarray] = []

            for _ in range(self.algo_config.n_steps):
                obs_tensor = observation_to_tensor(observation, self.device)
                with torch.no_grad():
                    output = self.model(obs_tensor)
                    distribution = Categorical(logits=output.logits)
                    actions = distribution.sample()

                next_observation, rewards, terminated, truncated, _ = self.envs.step(actions.cpu().numpy())
                dones = np.logical_or(terminated, truncated)

                rollout_observations.append(np.asarray(observation, dtype=np.float32))
                rollout_actions.append(actions.cpu().numpy())
                rollout_rewards.append(np.asarray(rewards, dtype=np.float32))
                rollout_dones.append(np.asarray(dones, dtype=np.float32))

                episode_returns += rewards
                episode_lengths += 1
                for index, done in enumerate(dones):
                    if done:
                        completed_returns.append(float(episode_returns[index]))
                        completed_lengths.append(int(episode_lengths[index]))
                        episode_returns[index] = 0.0
                        episode_lengths[index] = 0

                observation = next_observation
                self.global_step += self.algo_config.num_envs
                if self.global_step >= self.train_config.total_steps:
                    break

            obs_tensor = observation_to_tensor(observation, self.device)
            with torch.no_grad():
                bootstrap = self.model(obs_tensor).value

            rewards_tensor = torch.as_tensor(np.stack(rollout_rewards), dtype=torch.float32, device=self.device)
            dones_tensor = torch.as_tensor(np.stack(rollout_dones), dtype=torch.float32, device=self.device)
            returns = torch.zeros_like(rewards_tensor)
            next_return = bootstrap
            for step_index in reversed(range(rewards_tensor.shape[0])):
                next_return = rewards_tensor[step_index] + self.algo_config.gamma * next_return * (
                    1.0 - dones_tensor[step_index]
                )
                returns[step_index] = next_return

            flat_observations = torch.as_tensor(
                np.concatenate(rollout_observations, axis=0), dtype=torch.float32, device=self.device
            )
            flat_actions = torch.as_tensor(np.concatenate(rollout_actions, axis=0), dtype=torch.long, device=self.device)
            flat_returns = returns.reshape(-1)

            output = self.model(flat_observations)
            distribution = Categorical(logits=output.logits)
            log_probs = distribution.log_prob(flat_actions)
            entropy = distribution.entropy().mean()
            advantages = flat_returns - output.value

            policy_loss = -(advantages.detach() * log_probs).mean()
            value_loss = torch.mean(torch.square(advantages))
            loss = policy_loss + self.algo_config.value_coef * value_loss - self.algo_config.entropy_coef * entropy

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.algo_config.gradient_clip_norm)
            self.optimizer.step()

            if self.global_step % self.train_config.log_interval < self.algo_config.num_envs:
                mean_return = float(np.mean(completed_returns)) if completed_returns else 0.0
                mean_length = float(np.mean(completed_lengths)) if completed_lengths else 0.0
                print(
                    f"[A2C] step={self.global_step} episodes={len(completed_returns)} "
                    f"mean_return={mean_return:.3f} mean_length={mean_length:.1f} "
                    f"policy_loss={policy_loss.item():.4f} value_loss={value_loss.item():.4f}"
                )
                if completed_returns and mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self._save("best.pt", list(completed_returns))

            if self.global_step % self.train_config.checkpoint_interval < self.algo_config.num_envs:
                self._save("last.pt", list(completed_returns))

        last_path = self._save("last.pt", list(completed_returns))
        return {
            "last_checkpoint": str(last_path),
            "mean_return": float(np.mean(completed_returns)) if completed_returns else 0.0,
            "episodes": len(completed_returns),
        }
