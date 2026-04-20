"""Typed configuration helpers for the Flappy Bird RL project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PhysicsConfig:
    screen_width: int = 288
    screen_height: int = 512
    ground_height: int = 112
    bird_x: float = 82.0
    bird_radius: float = 14.0
    gravity: float = 0.35
    flap_velocity: float = -4.8
    max_fall_speed: float = 8.0
    pipe_width: float = 52.0
    pipe_gap: float = 110.0
    pipe_speed: float = 2.5
    pipe_spacing: float = 180.0
    spawn_offset_x: float = 120.0
    pipe_margin_top: float = 40.0
    pipe_margin_bottom: float = 40.0
    start_y: float = 200.0

    def __post_init__(self) -> None:
        play_height = self.screen_height - self.ground_height
        if self.ground_height <= 0 or self.ground_height >= self.screen_height:
            raise ValueError("ground_height must leave a positive playable area.")
        if self.pipe_gap <= self.bird_radius * 2:
            raise ValueError("pipe_gap must be larger than the bird diameter.")
        if self.pipe_margin_top < 0 or self.pipe_margin_bottom < 0:
            raise ValueError("Pipe margins must be non-negative.")
        if self.pipe_margin_top + self.pipe_gap + self.pipe_margin_bottom >= play_height:
            raise ValueError("Pipe gap and margins must fit in the playable area.")
        if self.pipe_spacing <= self.pipe_width:
            raise ValueError("pipe_spacing must exceed pipe_width.")


@dataclass
class RewardConfig:
    survival_reward: float = 0.01
    pipe_reward: float = 1.0
    terminal_penalty: float = -1.0
    alignment_reward_scale: float = 0.02


@dataclass
class EnvConfig:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    max_episode_steps: int = 2000
    render_scale: int = 2

    def __post_init__(self) -> None:
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive.")
        if self.render_scale < 1:
            raise ValueError("render_scale must be positive.")


@dataclass
class TrainConfig:
    seed: int = 42
    total_steps: int = 20_000
    log_interval: int = 500
    checkpoint_interval: int = 2_000
    output_dir: str = "outputs/default"
    eval_episodes: int = 5


@dataclass
class DQNConfig:
    learning_rate: float = 1.0e-4
    gamma: float = 0.99
    batch_size: int = 64
    replay_size: int = 50_000
    warmup_steps: int = 2_000
    target_update_interval: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    hidden_dim: int = 256
    train_frequency: int = 1
    gradient_clip_norm: float = 10.0


@dataclass
class A2CConfig:
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    num_envs: int = 8
    n_steps: int = 5
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    hidden_dim: int = 256
    gradient_clip_norm: float = 0.5


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dqn: DQNConfig | None = None
    a2c: A2CConfig | None = None


def _merge_dataclass(dataclass_type: type[Any], raw: dict[str, Any] | None) -> Any:
    raw = dict(raw or {})
    if dataclass_type is EnvConfig:
        if "physics" in raw:
            raw["physics"] = _merge_dataclass(PhysicsConfig, raw["physics"])
        if "reward" in raw:
            raw["reward"] = _merge_dataclass(RewardConfig, raw["reward"])
    return dataclass_type(**raw)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return experiment_config_from_dict(raw)


def experiment_config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    raw = raw or {}
    return ExperimentConfig(
        env=_merge_dataclass(EnvConfig, raw.get("env")),
        train=_merge_dataclass(TrainConfig, raw.get("train")),
        dqn=_merge_dataclass(DQNConfig, raw.get("dqn")) if raw.get("dqn") is not None else None,
        a2c=_merge_dataclass(A2CConfig, raw.get("a2c")) if raw.get("a2c") is not None else None,
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)
