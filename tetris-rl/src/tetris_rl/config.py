"""Typed configuration helpers for the Tetris RL project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RewardConfig:
    survival_reward: float = 0.05
    invalid_action_penalty: float = -0.05
    terminal_penalty: float = -5.0
    height_penalty: float = 0.02
    hole_penalty: float = 0.08
    bumpiness_penalty: float = 0.03
    line_clear_rewards: tuple[float, float, float, float] = (1.0, 3.0, 5.0, 8.0)


@dataclass
class EnvConfig:
    board_height: int = 20
    board_width: int = 10
    preview_count: int = 3
    max_episode_steps: int = 1000
    render_cell_size: int = 16
    invalid_action_behavior: str = "penalize"
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.board_height != 20 or self.board_width != 10:
            raise ValueError("This baseline keeps the classic 20x10 Tetris board fixed.")
        if self.preview_count != 3:
            raise ValueError("This baseline expects preview_count=3.")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive.")
        if self.invalid_action_behavior not in {"ignore", "penalize"}:
            raise ValueError("invalid_action_behavior must be 'ignore' or 'penalize'.")


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
    checkpoint_name: str = "dqn"


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
    checkpoint_name: str = "a2c"


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dqn: DQNConfig | None = None
    a2c: A2CConfig | None = None


def _merge_dataclass(dataclass_type: type[Any], raw: dict[str, Any] | None) -> Any:
    raw = dict(raw or {})
    if dataclass_type is EnvConfig and "reward" in raw:
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
