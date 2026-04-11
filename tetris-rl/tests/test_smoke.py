from __future__ import annotations

import sys
from pathlib import Path

import torch

from tetris_rl.a2c import A2CTrainer
from tetris_rl.config import A2CConfig, DQNConfig, EnvConfig, TrainConfig
from tetris_rl.demo_policy import main as demo_main
from tetris_rl.dqn import DQNTrainer
from tetris_rl.evaluation import evaluate_checkpoint


def make_env_config() -> EnvConfig:
    return EnvConfig(max_episode_steps=40)


def make_dqn_train_config(output_dir: Path) -> TrainConfig:
    return TrainConfig(seed=3, total_steps=64, log_interval=32, checkpoint_interval=64, output_dir=str(output_dir))


def make_a2c_train_config(output_dir: Path) -> TrainConfig:
    return TrainConfig(seed=5, total_steps=64, log_interval=32, checkpoint_interval=64, output_dir=str(output_dir))


def test_dqn_smoke_training_evaluation_and_demo(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "dqn"
    trainer = DQNTrainer(
        make_env_config(),
        make_dqn_train_config(output_dir),
        DQNConfig(
            batch_size=4,
            replay_size=128,
            warmup_steps=8,
            target_update_interval=16,
            epsilon_decay_steps=64,
            hidden_dim=64,
        ),
        device=torch.device("cpu"),
    )

    result = trainer.train()
    checkpoint_path = Path(result["last_checkpoint"])
    assert checkpoint_path.exists()

    evaluation = evaluate_checkpoint(checkpoint_path, episodes=1, device="cpu")
    assert evaluation["episodes"] == 1
    assert len(evaluation["returns"]) == 1

    gif_path = tmp_path / "demo.gif"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo-policy",
            "--checkpoint",
            str(checkpoint_path),
            "--render-mode",
            "ansi",
            "--sleep",
            "0.0",
        ],
    )
    demo_main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo-policy",
            "--checkpoint",
            str(checkpoint_path),
            "--render-mode",
            "rgb_array",
            "--gif-path",
            str(gif_path),
            "--sleep",
            "0.0",
        ],
    )
    demo_main()
    assert gif_path.exists()


def test_a2c_smoke_training_and_evaluation(tmp_path) -> None:
    output_dir = tmp_path / "a2c"
    trainer = A2CTrainer(
        make_env_config(),
        make_a2c_train_config(output_dir),
        A2CConfig(num_envs=4, n_steps=4, hidden_dim=64),
        device=torch.device("cpu"),
    )

    result = trainer.train()
    checkpoint_path = Path(result["last_checkpoint"])
    assert checkpoint_path.exists()

    evaluation = evaluate_checkpoint(checkpoint_path, episodes=1, device="cpu")
    assert evaluation["episodes"] == 1
    assert len(evaluation["scores"]) == 1
