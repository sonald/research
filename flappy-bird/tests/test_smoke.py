from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from flappy_bird_rl.a2c import A2CTrainer
from flappy_bird_rl.config import A2CConfig, DQNConfig, EnvConfig, TrainConfig
from flappy_bird_rl.demo_policy import main as demo_main
from flappy_bird_rl.dqn import DQNTrainer
from flappy_bird_rl.evaluation import evaluate_checkpoint
from flappy_bird_rl.play import main as play_main


def make_env_config() -> EnvConfig:
    return EnvConfig(max_episode_steps=60, render_scale=1)


def make_dqn_train_config(output_dir: Path) -> TrainConfig:
    return TrainConfig(seed=3, total_steps=96, log_interval=48, checkpoint_interval=96, output_dir=str(output_dir))


def make_a2c_train_config(output_dir: Path) -> TrainConfig:
    return TrainConfig(seed=5, total_steps=128, log_interval=64, checkpoint_interval=128, output_dir=str(output_dir))


def test_dqn_smoke_training_evaluation_and_demo(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "dqn"
    trainer = DQNTrainer(
        make_env_config(),
        make_dqn_train_config(output_dir),
        DQNConfig(
            batch_size=8,
            replay_size=256,
            warmup_steps=16,
            target_update_interval=24,
            epsilon_decay_steps=96,
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
            "demo-flappy-policy",
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
            "demo-flappy-policy",
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


def test_play_smoke_if_pygame_is_installed(monkeypatch) -> None:
    if importlib.util.find_spec("pygame") is None:
        return

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setattr(sys, "argv", ["play-flappy", "--max-frames", "2", "--scale", "1"])
    play_main()


def test_env_human_render_smoke_if_pygame_is_installed(monkeypatch) -> None:
    if importlib.util.find_spec("pygame") is None:
        return

    from flappy_bird_rl.env import FlappyBirdEnv

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = FlappyBirdEnv(EnvConfig(render_scale=1), render_mode="human")
    env.reset(seed=0)
    env.render()
    env.step(0)
    env.close()


def test_dqn_episode_seed_continues_growing(tmp_path) -> None:
    trainer = DQNTrainer(
        make_env_config(),
        make_dqn_train_config(tmp_path / "seed-check"),
        DQNConfig(hidden_dim=32),
        device=torch.device("cpu"),
    )
    trainer.completed_episode_count = 75

    assert trainer._episode_seed() == trainer.train_config.seed + 75
