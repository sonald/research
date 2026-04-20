"""CLI entrypoint for DQN training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import load_experiment_config
from .dqn import DQNTrainer


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Flappy Bird DQN baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if config.dqn is None:
        raise ValueError("The provided config does not include a dqn section.")

    device = resolve_device(args.device)
    trainer = DQNTrainer(config.env, config.train, config.dqn, device=device)
    result = trainer.train()
    print("=" * 80)
    print("DQN training finished")
    print(f"Config: {Path(args.config).resolve()}")
    print(f"Device: {device}")
    print(f"Last checkpoint: {result['last_checkpoint']}")
    print(f"Mean return: {result['mean_return']:.3f}")


if __name__ == "__main__":
    main()
