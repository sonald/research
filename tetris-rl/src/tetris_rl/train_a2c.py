"""CLI entrypoint for A2C training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import load_experiment_config
from .train_dqn import resolve_device
from .a2c import A2CTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Tetris A2C baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if config.a2c is None:
        raise ValueError("The provided config does not include an a2c section.")

    device = resolve_device(args.device)
    trainer = A2CTrainer(config.env, config.train, config.a2c, device=device)
    result = trainer.train()
    print("=" * 80)
    print("A2C training finished")
    print(f"Config: {Path(args.config).resolve()}")
    print(f"Device: {device}")
    print(f"Last checkpoint: {result['last_checkpoint']}")
    print(f"Mean return: {result['mean_return']:.3f}")


if __name__ == "__main__":
    main()
