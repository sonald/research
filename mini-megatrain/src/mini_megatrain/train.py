"""CLI entrypoint for the teaching implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import estimate_parameter_count, load_config
from .dataset import build_dataloader
from .engine import StreamingTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the teaching MegaTrain implementation")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    torch.manual_seed(config.training.seed)
    torch.set_float32_matmul_precision("high")

    trainer = StreamingTrainer(config)
    dataloader = build_dataloader(config)
    param_count = estimate_parameter_count(config.model)

    print("=" * 80)
    print("mini-megatrain")
    print("=" * 80)
    print(f"Config: {Path(args.config).resolve()}")
    print(f"Device: {trainer.device}")
    print(f"Compute dtype: {trainer.compute_dtype}")
    print(f"Approx parameter count: {param_count:,}")
    print(f"Checkpoint group size: {config.training.checkpoint_group_size}")
    print(f"Double buffer enabled: {config.training.use_double_buffer}")
    print("=" * 80)

    for step, batch in enumerate(dataloader, start=1):
        if step > config.training.total_steps:
            break

        metrics = trainer.train_step(batch)
        if step % config.training.log_interval == 0:
            print(
                f"step={step:04d} "
                f"loss={metrics.loss:.4f} "
                f"grad_norm={metrics.grad_norm:.4f} "
                f"tok/s={metrics.tokens_per_second:.1f} "
                f"peak_gpu_gb={metrics.peak_gpu_gb:.3f}"
            )


if __name__ == "__main__":
    main()
