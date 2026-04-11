"""CLI entrypoint for evaluating a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Tetris policy checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint file.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device name.")
    args = parser.parse_args()

    result = evaluate_checkpoint(args.checkpoint, episodes=args.episodes, device=args.device)
    print("=" * 80)
    print(f"Checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"Algorithm: {result['algo']}")
    print(f"Episodes: {result['episodes']}")
    print(f"Mean return: {result['mean_return']:.3f}")
    print(f"Mean score: {result['mean_score']:.3f}")
    print(f"Mean length: {result['mean_length']:.1f}")


if __name__ == "__main__":
    main()
