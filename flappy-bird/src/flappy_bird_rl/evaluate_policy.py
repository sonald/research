"""CLI entrypoint for policy evaluation."""

from __future__ import annotations

import argparse

from .evaluation import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Flappy Bird policy checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint file.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device name.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions for actor-critic checkpoints.")
    args = parser.parse_args()

    result = evaluate_checkpoint(
        args.checkpoint,
        episodes=args.episodes,
        device=args.device,
        stochastic_policy=args.stochastic,
    )
    print("=" * 80)
    print("Evaluation finished")
    print(f"algo={result['algo']}")
    print(f"episodes={result['episodes']}")
    print(f"mean_return={result['mean_return']:.3f}")
    print(f"mean_score={result['mean_score']:.3f}")
    print(f"mean_length={result['mean_length']:.1f}")


if __name__ == "__main__":
    main()
