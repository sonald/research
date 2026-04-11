"""CLI entrypoint for policy demos."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image
import torch

from .env import build_env
from .evaluation import load_policy, policy_action


def save_gif(frames: list[Image.Image], path: Path, duration_ms: int) -> None:
    if not frames:
        raise ValueError("No frames were captured for GIF export.")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ASCII or GIF demo for a trained Tetris policy.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a checkpoint file.")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="ansi",
        choices=["ansi", "rgb_array"],
        help="Rendering backend to use during the demo.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device name.")
    parser.add_argument("--sleep", type=float, default=0.1, help="Delay between ANSI frames.")
    parser.add_argument("--gif-path", type=str, default=None, help="Optional GIF output path for rgb_array demos.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions for actor-critic checkpoints.")
    args = parser.parse_args()

    device = torch.device(args.device)
    algo, config, model = load_policy(args.checkpoint, device=device)
    env = build_env(config.env, render_mode=args.render_mode)

    observation, _ = env.reset(seed=config.train.seed + 20_000)
    done = False
    frames: list[Image.Image] = []

    while not done:
        rendered = env.render()
        if args.render_mode == "ansi" and isinstance(rendered, str):
            print("\033[H\033[J", end="")
            print(rendered)
            time.sleep(args.sleep)
        elif args.render_mode == "rgb_array" and rendered is not None:
            frames.append(Image.fromarray(rendered))

        action = policy_action(algo, model, observation, device, stochastic=args.stochastic)
        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    rendered = env.render()
    if args.render_mode == "ansi" and isinstance(rendered, str):
        print("\033[H\033[J", end="")
        print(rendered)
        print(f"final_score={info['score']} total_reward={info['episode_reward']:.3f}")
    elif args.render_mode == "rgb_array" and rendered is not None:
        frames.append(Image.fromarray(rendered))
        if args.gif_path is None:
            raise ValueError("--gif-path is required when render-mode=rgb_array.")
        gif_path = Path(args.gif_path)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        save_gif(frames, gif_path, duration_ms=max(int(args.sleep * 1000), 40))
        print(f"Saved GIF demo to {gif_path.resolve()}")

    env.close()


if __name__ == "__main__":
    main()
