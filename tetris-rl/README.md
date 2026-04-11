# tetris-rl

Teaching-oriented Tetris reinforcement learning baselines implemented in pure Python and PyTorch.

## Features

- Gymnasium-compatible Tetris environment with hold, preview queue, 7-bag generation, and basic SRS wall kicks
- Dense-reward training setup that separates gameplay score from learning reward
- Reference DQN and A2C implementations without depending on a high-level RL framework
- CLI entrypoints for training, evaluation, and ASCII or GIF demos

## Quickstart

```bash
cd tetris-rl
python3 -m pip install -e .[dev]
train-dqn --config configs/debug_dqn.yaml --device cpu
train-a2c --config configs/debug_a2c.yaml --device cpu
evaluate-policy --checkpoint outputs/debug_dqn/last.pt --episodes 2
demo-policy --checkpoint outputs/debug_dqn/last.pt --render-mode ansi
```

## Notes

- The project includes a tiny Gymnasium compatibility fallback so the source tree remains runnable in environments where `gymnasium` is not installed yet.
- Long training runs should typically beat a random or purely exploratory policy, but the repository treats reproducibility and code clarity as the primary goal.
