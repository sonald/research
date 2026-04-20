# flappy-bird

Playable Flappy Bird clone with a Gymnasium-compatible reinforcement learning environment, reference DQN/A2C trainers, and a `pygame` human-play mode.

## Features

- Shared game core used by both the human-playable game and the RL environment
- Gymnasium-compatible environment with deterministic seeding plus `human`, `ansi`, and `rgb_array` rendering
- Dense reward setup that stays separate from the gameplay score
- Reference DQN and A2C baselines without depending on a high-level RL framework
- CLI entrypoints for training, evaluation, demos, and human play

## Quickstart

```bash
cd flappy-bird
python3 -m pip install -e '.[rl,play,dev]'
play-flappy
train-flappy-dqn --config configs/debug_dqn.yaml --device cpu
train-flappy-a2c --config configs/debug_a2c.yaml --device cpu
evaluate-flappy-policy --checkpoint outputs/debug_dqn/last.pt --episodes 3
demo-flappy-policy --checkpoint outputs/debug_dqn/last.pt --render-mode rgb_array --gif-path outputs/demo.gif
```

## Notes

- The project includes a tiny Gymnasium compatibility fallback so the source tree remains runnable before `gymnasium` is installed.
- The game uses simple original visuals rather than shipping copyrighted Flappy Bird assets.
- The RL observation is a compact numeric state vector; rendering is still available through the environment for demos and inspection.
- If you only want a subset of features, install the relevant extras: `.[play]` for the local game window, `.[rl]` for training/evaluation, or combine them.
