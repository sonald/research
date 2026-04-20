"""Compatibility layer for Gymnasium."""

from __future__ import annotations

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.vector import SyncVectorEnv
    from gymnasium.wrappers import TimeLimit
except ImportError:  # pragma: no cover - exercised when gymnasium is absent.
    from . import mini_gym as gym
    from .mini_gym import SyncVectorEnv, TimeLimit, spaces

__all__ = ["SyncVectorEnv", "TimeLimit", "gym", "spaces"]
