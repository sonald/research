"""A tiny subset of the Gymnasium API used as a local fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class Space:
    """Base space interface."""

    def sample(self, rng: np.random.Generator | None = None) -> Any:
        raise NotImplementedError


class Discrete(Space):
    def __init__(self, n: int) -> None:
        self.n = int(n)

    def sample(self, rng: np.random.Generator | None = None) -> int:
        generator = rng or np.random.default_rng()
        return int(generator.integers(0, self.n))


class Box(Space):
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any) -> None:
        self.low = low
        self.high = high
        self.shape = tuple(shape)
        self.dtype = dtype

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        generator = rng or np.random.default_rng()
        return generator.uniform(self.low, self.high, size=self.shape).astype(self.dtype)


class Dict(Space):
    def __init__(self, spaces: dict[str, Space]) -> None:
        self.spaces = dict(spaces)

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        return {key: space.sample(rng=rng) for key, space in self.spaces.items()}


class spaces:
    Box = Box
    Dict = Dict
    Discrete = Discrete


class Env:
    metadata: dict[str, Any] = {}

    def __init__(self) -> None:
        self.np_random = np.random.default_rng()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        return None, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        raise NotImplementedError

    def render(self) -> Any:
        return None

    def close(self) -> None:
        return None


class TimeLimit:
    """Simple step-count truncation wrapper."""

    def __init__(self, env: Env, max_episode_steps: int) -> None:
        self.env = env
        self.max_episode_steps = int(max_episode_steps)
        self.elapsed_steps = 0
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.elapsed_steps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed_steps += 1
        if not terminated and not truncated and self.elapsed_steps >= self.max_episode_steps:
            truncated = True
            info = dict(info)
            info["time_limit_reached"] = True
        return observation, reward, terminated, truncated, info

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()


def _stack_observations(observations: list[Any]) -> Any:
    sample = observations[0]
    if isinstance(sample, dict):
        return {key: _stack_observations([obs[key] for obs in observations]) for key in sample}
    return np.stack(observations, axis=0)


def _listify_info(infos: list[dict[str, Any]]) -> dict[str, Any]:
    keys: set[str] = set()
    for info in infos:
        keys.update(info.keys())

    packed: dict[str, Any] = {}
    for key in keys:
        values = [info.get(key) for info in infos]
        if all(isinstance(value, np.ndarray) for value in values if value is not None):
            packed[key] = np.stack(values, axis=0)
        else:
            packed[key] = values
    return packed


class SyncVectorEnv:
    """A tiny synchronous vector environment with immediate auto-reset."""

    def __init__(self, env_fns: list[Any]) -> None:
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_action_space = self.envs[0].action_space
        self.single_observation_space = self.envs[0].observation_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        observations: list[Any] = []
        infos: list[dict[str, Any]] = []
        for index, env in enumerate(self.envs):
            child_seed = None if seed is None else int(seed) + index
            observation, info = env.reset(seed=child_seed, options=options)
            observations.append(observation)
            infos.append(info)
        return _stack_observations(observations), _listify_info(infos)

    def step(self, actions: np.ndarray | list[int]) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        observations: list[Any] = []
        rewards: list[float] = []
        terminateds: list[bool] = []
        truncateds: list[bool] = []
        infos: list[dict[str, Any]] = []
        final_observations: list[Any | None] = []
        final_infos: list[dict[str, Any] | None] = []

        for env, action in zip(self.envs, actions):
            observation, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            if done:
                final_observations.append(observation)
                final_infos.append(info)
                observation, reset_info = env.reset()
                info = dict(reset_info)
            else:
                final_observations.append(None)
                final_infos.append(None)

            observations.append(observation)
            rewards.append(float(reward))
            terminateds.append(bool(terminated))
            truncateds.append(bool(truncated))
            infos.append(dict(info))

        packed_info = _listify_info(infos)
        packed_info["final_observation"] = final_observations
        packed_info["final_info"] = final_infos
        return (
            _stack_observations(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminateds, dtype=bool),
            np.asarray(truncateds, dtype=bool),
            packed_info,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


@dataclass
class _WrapperModule:
    TimeLimit = TimeLimit


wrappers = _WrapperModule()
