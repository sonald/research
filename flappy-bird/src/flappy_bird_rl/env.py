"""Gymnasium-compatible Flappy Bird environment."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .config import EnvConfig
from .core import FlappyBirdGame
from .gym_compat import TimeLimit, gym, spaces

ACTION_NOOP = 0
ACTION_FLAP = 1
ACTION_NAMES = ("noop", "flap")


class FlappyBirdEnv(gym.Env):
    """Compact state-based environment for Flappy Bird."""

    metadata = {"render_modes": ["ansi", "human", "rgb_array"], "render_fps": 60}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.game = FlappyBirdGame(self.config)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(8,), dtype=np.float32)
        self.episode_reward = 0.0
        self.last_seed: int | None = None
        self._pygame: Any | None = None
        self._screen: Any | None = None
        self._clock: Any | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.last_seed = int(seed)
        else:
            self.last_seed = None
        self.game.reset(seed=seed)
        self.episode_reward = 0.0
        observation = self.game.observation()
        if self.render_mode == "human":
            self._render_human()
        return observation, self._build_info(reward=0.0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.game.terminated:
            raise RuntimeError("Cannot call step() on a terminated episode. Reset the environment first.")

        action_index = int(action)
        if action_index < 0 or action_index >= len(ACTION_NAMES):
            raise ValueError(f"Action {action_index} is out of range.")

        self.game.step(flap=action_index == ACTION_FLAP)
        reward = self._compute_reward()
        self.episode_reward += reward
        observation = self.game.observation()
        if self.render_mode == "human":
            self._render_human()
        info = self._build_info(reward=reward)
        return observation, reward, self.game.terminated, False, info

    def render(self) -> str | np.ndarray | None:
        if self.render_mode == "ansi":
            return self.game.render_ansi()
        if self.render_mode == "human":
            self._render_human()
            return None
        if self.render_mode == "rgb_array":
            return self.game.render_rgb_array()
        return None

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self._screen = None
            self._clock = None
        return None

    def _compute_reward(self) -> float:
        reward = self.config.reward.survival_reward
        reward += self.config.reward.alignment_reward_scale * self.game.alignment_term()
        if self.game.last_passed_pipe:
            reward += self.config.reward.pipe_reward
        if self.game.terminated:
            reward += self.config.reward.terminal_penalty
        return float(reward)

    def _build_info(self, *, reward: float) -> dict[str, Any]:
        info = self.game.info()
        info["episode_reward"] = float(self.episode_reward)
        info["reward"] = float(reward)
        info["seed"] = self.last_seed
        return info

    def _render_human(self) -> None:
        pygame = self._ensure_pygame()
        frame = self.game.render_rgb_array()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        assert self._screen is not None
        self._screen.blit(surface, (0, 0))
        pygame.display.flip()
        assert self._clock is not None
        self._clock.tick(self.metadata["render_fps"])

    def _ensure_pygame(self) -> Any:
        if self._pygame is not None:
            return self._pygame

        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - depends on local environment.
            raise RuntimeError("pygame is required for render_mode='human'. Install the play extra.") from exc

        os.environ.setdefault("SDL_HINT_RENDER_SCALE_QUALITY", "0")
        pygame.init()
        width = self.config.physics.screen_width * self.config.render_scale
        height = self.config.physics.screen_height * self.config.render_scale
        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Flappy Bird RL")
        self._clock = pygame.time.Clock()
        self._pygame = pygame
        return pygame


def build_env(config: EnvConfig | None = None, render_mode: str | None = None) -> TimeLimit:
    """Build a default environment wrapped with a time limit."""

    env_config = config or EnvConfig()
    env = FlappyBirdEnv(config=env_config, render_mode=render_mode)
    return TimeLimit(env, max_episode_steps=env_config.max_episode_steps)
