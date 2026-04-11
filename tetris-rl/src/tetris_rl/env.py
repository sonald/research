"""Gymnasium-compatible Tetris environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import EnvConfig
from .gym_compat import TimeLimit, gym, spaces
from .pieces import EMPTY_HOLD_INDEX, PIECE_NAMES, PIECE_STATES, PIECE_TO_INDEX, get_kicks

ACTION_NAMES = (
    "noop",
    "left",
    "right",
    "rotate_cw",
    "rotate_ccw",
    "soft_drop",
    "hard_drop",
    "hold",
)

ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}
LINE_SCORE_TABLE = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}
EMPTY_COLOR = np.asarray((18, 24, 33), dtype=np.uint8)
GHOST_COLOR = np.asarray((95, 107, 120), dtype=np.uint8)
PIECE_COLORS = {
    "I": np.asarray((75, 206, 250), dtype=np.uint8),
    "J": np.asarray((65, 105, 225), dtype=np.uint8),
    "L": np.asarray((255, 165, 0), dtype=np.uint8),
    "O": np.asarray((245, 215, 66), dtype=np.uint8),
    "S": np.asarray((60, 179, 113), dtype=np.uint8),
    "T": np.asarray((186, 85, 211), dtype=np.uint8),
    "Z": np.asarray((220, 20, 60), dtype=np.uint8),
}


class TetrisEnv(gym.Env):
    """A compact Tetris environment with basic SRS rotation and dense rewards."""

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 8}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.height = self.config.board_height
        self.width = self.config.board_width
        self.preview_count = self.config.preview_count

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0.0, high=1.0, shape=(3, self.height, self.width), dtype=np.float32),
                "meta": spaces.Box(low=0.0, high=1.0, shape=(30,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(len(ACTION_NAMES),), dtype=bool),
            }
        )

        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.current_piece: str = "T"
        self.current_rotation = 0
        self.current_x = 3
        self.current_y = 0
        self.hold_piece: str | None = None
        self.can_hold = True
        self.piece_queue: list[str] = []
        self.score = 0
        self.lines_cleared = 0
        self.step_count = 0
        self.episode_reward = 0.0
        self.terminated = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.board.fill(0)
        self.hold_piece = None
        self.can_hold = True
        self.piece_queue = []
        self.score = 0
        self.lines_cleared = 0
        self.step_count = 0
        self.episode_reward = 0.0
        self.terminated = False
        self._ensure_queue()
        self._spawn_piece(can_hold=True)
        observation = self._get_observation()
        return observation, self._build_info(lines_cleared=0, invalid_action=False)

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.terminated:
            raise RuntimeError("Cannot call step() on a terminated episode. Reset the environment first.")

        action_index = int(action)
        if action_index < 0 or action_index >= len(ACTION_NAMES):
            raise ValueError(f"Action {action_index} is out of range.")

        before_features = self._board_features()
        invalid_action = False
        lines_cleared = 0

        if action_index == ACTION_TO_INDEX["left"]:
            invalid_action = not self._attempt_move(dx=-1, dy=0)
        elif action_index == ACTION_TO_INDEX["right"]:
            invalid_action = not self._attempt_move(dx=1, dy=0)
        elif action_index == ACTION_TO_INDEX["rotate_cw"]:
            invalid_action = not self._attempt_rotation(delta=1)
        elif action_index == ACTION_TO_INDEX["rotate_ccw"]:
            invalid_action = not self._attempt_rotation(delta=-1)
        elif action_index == ACTION_TO_INDEX["soft_drop"]:
            self._attempt_move(dx=0, dy=1)
        elif action_index == ACTION_TO_INDEX["hard_drop"]:
            self.current_y += self._drop_distance()
            lines_cleared = self._lock_current_piece()
        elif action_index == ACTION_TO_INDEX["hold"]:
            invalid_action = not self._execute_hold()

        if action_index != ACTION_TO_INDEX["hard_drop"] and not self.terminated:
            if not self._attempt_move(dx=0, dy=1):
                lines_cleared = self._lock_current_piece()

        after_features = self._board_features()
        reward = self._compute_reward(
            before_features=before_features,
            after_features=after_features,
            lines_cleared=lines_cleared,
            invalid_action=invalid_action,
            terminated=self.terminated,
        )
        self.episode_reward += reward
        self.step_count += 1

        observation = self._get_observation()
        info = self._build_info(lines_cleared=lines_cleared, invalid_action=invalid_action)
        return observation, reward, self.terminated, False, info

    def render(self) -> str | np.ndarray | None:
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None

    def close(self) -> None:
        return None

    def _ensure_queue(self) -> None:
        target = self.preview_count + 1
        while len(self.piece_queue) < target:
            shuffled = list(self.np_random.permutation(np.asarray(PIECE_NAMES)))
            self.piece_queue.extend(str(name) for name in shuffled)

    def _spawn_piece(self, piece_name: str | None = None, *, can_hold: bool = True) -> None:
        self._ensure_queue()
        self.current_piece = piece_name or self.piece_queue.pop(0)
        self.current_rotation = 0
        self.current_x = 3
        self.current_y = 0
        self.can_hold = can_hold
        if self._collides(self.current_piece, self.current_rotation, self.current_x, self.current_y):
            self.terminated = True

    def _piece_cells(self, piece_name: str | None = None, rotation: int | None = None) -> tuple[tuple[int, int], ...]:
        name = piece_name or self.current_piece
        state = rotation if rotation is not None else self.current_rotation
        return PIECE_STATES[name][state]

    def _absolute_cells(
        self,
        piece_name: str | None = None,
        rotation: int | None = None,
        x: int | None = None,
        y: int | None = None,
    ) -> list[tuple[int, int]]:
        cells = self._piece_cells(piece_name=piece_name, rotation=rotation)
        anchor_x = self.current_x if x is None else x
        anchor_y = self.current_y if y is None else y
        return [(anchor_y + row, anchor_x + col) for row, col in cells]

    def _collides(self, piece_name: str, rotation: int, x: int, y: int) -> bool:
        for row, col in self._absolute_cells(piece_name=piece_name, rotation=rotation, x=x, y=y):
            if col < 0 or col >= self.width or row >= self.height:
                return True
            if row >= 0 and self.board[row, col] != 0:
                return True
        return False

    def _attempt_move(self, dx: int, dy: int) -> bool:
        new_x = self.current_x + dx
        new_y = self.current_y + dy
        if self._collides(self.current_piece, self.current_rotation, new_x, new_y):
            return False
        self.current_x = new_x
        self.current_y = new_y
        return True

    def _rotation_valid(self, delta: int) -> bool:
        from_rotation = self.current_rotation
        to_rotation = (from_rotation + delta) % 4
        for kick_x, kick_y in get_kicks(self.current_piece, from_rotation, to_rotation):
            candidate_x = self.current_x + kick_x
            candidate_y = self.current_y - kick_y
            if not self._collides(self.current_piece, to_rotation, candidate_x, candidate_y):
                return True
        return False

    def _attempt_rotation(self, delta: int) -> bool:
        from_rotation = self.current_rotation
        to_rotation = (from_rotation + delta) % 4
        for kick_x, kick_y in get_kicks(self.current_piece, from_rotation, to_rotation):
            candidate_x = self.current_x + kick_x
            candidate_y = self.current_y - kick_y
            if not self._collides(self.current_piece, to_rotation, candidate_x, candidate_y):
                self.current_rotation = to_rotation
                self.current_x = candidate_x
                self.current_y = candidate_y
                return True
        return False

    def _drop_distance(self) -> int:
        distance = 0
        while not self._collides(self.current_piece, self.current_rotation, self.current_x, self.current_y + distance + 1):
            distance += 1
        return distance

    def _execute_hold(self) -> bool:
        if not self.can_hold:
            return False

        current_piece = self.current_piece
        if self.hold_piece is None:
            self.hold_piece = current_piece
            self._spawn_piece(can_hold=False)
        else:
            swapped_piece = self.hold_piece
            self.hold_piece = current_piece
            self._spawn_piece(piece_name=swapped_piece, can_hold=False)
        return True

    def _lock_current_piece(self) -> int:
        for row, col in self._absolute_cells():
            if row < 0:
                self.terminated = True
                return 0
            self.board[row, col] = PIECE_TO_INDEX[self.current_piece] + 1

        full_rows = np.where(np.all(self.board != 0, axis=1))[0]
        cleared = int(len(full_rows))
        if cleared:
            self.board = np.delete(self.board, full_rows, axis=0)
            padding = np.zeros((cleared, self.width), dtype=np.int8)
            self.board = np.vstack([padding, self.board])
            self.score += LINE_SCORE_TABLE[cleared]
            self.lines_cleared += cleared

        if not self.terminated:
            self._spawn_piece(can_hold=True)
        return cleared

    def _board_features(self) -> dict[str, float]:
        occupied = self.board != 0
        heights = np.zeros(self.width, dtype=np.int32)
        holes = 0
        for col in range(self.width):
            filled_rows = np.flatnonzero(occupied[:, col])
            if len(filled_rows) == 0:
                continue
            first = int(filled_rows[0])
            heights[col] = self.height - first
            holes += int(np.sum(~occupied[first:, col]))
        bumpiness = int(np.sum(np.abs(np.diff(heights))))
        aggregate_height = int(np.sum(heights))
        return {
            "aggregate_height": float(aggregate_height),
            "holes": float(holes),
            "bumpiness": float(bumpiness),
        }

    def _compute_reward(
        self,
        *,
        before_features: dict[str, float],
        after_features: dict[str, float],
        lines_cleared: int,
        invalid_action: bool,
        terminated: bool,
    ) -> float:
        reward = self.config.reward.survival_reward
        reward += self.config.reward.height_penalty * (
            before_features["aggregate_height"] - after_features["aggregate_height"]
        )
        reward += self.config.reward.hole_penalty * (before_features["holes"] - after_features["holes"])
        reward += self.config.reward.bumpiness_penalty * (
            before_features["bumpiness"] - after_features["bumpiness"]
        )
        if lines_cleared:
            reward += self.config.reward.line_clear_rewards[lines_cleared - 1]
        if invalid_action and self.config.invalid_action_behavior == "penalize":
            reward += self.config.reward.invalid_action_penalty
        if terminated:
            reward += self.config.reward.terminal_penalty
        return float(reward)

    def _get_action_mask(self) -> np.ndarray:
        mask = np.zeros(len(ACTION_NAMES), dtype=bool)
        if self.terminated:
            return mask

        mask[ACTION_TO_INDEX["noop"]] = True
        mask[ACTION_TO_INDEX["soft_drop"]] = True
        mask[ACTION_TO_INDEX["hard_drop"]] = True
        mask[ACTION_TO_INDEX["hold"]] = self.can_hold
        mask[ACTION_TO_INDEX["left"]] = not self._collides(
            self.current_piece, self.current_rotation, self.current_x - 1, self.current_y
        )
        mask[ACTION_TO_INDEX["right"]] = not self._collides(
            self.current_piece, self.current_rotation, self.current_x + 1, self.current_y
        )
        mask[ACTION_TO_INDEX["rotate_cw"]] = self._rotation_valid(delta=1)
        mask[ACTION_TO_INDEX["rotate_ccw"]] = self._rotation_valid(delta=-1)
        return mask

    def _get_board_channels(self) -> np.ndarray:
        settled = (self.board != 0).astype(np.float32)
        active = np.zeros_like(settled, dtype=np.float32)
        ghost = np.zeros_like(settled, dtype=np.float32)

        drop_y = self.current_y + self._drop_distance()
        active_cells = set(self._absolute_cells())
        for row, col in self._absolute_cells(x=self.current_x, y=drop_y):
            if 0 <= row < self.height and 0 <= col < self.width and (row, col) not in active_cells:
                ghost[row, col] = 1.0
        for row, col in active_cells:
            if 0 <= row < self.height and 0 <= col < self.width:
                active[row, col] = 1.0
        return np.stack([settled, active, ghost], axis=0)

    def _get_meta_vector(self) -> np.ndarray:
        hold = np.zeros(len(PIECE_NAMES) + 1, dtype=np.float32)
        hold_index = EMPTY_HOLD_INDEX if self.hold_piece is None else PIECE_TO_INDEX[self.hold_piece]
        hold[hold_index] = 1.0

        preview = np.zeros(self.preview_count * len(PIECE_NAMES), dtype=np.float32)
        self._ensure_queue()
        for preview_index, piece_name in enumerate(self.piece_queue[: self.preview_count]):
            preview_offset = preview_index * len(PIECE_NAMES)
            preview[preview_offset + PIECE_TO_INDEX[piece_name]] = 1.0

        can_hold = np.asarray([1.0 if self.can_hold else 0.0], dtype=np.float32)
        return np.concatenate([hold, preview, can_hold], axis=0)

    def _get_observation(self) -> dict[str, np.ndarray]:
        return {
            "board": self._get_board_channels().astype(np.float32),
            "meta": self._get_meta_vector().astype(np.float32),
            "action_mask": self._get_action_mask(),
        }

    def _build_info(self, *, lines_cleared: int, invalid_action: bool) -> dict[str, Any]:
        return {
            "score": int(self.score),
            "lines_cleared_total": int(self.lines_cleared),
            "lines_cleared_step": int(lines_cleared),
            "current_piece": self.current_piece,
            "hold_piece": self.hold_piece,
            "can_hold": bool(self.can_hold),
            "step_count": int(self.step_count),
            "episode_reward": float(self.episode_reward),
            "invalid_action": bool(invalid_action),
        }

    def _render_ansi(self) -> str:
        canvas = np.full((self.height, self.width), ".", dtype="<U1")
        for row in range(self.height):
            for col in range(self.width):
                if self.board[row, col] != 0:
                    canvas[row, col] = PIECE_NAMES[self.board[row, col] - 1]

        ghost_y = self.current_y + self._drop_distance()
        active_cells = set(self._absolute_cells())
        for row, col in self._absolute_cells(x=self.current_x, y=ghost_y):
            if 0 <= row < self.height and 0 <= col < self.width and canvas[row, col] == ".":
                canvas[row, col] = "*"
        for row, col in active_cells:
            if 0 <= row < self.height and 0 <= col < self.width:
                canvas[row, col] = self.current_piece

        lines = ["+" + "-" * self.width + "+"]
        lines.extend("|" + "".join(row) + "|" for row in canvas)
        lines.append("+" + "-" * self.width + "+")
        lines.append(
            f"score={self.score} lines={self.lines_cleared} hold={self.hold_piece or '-'} "
            f"next={','.join(self.piece_queue[:self.preview_count])}"
        )
        return "\n".join(lines)

    def _render_rgb_array(self) -> np.ndarray:
        cell = self.config.render_cell_size
        image = np.zeros((self.height * cell, self.width * cell, 3), dtype=np.uint8)
        image[:] = EMPTY_COLOR

        def paint(row: int, col: int, color: np.ndarray) -> None:
            top = row * cell
            left = col * cell
            image[top : top + cell, left : left + cell] = color
            image[top : top + 1, left : left + cell] = 0
            image[top : top + cell, left : left + 1] = 0

        for row in range(self.height):
            for col in range(self.width):
                if self.board[row, col] != 0:
                    piece_name = PIECE_NAMES[self.board[row, col] - 1]
                    paint(row, col, PIECE_COLORS[piece_name])

        ghost_y = self.current_y + self._drop_distance()
        active_cells = set(self._absolute_cells())
        for row, col in self._absolute_cells(x=self.current_x, y=ghost_y):
            if 0 <= row < self.height and 0 <= col < self.width and (row, col) not in active_cells:
                paint(row, col, GHOST_COLOR)
        for row, col in active_cells:
            if 0 <= row < self.height and 0 <= col < self.width:
                paint(row, col, PIECE_COLORS[self.current_piece])
        return image


def build_env(config: EnvConfig | None = None, render_mode: str | None = None) -> TimeLimit:
    """Build a default environment wrapped with a time limit."""

    env_config = config or EnvConfig()
    env = TetrisEnv(config=env_config, render_mode=render_mode)
    return TimeLimit(env, max_episode_steps=env_config.max_episode_steps)
