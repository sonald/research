"""Shared Flappy Bird gameplay core used by both RL and human play."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import EnvConfig, PhysicsConfig

SKY_COLOR = np.asarray((117, 202, 255), dtype=np.uint8)
GROUND_COLOR = np.asarray((222, 216, 149), dtype=np.uint8)
PIPE_COLOR = np.asarray((93, 201, 74), dtype=np.uint8)
PIPE_EDGE_COLOR = np.asarray((61, 143, 49), dtype=np.uint8)
BIRD_COLOR = np.asarray((255, 223, 63), dtype=np.uint8)
BIRD_WING_COLOR = np.asarray((242, 153, 74), dtype=np.uint8)
EYE_COLOR = np.asarray((255, 255, 255), dtype=np.uint8)
PUPIL_COLOR = np.asarray((35, 35, 35), dtype=np.uint8)
SCORE_COLOR = np.asarray((255, 255, 255), dtype=np.uint8)
DIGIT_BITMAPS = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}


@dataclass
class PipePair:
    x: float
    gap_y: float
    scored: bool = False


class FlappyBirdGame:
    """Deterministic fixed-timestep Flappy Bird core."""

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        self.physics = self.config.physics
        self.np_random = np.random.default_rng()
        self.bird_y = self.physics.start_y
        self.bird_velocity = 0.0
        self.pipes: list[PipePair] = []
        self.score = 0
        self.pipes_cleared = 0
        self.distance_travelled = 0.0
        self.tick_count = 0
        self.terminated = False
        self.collision_reason = ""
        self.last_passed_pipe = False

    @property
    def play_height(self) -> float:
        return float(self.physics.screen_height - self.physics.ground_height)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.bird_y = self.physics.start_y
        self.bird_velocity = 0.0
        self.score = 0
        self.pipes_cleared = 0
        self.distance_travelled = 0.0
        self.tick_count = 0
        self.terminated = False
        self.collision_reason = ""
        self.last_passed_pipe = False
        self.pipes = []
        first_x = self.physics.screen_width + self.physics.spawn_offset_x
        for index in range(3):
            self.pipes.append(PipePair(x=first_x + index * self.physics.pipe_spacing, gap_y=self._sample_gap_y()))

    def step(self, flap: bool) -> None:
        if self.terminated:
            raise RuntimeError("Cannot step a terminated game. Reset first.")

        self.last_passed_pipe = False
        if flap:
            self.bird_velocity = self.physics.flap_velocity

        self.bird_velocity = min(self.bird_velocity + self.physics.gravity, self.physics.max_fall_speed)
        self.bird_y += self.bird_velocity
        self.tick_count += 1

        for pipe in self.pipes:
            pipe.x -= self.physics.pipe_speed

        self.distance_travelled += self.physics.pipe_speed
        self.pipes = [pipe for pipe in self.pipes if pipe.x + self.physics.pipe_width > -8.0]
        while len(self.pipes) < 3:
            last_x = self.pipes[-1].x if self.pipes else self.physics.screen_width + self.physics.spawn_offset_x
            self.pipes.append(PipePair(x=last_x + self.physics.pipe_spacing, gap_y=self._sample_gap_y()))

        self._update_collision()
        if not self.terminated:
            self._score_completed_pipes()

    def _sample_gap_y(self) -> float:
        min_gap_y = self.physics.pipe_margin_top + self.physics.pipe_gap / 2.0
        max_gap_y = self.play_height - self.physics.pipe_margin_bottom - self.physics.pipe_gap / 2.0
        return float(self.np_random.uniform(min_gap_y, max_gap_y))

    def _update_collision(self) -> None:
        bird_left = self.physics.bird_x - self.physics.bird_radius
        bird_right = self.physics.bird_x + self.physics.bird_radius
        bird_top = self.bird_y - self.physics.bird_radius
        bird_bottom = self.bird_y + self.physics.bird_radius

        if bird_top <= 0.0:
            self.terminated = True
            self.collision_reason = "ceiling"
            return
        if bird_bottom >= self.play_height:
            self.terminated = True
            self.collision_reason = "floor"
            return

        for pipe in self.pipes:
            pipe_left = pipe.x
            pipe_right = pipe.x + self.physics.pipe_width
            gap_top = pipe.gap_y - self.physics.pipe_gap / 2.0
            gap_bottom = pipe.gap_y + self.physics.pipe_gap / 2.0
            overlaps_x = bird_right >= pipe_left and bird_left <= pipe_right
            outside_gap = bird_top <= gap_top or bird_bottom >= gap_bottom
            if overlaps_x and outside_gap:
                self.terminated = True
                self.collision_reason = "pipe"
                return

    def _score_completed_pipes(self) -> None:
        bird_left = self.physics.bird_x - self.physics.bird_radius
        for pipe in self.pipes:
            if not pipe.scored and pipe.x + self.physics.pipe_width < bird_left:
                pipe.scored = True
                self.score += 1
                self.pipes_cleared += 1
                self.last_passed_pipe = True

    def next_pipes(self) -> tuple[PipePair, PipePair]:
        ahead = [
            pipe
            for pipe in self.pipes
            if pipe.x + self.physics.pipe_width >= self.physics.bird_x - self.physics.bird_radius
        ]
        if not ahead:
            ahead = self.pipes[:]
        ahead.sort(key=lambda pipe: pipe.x)
        first = ahead[0]
        second = ahead[1] if len(ahead) > 1 else ahead[0]
        return first, second

    def observation(self) -> np.ndarray:
        next_pipe, second_pipe = self.next_pipes()
        play_height = self.play_height
        screen_width = float(self.physics.screen_width)
        return np.asarray(
            [
                self.bird_y / play_height,
                self.bird_velocity / self.physics.max_fall_speed,
                (next_pipe.x + self.physics.pipe_width - self.physics.bird_x) / screen_width,
                next_pipe.gap_y / play_height,
                (next_pipe.gap_y - self.physics.pipe_gap / 2.0) / play_height,
                (next_pipe.gap_y + self.physics.pipe_gap / 2.0) / play_height,
                (second_pipe.x + self.physics.pipe_width - self.physics.bird_x) / screen_width,
                second_pipe.gap_y / play_height,
            ],
            dtype=np.float32,
        )

    def alignment_term(self) -> float:
        next_pipe, _ = self.next_pipes()
        gap_half = self.physics.pipe_gap / 2.0
        normalized_error = abs(self.bird_y - next_pipe.gap_y) / max(gap_half, 1.0)
        return max(0.0, 1.0 - normalized_error)

    def info(self) -> dict[str, Any]:
        return {
            "score": int(self.score),
            "pipes_cleared": int(self.pipes_cleared),
            "distance_travelled": float(self.distance_travelled),
            "tick_count": int(self.tick_count),
            "collision_reason": self.collision_reason,
            "passed_pipe": bool(self.last_passed_pipe),
        }

    def render_rgb_array(self) -> np.ndarray:
        width = self.physics.screen_width * self.config.render_scale
        height = self.physics.screen_height * self.config.render_scale
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = SKY_COLOR

        play_height = int(self.play_height * self.config.render_scale)
        frame[play_height:, :] = GROUND_COLOR

        for pipe in self.pipes:
            self._draw_pipe(frame, pipe, self.physics, self.config.render_scale)
        self._draw_bird(frame, self.config.render_scale)
        self._draw_score_marks(frame, self.score, self.config.render_scale)
        return frame

    def render_ansi(self) -> str:
        next_pipe, second_pipe = self.next_pipes()
        return (
            f"score={self.score} y={self.bird_y:.1f} vy={self.bird_velocity:.2f} "
            f"next_dx={next_pipe.x - self.physics.bird_x:.1f} next_gap={next_pipe.gap_y:.1f} "
            f"second_dx={second_pipe.x - self.physics.bird_x:.1f} done={self.terminated} "
            f"collision={self.collision_reason or '-'}"
        )

    @staticmethod
    def _draw_rect(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: np.ndarray) -> None:
        height, width, _ = frame.shape
        left = max(0, min(width, x0))
        right = max(0, min(width, x1))
        top = max(0, min(height, y0))
        bottom = max(0, min(height, y1))
        if left < right and top < bottom:
            frame[top:bottom, left:right] = color

    @classmethod
    def _draw_pipe(cls, frame: np.ndarray, pipe: PipePair, physics: PhysicsConfig, scale: int) -> None:
        left = int(pipe.x * scale)
        right = int((pipe.x + physics.pipe_width) * scale)
        gap_top = int((pipe.gap_y - physics.pipe_gap / 2.0) * scale)
        gap_bottom = int((pipe.gap_y + physics.pipe_gap / 2.0) * scale)
        play_height = int((physics.screen_height - physics.ground_height) * scale)

        cls._draw_rect(frame, left, 0, right, gap_top, PIPE_COLOR)
        cls._draw_rect(frame, left, gap_bottom, right, play_height, PIPE_COLOR)
        cls._draw_rect(frame, left, max(0, gap_top - 6 * scale), right, gap_top, PIPE_EDGE_COLOR)
        cls._draw_rect(frame, left, gap_bottom, right, min(play_height, gap_bottom + 6 * scale), PIPE_EDGE_COLOR)

    def _draw_bird(self, frame: np.ndarray, scale: int) -> None:
        center_x = int(self.physics.bird_x * scale)
        center_y = int(self.bird_y * scale)
        radius = max(1, int(self.physics.bird_radius * scale))
        y_grid, x_grid = np.ogrid[: frame.shape[0], : frame.shape[1]]
        circle = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius**2
        frame[circle] = BIRD_COLOR

        wing_center_x = center_x - radius // 4
        wing_center_y = center_y + radius // 4
        wing_radius = max(1, radius // 2)
        wing = (x_grid - wing_center_x) ** 2 + (y_grid - wing_center_y) ** 2 <= wing_radius**2
        frame[wing] = BIRD_WING_COLOR

        eye_radius = max(1, radius // 4)
        eye_center_x = center_x + radius // 3
        eye_center_y = center_y - radius // 4
        eye = (x_grid - eye_center_x) ** 2 + (y_grid - eye_center_y) ** 2 <= eye_radius**2
        pupil = (x_grid - (eye_center_x + eye_radius // 3)) ** 2 + (y_grid - eye_center_y) ** 2 <= max(
            1, eye_radius // 2
        ) ** 2
        frame[eye] = EYE_COLOR
        frame[pupil] = PUPIL_COLOR

    @classmethod
    def _draw_score_marks(cls, frame: np.ndarray, score: int, scale: int) -> None:
        digits = str(max(0, score))
        pixel = max(1, scale * 2)
        digit_width = 3 * pixel
        digit_spacing = pixel
        total_width = len(digits) * digit_width + max(0, len(digits) - 1) * digit_spacing
        left = (frame.shape[1] - total_width) // 2
        top = 12 * scale

        for digit in digits:
            bitmap = DIGIT_BITMAPS[digit]
            for row_index, row in enumerate(bitmap):
                for col_index, bit in enumerate(row):
                    if bit == "1":
                        x0 = left + col_index * pixel
                        y0 = top + row_index * pixel
                        cls._draw_rect(frame, x0, y0, x0 + pixel, y0 + pixel, SCORE_COLOR)
            left += digit_width + digit_spacing
