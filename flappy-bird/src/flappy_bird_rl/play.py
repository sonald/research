"""Human-playable pygame entrypoint."""

from __future__ import annotations

import argparse
import os

import numpy as np

from .config import EnvConfig
from .core import FlappyBirdGame


def _import_pygame() -> object:
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError("pygame is required for play-flappy. Install project dependencies first.") from exc
    return pygame


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the Flappy Bird clone with pygame.")
    parser.add_argument("--scale", type=int, default=2, help="Render scale multiplier.")
    parser.add_argument("--fps", type=int, default=60, help="Target frames per second.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional hard frame limit for automated smoke runs. Zero means unlimited.",
    )
    args = parser.parse_args()

    pygame = _import_pygame()
    os.environ.setdefault("SDL_HINT_RENDER_SCALE_QUALITY", "0")
    pygame.init()
    try:
        pygame.font.init()
        font = pygame.font.Font(None, 28)
        small_font = pygame.font.Font(None, 22)
    except Exception:  # pragma: no cover - depends on platform pygame build.
        font = None
        small_font = None

    config = EnvConfig(render_scale=args.scale)
    game = FlappyBirdGame(config)
    game.reset()

    width = config.physics.screen_width * config.render_scale
    height = config.physics.screen_height * config.render_scale
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Flappy Bird RL")
    clock = pygame.time.Clock()

    state = "start"
    frame_count = 0
    running = True
    while running:
        flap = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    if state in {"start", "game_over"}:
                        game.reset()
                        state = "running"
                    flap = True
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if state in {"start", "game_over"}:
                    game.reset()
                    state = "running"
                flap = True

        if state == "running":
            game.step(flap=flap)
            if game.terminated:
                state = "game_over"

        frame = game.render_rgb_array()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surface, (0, 0))

        if state == "start":
            _blit_center(screen, font, "Press Space to Start", height // 2 - 20)
            _blit_center(screen, small_font, "Space / Up / Click to flap", height // 2 + 10)
        elif state == "game_over":
            _blit_center(screen, font, "Game Over", height // 2 - 20)
            _blit_center(screen, small_font, f"Collision: {game.collision_reason}", height // 2 + 6)
            _blit_center(screen, small_font, "Press Space to Restart", height // 2 + 28)

        pygame.display.flip()
        clock.tick(args.fps)

        frame_count += 1
        if args.max_frames > 0 and frame_count >= args.max_frames:
            running = False

    pygame.quit()


def _blit_center(screen: object, font: object | None, text: str, y: int) -> None:
    if font is None:
        return
    surface = font.render(text, True, (20, 20, 20))
    rect = surface.get_rect(center=(screen.get_width() // 2, y))
    screen.blit(surface, rect)


if __name__ == "__main__":
    main()
