import numpy as np

from flappy_bird_rl.config import EnvConfig
from flappy_bird_rl.core import FlappyBirdGame, PipePair
from flappy_bird_rl.env import ACTION_FLAP, ACTION_NOOP, FlappyBirdEnv


def test_reset_observation_shape_and_info() -> None:
    env = FlappyBirdEnv()
    observation, info = env.reset(seed=0)

    assert observation.shape == (8,)
    assert observation.dtype == np.float32
    assert info["score"] == 0
    assert info["pipes_cleared"] == 0
    assert info["seed"] == 0


def test_seed_reproducibility() -> None:
    env_a = FlappyBirdEnv()
    env_b = FlappyBirdEnv()
    env_a.reset(seed=123)
    env_b.reset(seed=123)

    gaps_a = [pipe.gap_y for pipe in env_a.game.pipes]
    gaps_b = [pipe.gap_y for pipe in env_b.game.pipes]
    assert gaps_a == gaps_b


def test_flap_changes_velocity() -> None:
    env = FlappyBirdEnv()
    env.reset(seed=0)
    env.step(ACTION_FLAP)

    assert env.game.bird_velocity < 0.0


def test_passing_pipe_updates_score_once() -> None:
    env = FlappyBirdEnv()
    env.reset(seed=0)
    pipe = env.game.pipes[0]
    pipe.x = env.game.physics.bird_x - env.game.physics.pipe_width - env.game.physics.bird_radius - 1.0
    pipe.scored = False

    _, reward, terminated, truncated, info = env.step(ACTION_NOOP)

    assert not terminated
    assert not truncated
    assert info["score"] == 1
    assert info["passed_pipe"] is True
    assert reward > env.config.reward.pipe_reward

    _, _, _, _, info = env.step(ACTION_NOOP)
    assert info["score"] == 1


def test_floor_collision_terminates_episode() -> None:
    env = FlappyBirdEnv()
    env.reset(seed=0)
    env.game.bird_y = env.game.play_height - env.game.physics.bird_radius - 0.5
    env.game.bird_velocity = 10.0

    _, reward, terminated, truncated, info = env.step(ACTION_NOOP)

    assert terminated
    assert not truncated
    assert info["collision_reason"] == "floor"
    assert reward < 0.0


def test_pipe_collision_terminates_episode() -> None:
    env = FlappyBirdEnv()
    env.reset(seed=0)
    env.game.pipes = [
        PipePair(
            x=env.game.physics.bird_x - env.game.physics.pipe_width / 2.0,
            gap_y=220.0,
            scored=False,
        )
    ]
    env.game.bird_y = 40.0

    _, _, terminated, _, info = env.step(ACTION_NOOP)

    assert terminated
    assert info["collision_reason"] == "pipe"


def test_collision_does_not_award_score_or_pipe_reward() -> None:
    env = FlappyBirdEnv()
    env.reset(seed=0)
    env.game.pipes = [
        PipePair(
            x=env.game.physics.bird_x - env.game.physics.pipe_width - 1.0,
            gap_y=200.0,
            scored=False,
        )
    ]
    env.game.bird_y = 40.0
    env.game.bird_velocity = 0.0

    _, reward, terminated, _, info = env.step(ACTION_NOOP)

    assert terminated
    assert info["collision_reason"] == "pipe"
    assert info["score"] == 0
    assert info["passed_pipe"] is False
    assert reward < env.config.reward.pipe_reward


def test_render_rgb_array_shape_and_dtype() -> None:
    env = FlappyBirdEnv(EnvConfig(render_scale=1), render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (512, 288, 3)
    assert frame.dtype == np.uint8


def test_score_render_uses_decimal_digits_not_tally_marks() -> None:
    game = FlappyBirdGame(EnvConfig(render_scale=1))
    game.reset(seed=0)
    game.score = 10
    frame = game.render_rgb_array()

    score_band = np.all(frame[12:24] == np.asarray((255, 255, 255), dtype=np.uint8), axis=2)
    active_columns = np.flatnonzero(np.any(score_band, axis=0))
    assert len(active_columns) > 0
    assert int(active_columns[-1] - active_columns[0]) < 30


def test_alignment_reward_is_higher_near_gap_center() -> None:
    game = FlappyBirdGame()
    game.reset(seed=0)
    next_pipe = game.pipes[0]
    game.bird_y = next_pipe.gap_y
    centered = game.alignment_term()
    game.bird_y = next_pipe.gap_y + game.physics.pipe_gap
    offset = game.alignment_term()

    assert centered > offset
