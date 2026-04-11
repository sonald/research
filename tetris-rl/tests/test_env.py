import numpy as np

from tetris_rl.config import EnvConfig
from tetris_rl.env import ACTION_TO_INDEX, TetrisEnv
from tetris_rl.pieces import PIECE_NAMES


def test_reset_observation_shapes() -> None:
    env = TetrisEnv()
    observation, info = env.reset(seed=0)

    assert observation["board"].shape == (3, 20, 10)
    assert observation["meta"].shape == (30,)
    assert observation["action_mask"].shape == (8,)
    assert observation["action_mask"].dtype == bool
    assert info["score"] == 0
    assert info["lines_cleared_total"] == 0


def test_seed_reproducibility_and_first_bag_is_permutation() -> None:
    env_a = TetrisEnv()
    env_b = TetrisEnv()
    env_a.reset(seed=123)
    env_b.reset(seed=123)

    first_bag_a = [env_a.current_piece, *env_a.piece_queue[:6]]
    first_bag_b = [env_b.current_piece, *env_b.piece_queue[:6]]

    assert first_bag_a == first_bag_b
    assert sorted(first_bag_a) == sorted(PIECE_NAMES)


def test_hold_only_available_once_per_drop_cycle() -> None:
    env = TetrisEnv()
    observation, _ = env.reset(seed=0)

    observation, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX["hold"])
    assert not terminated
    assert not truncated
    assert env.can_hold is False
    assert info["invalid_action"] is False

    observation, reward, _, _, info = env.step(ACTION_TO_INDEX["hold"])
    assert info["invalid_action"] is True
    assert reward == 0.0


def test_soft_drop_and_hard_drop_progress_and_lock_piece() -> None:
    env = TetrisEnv()
    env.reset(seed=0)
    start_y = env.current_y

    env.step(ACTION_TO_INDEX["soft_drop"])
    assert env.current_y >= start_y + 1

    current_piece = env.current_piece
    env.step(ACTION_TO_INDEX["hard_drop"])
    assert env.current_piece != current_piece
    assert np.any(env.board != 0)


def test_basic_srs_wall_kick_changes_position_for_i_piece() -> None:
    env = TetrisEnv()
    env.reset(seed=0)
    env.current_piece = "I"
    env.current_rotation = 0
    env.current_x = -2
    env.current_y = 0

    rotated = env._attempt_rotation(delta=-1)

    assert rotated is True
    assert env.current_rotation == 3
    assert env.current_x == 0


def test_line_clear_updates_score_and_reward() -> None:
    env = TetrisEnv()
    env.reset(seed=0)
    env.board[-1, :] = 1
    env.board[-1, 4] = 0
    env.current_piece = "I"
    env.current_rotation = 1
    env.current_x = 2
    env.current_y = 15

    _, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX["hard_drop"])

    assert not terminated
    assert not truncated
    assert info["lines_cleared_step"] == 1
    assert info["score"] == 40
    assert reward > 1.0


def test_spawn_collision_terminates_episode() -> None:
    env = TetrisEnv()
    env.reset(seed=0)
    env.board[1:3, 3:7] = 1

    env._spawn_piece(piece_name="T", can_hold=True)

    assert env.terminated is True


def test_reward_shaping_penalizes_worse_board_features() -> None:
    config = EnvConfig()
    env = TetrisEnv(config=config)
    before = {"aggregate_height": 0.0, "holes": 0.0, "bumpiness": 0.0}
    after = {"aggregate_height": 5.0, "holes": 3.0, "bumpiness": 4.0}

    reward = env._compute_reward(
        before_features=before,
        after_features=after,
        lines_cleared=0,
        invalid_action=False,
        terminated=False,
    )

    assert reward < config.reward.survival_reward
