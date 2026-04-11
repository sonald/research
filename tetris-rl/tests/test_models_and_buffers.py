import numpy as np
import torch

from tetris_rl.models import TetrisEncoder, mask_logits, masked_max
from tetris_rl.replay import ReplayBuffer


def test_encoder_batch_shapes() -> None:
    encoder = TetrisEncoder(hidden_dim=128)
    board = torch.randn(4, 3, 20, 10)
    meta = torch.randn(4, 30)

    encoded = encoder(board, meta)

    assert encoded.shape == (4, 128)


def test_masked_max_ignores_invalid_actions() -> None:
    logits = torch.tensor([[1.0, 100.0, 3.0]])
    mask = torch.tensor([[True, False, True]])

    masked = mask_logits(logits, mask)
    value = masked_max(logits, mask)

    assert masked[0, 1] < -1.0e20
    assert torch.allclose(value, torch.tensor([3.0]))


def test_replay_buffer_sample_shapes() -> None:
    replay = ReplayBuffer(capacity=8, board_shape=(3, 20, 10), meta_dim=30, action_dim=8)
    rng = np.random.default_rng(0)
    observation = {
        "board": np.zeros((3, 20, 10), dtype=np.float32),
        "meta": np.zeros((30,), dtype=np.float32),
        "action_mask": np.ones((8,), dtype=bool),
    }

    for index in range(6):
        replay.add(observation, action=index % 8, reward=float(index), next_observation=observation, done=False)

    batch = replay.sample(batch_size=4, rng=rng)

    assert batch["board"].shape == (4, 3, 20, 10)
    assert batch["meta"].shape == (4, 30)
    assert batch["actions"].shape == (4,)
    assert batch["next_action_mask"].shape == (4, 8)
