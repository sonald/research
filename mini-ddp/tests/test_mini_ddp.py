from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from mini_ddp import MiniDDP, shard_batch


class ToyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 7), nn.ReLU(), nn.Linear(7, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_dense_reference(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> nn.Module:
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    opt.step()
    return model


def _train_mini_ddp(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> MiniDDP:
    ddp = MiniDDP(model, world_size=2, device=device, bucket_cap_mb=0.0001)
    opts = ddp.optimizers(torch.optim.SGD, lr=0.1)
    xs = shard_batch(x.to(device), 2)
    ys = shard_batch(y.to(device), 2)
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    losses = [F.cross_entropy(ddp(rank, xs[rank]), ys[rank]) for rank in range(2)]
    ddp.backward(losses)
    for opt in opts:
        opt.step()
    return ddp


def test_one_step_matches_single_model_large_batch_on_cpu() -> None:
    torch.manual_seed(11)
    base = ToyNet()
    reference = copy.deepcopy(base)
    x = torch.randn(8, 5)
    y = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])

    reference = _train_dense_reference(reference, x, y)
    ddp = _train_mini_ddp(base, x, y, torch.device("cpu"))

    for ref_param, ddp_param in zip(reference.parameters(), ddp.replicas[0].parameters()):
        assert torch.allclose(ref_param, ddp_param.cpu(), atol=1e-6)
    ddp.assert_replicas_equal()


def test_bucket_hooks_mark_every_parameter_ready() -> None:
    torch.manual_seed(3)
    ddp = MiniDDP(ToyNet(), world_size=2, bucket_cap_mb=0.0001)
    x = torch.randn(6, 5)
    y = torch.tensor([0, 1, 1, 0, 0, 1])

    xs = shard_batch(x, 2)
    ys = shard_batch(y, 2)
    losses = [F.cross_entropy(ddp(rank, xs[rank]), ys[rank]) for rank in range(2)]
    ddp.backward(losses)

    assert len(ddp.reducer.buckets) > 1
    for bucket in ddp.reducer.buckets:
        expected = {
            (rank, param_index)
            for rank in range(ddp.world_size)
            for param_index in bucket.parameter_indices
        }
        assert ddp.reducer._ready[bucket.index] == expected


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_mps_smoke_step_keeps_replicas_synchronized() -> None:
    torch.manual_seed(17)
    base = ToyNet()
    x = torch.randn(8, 5)
    y = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])
    ddp = _train_mini_ddp(base, x, y, torch.device("mps"))
    ddp.assert_replicas_equal(atol=1e-5)
