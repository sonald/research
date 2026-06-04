from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from mini_ddp2 import MiniDDP, shard_batch


class ToyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 7), nn.ReLU(), nn.Linear(7, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BufferNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("scale", torch.tensor([1.0, 2.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.scale


class BranchNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.optional = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, *, use_optional: bool) -> torch.Tensor:
        h = torch.relu(self.shared(x))
        if use_optional:
            h = torch.relu(self.optional(h))
        return self.head(h)


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


def test_rank0_broadcast_copies_parameters_and_buffers() -> None:
    torch.manual_seed(19)
    ddp = MiniDDP(BufferNet(), world_size=3)

    with torch.no_grad():
        ddp.replicas[1].linear.weight.add_(10)
        ddp.replicas[2].scale.add_(7)
    ddp.broadcast_parameters_and_buffers()

    reference = ddp.replicas[0].state_dict()
    for replica in ddp.replicas[1:]:
        for name, tensor in replica.state_dict().items():
            assert torch.allclose(reference[name], tensor)


def test_bucket_construction_assigns_every_parameter_once() -> None:
    ddp = MiniDDP(ToyNet(), world_size=2, bucket_cap_mb=0.0001)
    assigned = [
        param_index
        for bucket in ddp.reducer.buckets
        for param_index in bucket.parameter_indices
    ]

    assert len(ddp.reducer.buckets) > 1
    assert sorted(assigned) == list(range(len(list(ddp.replicas[0].parameters()))))
    assert len(assigned) == len(set(assigned))


def test_autograd_hooks_mark_every_used_parameter_ready() -> None:
    torch.manual_seed(3)
    ddp = MiniDDP(ToyNet(), world_size=2, bucket_cap_mb=0.0001)
    x = torch.randn(6, 5)
    y = torch.tensor([0, 1, 1, 0, 0, 1])

    xs = shard_batch(x, 2)
    ys = shard_batch(y, 2)
    losses = [F.cross_entropy(ddp(rank, xs[rank]), ys[rank]) for rank in range(2)]
    ddp.backward(losses)

    for bucket in ddp.reducer.buckets:
        expected = {
            (rank, param_index)
            for rank in range(ddp.world_size)
            for param_index in bucket.parameter_indices
        }
        assert ddp.reducer._ready[bucket.index] == expected


def test_no_sync_accumulates_then_next_backward_synchronizes() -> None:
    torch.manual_seed(31)
    base = ToyNet()
    reference = copy.deepcopy(base)
    ddp = MiniDDP(base, world_size=2, bucket_cap_mb=0.0001)

    x1 = torch.randn(8, 5)
    y1 = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])
    x2 = torch.randn(8, 5)
    y2 = torch.tensor([1, 1, 0, 0, 1, 0, 1, 0])

    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.05)
    ref_opt.zero_grad(set_to_none=True)
    (F.cross_entropy(reference(x1), y1) + F.cross_entropy(reference(x2), y2)).backward()
    ref_opt.step()

    opts = ddp.optimizers(torch.optim.SGD, lr=0.05)
    for opt in opts:
        opt.zero_grad(set_to_none=True)

    x1s = shard_batch(x1, 2)
    y1s = shard_batch(y1, 2)
    with ddp.no_sync():
        losses = [F.cross_entropy(ddp(rank, x1s[rank]), y1s[rank]) for rank in range(2)]
        trace = ddp.backward(losses)
    assert trace[-1].step == "no_sync"

    x2s = shard_batch(x2, 2)
    y2s = shard_batch(y2, 2)
    losses = [F.cross_entropy(ddp(rank, x2s[rank]), y2s[rank]) for rank in range(2)]
    ddp.backward(losses)
    for opt in opts:
        opt.step()

    for ref_param, ddp_param in zip(reference.parameters(), ddp.replicas[0].parameters()):
        assert torch.allclose(ref_param, ddp_param.cpu(), atol=1e-6)
    ddp.assert_replicas_equal()


def test_unused_parameters_raise_by_default() -> None:
    torch.manual_seed(7)
    ddp = MiniDDP(BranchNet(), world_size=2)
    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 1, 0, 1, 0])
    xs = shard_batch(x, 2)
    ys = shard_batch(y, 2)
    losses = [
        F.cross_entropy(ddp(rank, xs[rank], use_optional=False), ys[rank])
        for rank in range(2)
    ]

    with pytest.raises(RuntimeError, match="did not receive a gradient"):
        ddp.backward(losses)


def test_allow_unused_parameters_skips_only_when_unused_on_every_rank() -> None:
    torch.manual_seed(13)
    ddp = MiniDDP(BranchNet(), world_size=2, allow_unused_parameters=True)
    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 1, 0, 1, 0])
    xs = shard_batch(x, 2)
    ys = shard_batch(y, 2)

    losses = [
        F.cross_entropy(ddp(rank, xs[rank], use_optional=False), ys[rank])
        for rank in range(2)
    ]
    trace = ddp.backward(losses)

    assert any(event.step == "unused" and "optional" in event.detail for event in trace)


def test_rank_divergent_parameter_usage_raises_even_when_unused_is_allowed() -> None:
    torch.manual_seed(23)
    ddp = MiniDDP(BranchNet(), world_size=2, allow_unused_parameters=True)
    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 1, 0, 1, 0])
    xs = shard_batch(x, 2)
    ys = shard_batch(y, 2)
    losses = [
        F.cross_entropy(ddp(0, xs[0], use_optional=True), ys[0]),
        F.cross_entropy(ddp(1, xs[1], use_optional=False), ys[1]),
    ]

    with pytest.raises(RuntimeError, match="used on ranks"):
        ddp.backward(losses)


def test_shard_batch_requires_equal_shards() -> None:
    with pytest.raises(ValueError, match="cannot be split"):
        shard_batch(torch.randn(5, 3), 2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_mps_smoke_step_keeps_replicas_synchronized() -> None:
    torch.manual_seed(17)
    base = ToyNet()
    x = torch.randn(8, 5)
    y = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])
    ddp = _train_mini_ddp(base, x, y, torch.device("mps"))
    ddp.assert_replicas_equal(atol=1e-5)
