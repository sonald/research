from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence
import copy

import torch
from torch import Tensor, nn


def _tensor_nbytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


@dataclass(frozen=True)
class Bucket:
    """A communication bucket containing same-index parameters from all replicas."""

    index: int
    parameter_indices: tuple[int, ...]
    bytes: int


class MiniReducer:
    """A tiny Reducer modeled after PyTorch DDP's bucketed gradient path.

    PyTorch's production Reducer launches asynchronous all-reduce when a bucket
    becomes ready during autograd. This teaching reducer keeps the same hook and
    bucket readiness machinery, then performs the all-reduce at the end of
    backward so it works on CPU and MPS without a distributed backend.
    """

    def __init__(self, replicas: Sequence[nn.Module], bucket_cap_mb: float) -> None:
        self.replicas = list(replicas)
        self.world_size = len(self.replicas)
        self.params_by_rank = [list(module.parameters()) for module in self.replicas]
        if not self.params_by_rank or not self.params_by_rank[0]:
            raise ValueError("MiniDDP requires at least one trainable parameter")

        reference_len = len(self.params_by_rank[0])
        if any(len(params) != reference_len for params in self.params_by_rank):
            raise ValueError("All replicas must have the same parameter structure")

        self.buckets = self._build_buckets(bucket_cap_mb)
        self.param_to_bucket = {
            param_index: bucket.index
            for bucket in self.buckets
            for param_index in bucket.parameter_indices
        }
        self._ready: dict[int, set[tuple[int, int]]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._sync_enabled = True
        self._register_autograd_hooks()

    def _build_buckets(self, bucket_cap_mb: float) -> list[Bucket]:
        cap_bytes = max(1, int(bucket_cap_mb * 1024 * 1024))
        reference_params = self.params_by_rank[0]
        buckets_reversed: list[list[int]] = []
        current: list[int] = []
        current_bytes = 0

        for param_index in reversed(range(len(reference_params))):
            param_bytes = _tensor_nbytes(reference_params[param_index])
            if current and current_bytes + param_bytes > cap_bytes:
                buckets_reversed.append(current)
                current = []
                current_bytes = 0
            current.append(param_index)
            current_bytes += param_bytes

        if current:
            buckets_reversed.append(current)

        buckets: list[Bucket] = []
        for bucket_index, indices in enumerate(reversed(buckets_reversed)):
            bucket_bytes = sum(_tensor_nbytes(reference_params[i]) for i in indices)
            buckets.append(Bucket(bucket_index, tuple(indices), bucket_bytes))
        return buckets

    def _register_autograd_hooks(self) -> None:
        for rank, params in enumerate(self.params_by_rank):
            for param_index, param in enumerate(params):
                self._handles.append(param.register_hook(self._make_hook(rank, param_index)))

    def _make_hook(self, rank: int, param_index: int):
        def hook(grad: Tensor) -> Tensor:
            bucket_index = self.param_to_bucket[param_index]
            self._ready.setdefault(bucket_index, set()).add((rank, param_index))
            return grad

        return hook

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        old = self._sync_enabled
        self._sync_enabled = False
        try:
            yield
        finally:
            self._sync_enabled = old

    def prepare_for_backward(self) -> None:
        self._ready.clear()

    def synchronize(self) -> None:
        if not self._sync_enabled:
            return

        for bucket in self.buckets:
            expected = {
                (rank, param_index)
                for rank in range(self.world_size)
                for param_index in bucket.parameter_indices
            }
            missing = expected - self._ready.get(bucket.index, set())
            if missing:
                readable = ", ".join(f"rank{r}:p{i}" for r, i in sorted(missing))
                raise RuntimeError(
                    f"Bucket {bucket.index} is not ready; missing gradients for {readable}. "
                    "This mini implementation expects every parameter to participate."
                )

            for param_index in bucket.parameter_indices:
                grads = [
                    self.params_by_rank[rank][param_index].grad
                    for rank in range(self.world_size)
                ]
                if any(grad is None for grad in grads):
                    raise RuntimeError(f"Parameter {param_index} has no gradient on at least one replica")

                averaged = torch.stack([grad.detach().to("cpu") for grad in grads]).mean(dim=0)
                for rank, params in enumerate(self.params_by_rank):
                    target = averaged.to(params[param_index].device)
                    params[param_index].grad = target.clone()

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class MiniDDP(nn.Module):
    """A readable, testable DDP clone for one Python process.

    Each rank owns a full model replica. The user shards inputs, computes one
    local loss per rank, calls ``backward(losses)``, then steps local optimizers.
    Since every replica starts from identical parameters and receives identical
    averaged gradients, all replicas remain synchronized after optimizer steps.
    """

    def __init__(
        self,
        module: nn.Module,
        world_size: int = 2,
        *,
        device: str | torch.device = "cpu",
        bucket_cap_mb: float = 0.25,
        broadcast_buffers: bool = True,
    ) -> None:
        super().__init__()
        if world_size < 1:
            raise ValueError("world_size must be >= 1")

        self.world_size = world_size
        self.device = torch.device(device)
        self.broadcast_buffers = broadcast_buffers
        self.replicas = nn.ModuleList(
            [copy.deepcopy(module).to(self.device) for _ in range(world_size)]
        )
        self.broadcast_parameters_and_buffers()
        self.reducer = MiniReducer(self.replicas, bucket_cap_mb=bucket_cap_mb)

    def broadcast_parameters_and_buffers(self) -> None:
        source_state = {
            key: value.detach().clone()
            for key, value in self.replicas[0].state_dict().items()
        }
        for replica in self.replicas[1:]:
            replica.load_state_dict(source_state)

    def forward(self, rank: int, *args, **kwargs):
        if rank < 0 or rank >= self.world_size:
            raise IndexError(f"rank must be in [0, {self.world_size})")
        if self.broadcast_buffers:
            self._broadcast_buffers()
        return self.replicas[rank](*args, **kwargs)

    def _broadcast_buffers(self) -> None:
        source_buffers = dict(self.replicas[0].named_buffers())
        for replica in self.replicas[1:]:
            for name, buffer in replica.named_buffers():
                buffer.copy_(source_buffers[name].to(buffer.device))

    def backward(self, losses: Sequence[Tensor] | Tensor) -> None:
        if isinstance(losses, Tensor):
            losses = [losses]
        if len(losses) != self.world_size:
            raise ValueError(f"Expected {self.world_size} losses, got {len(losses)}")

        self.reducer.prepare_for_backward()
        for loss in losses:
            loss.backward()
        self.reducer.synchronize()

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        with self.reducer.no_sync():
            yield

    def optimizers(self, optimizer_cls, **kwargs):
        return [optimizer_cls(replica.parameters(), **kwargs) for replica in self.replicas]

    def assert_replicas_equal(self, *, atol: float = 1e-6) -> None:
        reference = self.replicas[0].state_dict()
        for rank, replica in enumerate(self.replicas[1:], start=1):
            for name, tensor in replica.state_dict().items():
                if not torch.allclose(reference[name].cpu(), tensor.cpu(), atol=atol):
                    raise AssertionError(f"Replica {rank} diverged at {name}")


def shard_batch(tensor: Tensor, world_size: int) -> list[Tensor]:
    """Split the leading batch dimension the way DistributedSampler would."""

    if tensor.shape[0] % world_size != 0:
        raise ValueError("Batch size must be divisible by world_size for this teaching helper")
    return list(tensor.chunk(world_size, dim=0))
