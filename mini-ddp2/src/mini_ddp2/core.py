from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence
import copy

import torch
from torch import Tensor, nn


def _tensor_nbytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _clone_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in state_dict.items()}


def _flatten_tensors(tensors: Sequence[Tensor]) -> Tensor:
    if not tensors:
        raise ValueError("Cannot flatten an empty tensor list")
    return torch.cat([tensor.detach().reshape(-1).to("cpu") for tensor in tensors])


def _unflatten_like(flat: Tensor, templates: Sequence[Tensor]) -> list[Tensor]:
    outputs: list[Tensor] = []
    offset = 0
    for template in templates:
        count = template.numel()
        chunk = flat[offset : offset + count].reshape_as(template)
        outputs.append(chunk.to(device=template.device, dtype=template.dtype).clone())
        offset += count
    if offset != flat.numel():
        raise ValueError("Flat tensor length does not match templates")
    return outputs


def shard_batch(tensor: Tensor, world_size: int, *, dim: int = 0) -> list[Tensor]:
    """Split a tensor into equal rank-local shards.

    Equal shards keep the DDP gradient average mathematically equivalent to a
    single large-batch mean loss. Uneven inputs are important in production DDP,
    but they add a separate join/weighting topic, so this teaching helper rejects
    them explicitly.
    """

    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if tensor.size(dim) % world_size != 0:
        raise ValueError(
            f"Dimension {dim} with size {tensor.size(dim)} cannot be split into "
            f"{world_size} equal shards"
        )
    return list(torch.chunk(tensor, chunks=world_size, dim=dim))


@dataclass(frozen=True)
class Bucket:
    index: int
    parameter_indices: tuple[int, ...]
    bytes: int


@dataclass(frozen=True)
class ReducerTraceEvent:
    step: str
    detail: str


class InMemoryProcessGroup:
    """Single-process stand-in for DDP collective communication.

    PyTorch DDP delegates communication to c10d ``ProcessGroup`` backends such
    as Gloo or NCCL. This class keeps only the collective semantics needed for a
    teaching implementation: rank-0 broadcast and all-reduce mean.
    """

    def __init__(self, world_size: int) -> None:
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        self.world_size = world_size
        self.trace: list[ReducerTraceEvent] = []

    def reset_trace(self) -> None:
        self.trace.clear()

    def record(self, step: str, detail: str) -> None:
        self.trace.append(ReducerTraceEvent(step=step, detail=detail))

    def broadcast_state_dict(self, replicas: Sequence[nn.Module], *, source_rank: int = 0) -> None:
        if len(replicas) != self.world_size:
            raise ValueError("replica count must match world_size")
        source = _clone_state_dict(replicas[source_rank].state_dict())
        for rank, replica in enumerate(replicas):
            if rank == source_rank:
                continue
            replica.load_state_dict(source)
        self.record("broadcast", f"copied rank {source_rank} state_dict to {self.world_size - 1} ranks")

    def all_reduce_mean(self, tensors_by_rank: Sequence[Tensor], *, bucket_index: int) -> Tensor:
        if len(tensors_by_rank) != self.world_size:
            raise ValueError("all_reduce_mean expects one tensor per rank")
        shapes = {tuple(tensor.shape) for tensor in tensors_by_rank}
        if len(shapes) != 1:
            raise ValueError(f"all_reduce_mean got mismatched shapes: {sorted(shapes)}")
        averaged = torch.stack([tensor.detach().to("cpu") for tensor in tensors_by_rank]).mean(dim=0)
        self.record(
            "all_reduce_mean",
            f"bucket {bucket_index}: averaged {averaged.numel()} flattened gradient values",
        )
        return averaged


class MiniReducer:
    """A small Reducer modeled on PyTorch DDP's gradient synchronization path."""

    def __init__(
        self,
        replicas: Sequence[nn.Module],
        process_group: InMemoryProcessGroup,
        *,
        bucket_cap_mb: float = 25.0,
        allow_unused_parameters: bool = False,
    ) -> None:
        self.replicas = list(replicas)
        self.process_group = process_group
        self.world_size = process_group.world_size
        self.allow_unused_parameters = allow_unused_parameters
        self._sync_enabled = True
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._ready: dict[int, set[tuple[int, int]]] = defaultdict(set)

        self.named_params_by_rank = [
            [(name, param) for name, param in replica.named_parameters() if param.requires_grad]
            for replica in self.replicas
        ]
        if not self.named_params_by_rank or not self.named_params_by_rank[0]:
            raise ValueError("MiniDDP requires at least one trainable parameter")
        self._validate_replicas()

        self.parameter_names = tuple(name for name, _ in self.named_params_by_rank[0])
        self.params_by_rank = [[param for _, param in named] for named in self.named_params_by_rank]
        self.buckets = self._build_buckets(bucket_cap_mb)
        self.param_to_bucket = {
            param_index: bucket.index
            for bucket in self.buckets
            for param_index in bucket.parameter_indices
        }
        self._register_autograd_hooks()

    def _validate_replicas(self) -> None:
        reference = self.named_params_by_rank[0]
        reference_signature = [
            (name, tuple(param.shape), param.dtype) for name, param in reference
        ]
        for rank, named_params in enumerate(self.named_params_by_rank[1:], start=1):
            signature = [(name, tuple(param.shape), param.dtype) for name, param in named_params]
            if signature != reference_signature:
                raise ValueError(f"Replica {rank} does not match rank 0 parameter structure")

    def _build_buckets(self, bucket_cap_mb: float) -> list[Bucket]:
        cap_bytes = max(1, int(bucket_cap_mb * 1024 * 1024))
        reference_params = self.params_by_rank[0]
        reversed_buckets: list[list[int]] = []
        current: list[int] = []
        current_bytes = 0

        for param_index in reversed(range(len(reference_params))):
            param_bytes = _tensor_nbytes(reference_params[param_index])
            if current and current_bytes + param_bytes > cap_bytes:
                reversed_buckets.append(current)
                current = []
                current_bytes = 0
            current.append(param_index)
            current_bytes += param_bytes

        if current:
            reversed_buckets.append(current)

        buckets: list[Bucket] = []
        for bucket_index, indices in enumerate(reversed(reversed_buckets)):
            bucket_bytes = sum(_tensor_nbytes(reference_params[index]) for index in indices)
            buckets.append(Bucket(bucket_index, tuple(indices), bucket_bytes))
        return buckets

    def _register_autograd_hooks(self) -> None:
        for rank, params in enumerate(self.params_by_rank):
            for param_index, param in enumerate(params):
                self._handles.append(param.register_hook(self._make_hook(rank, param_index)))

    def _make_hook(self, rank: int, param_index: int):
        def hook(grad: Tensor) -> Tensor:
            bucket_index = self.param_to_bucket[param_index]
            self._ready[bucket_index].add((rank, param_index))
            self.process_group.record(
                "hook",
                f"rank {rank} produced grad for {self.parameter_names[param_index]} in bucket {bucket_index}",
            )
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
        self.process_group.reset_trace()
        self.process_group.record("prepare", "cleared bucket ready state before backward")

    def synchronize(self) -> None:
        if not self._sync_enabled:
            self.process_group.record("no_sync", "left rank-local gradients unsynchronized")
            return

        for bucket in self.buckets:
            active_indices = self._active_parameter_indices(bucket)
            if not active_indices:
                self.process_group.record("bucket_skip", f"bucket {bucket.index}: no active gradients")
                continue

            flat_grads_by_rank = []
            templates = [self.params_by_rank[0][index].grad for index in active_indices]
            if any(template is None for template in templates):
                names = [self.parameter_names[index] for index in active_indices]
                raise RuntimeError(f"Bucket {bucket.index} has missing rank 0 gradients for {names}")

            for rank in range(self.world_size):
                rank_grads = [self.params_by_rank[rank][index].grad for index in active_indices]
                if any(grad is None for grad in rank_grads):
                    names = [
                        self.parameter_names[index]
                        for index, grad in zip(active_indices, rank_grads)
                        if grad is None
                    ]
                    raise RuntimeError(
                        f"Bucket {bucket.index} is rank-divergent; rank {rank} is missing {names}"
                    )
                flat_grads_by_rank.append(_flatten_tensors([grad for grad in rank_grads if grad is not None]))

            averaged_flat = self.process_group.all_reduce_mean(
                flat_grads_by_rank,
                bucket_index=bucket.index,
            )
            averaged_grads = _unflatten_like(
                averaged_flat,
                [template for template in templates if template is not None],
            )
            for rank in range(self.world_size):
                for param_index, averaged_grad in zip(active_indices, averaged_grads):
                    param = self.params_by_rank[rank][param_index]
                    param.grad = averaged_grad.to(param.device).clone()
            self.process_group.record(
                "writeback",
                f"bucket {bucket.index}: wrote averaged gradients for {len(active_indices)} parameters",
            )

    def _active_parameter_indices(self, bucket: Bucket) -> list[int]:
        active: list[int] = []
        for param_index in bucket.parameter_indices:
            ready_ranks = {
                rank
                for rank in range(self.world_size)
                if (rank, param_index) in self._ready[bucket.index]
            }
            if len(ready_ranks) == self.world_size:
                active.append(param_index)
                continue
            if not ready_ranks and self.allow_unused_parameters:
                self.process_group.record(
                    "unused",
                    f"bucket {bucket.index}: skipped unused parameter {self.parameter_names[param_index]}",
                )
                continue

            missing = sorted(set(range(self.world_size)) - ready_ranks)
            name = self.parameter_names[param_index]
            if ready_ranks:
                raise RuntimeError(
                    f"Parameter {name} in bucket {bucket.index} was used on ranks "
                    f"{sorted(ready_ranks)} but missing on ranks {missing}. "
                    "DDP requires all ranks to follow the same parameter-usage graph."
                )
            raise RuntimeError(
                f"Parameter {name} in bucket {bucket.index} did not receive a gradient. "
                "Pass allow_unused_parameters=True only when the parameter is unused on every rank."
            )
        return active

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class MiniDDP(nn.Module):
    """Teaching implementation of the semantic core of DistributedDataParallel."""

    def __init__(
        self,
        module: nn.Module,
        world_size: int = 2,
        *,
        device: str | torch.device = "cpu",
        bucket_cap_mb: float = 25.0,
        broadcast_buffers: bool = True,
        allow_unused_parameters: bool = False,
    ) -> None:
        super().__init__()
        if world_size < 1:
            raise ValueError("world_size must be >= 1")

        self.world_size = world_size
        self.device = torch.device(device)
        self.broadcast_buffers = broadcast_buffers
        self.process_group = InMemoryProcessGroup(world_size)
        self.replicas = nn.ModuleList(
            [copy.deepcopy(module).to(self.device) for _ in range(world_size)]
        )
        self.broadcast_parameters_and_buffers()
        self.reducer = MiniReducer(
            self.replicas,
            self.process_group,
            bucket_cap_mb=bucket_cap_mb,
            allow_unused_parameters=allow_unused_parameters,
        )

    def broadcast_parameters_and_buffers(self) -> None:
        self.process_group.broadcast_state_dict(self.replicas, source_rank=0)

    def forward(self, rank: int, *args, **kwargs):
        if rank < 0 or rank >= self.world_size:
            raise IndexError(f"rank must be in [0, {self.world_size})")
        if self.broadcast_buffers:
            self._broadcast_buffers_from_rank0()
        return self.replicas[rank](*args, **kwargs)

    def _broadcast_buffers_from_rank0(self) -> None:
        source_buffers = dict(self.replicas[0].named_buffers())
        for replica in self.replicas[1:]:
            for name, buffer in replica.named_buffers():
                buffer.copy_(source_buffers[name].to(buffer.device))

    def backward(self, losses: Sequence[Tensor] | Tensor) -> list[ReducerTraceEvent]:
        if isinstance(losses, Tensor):
            losses = [losses]
        if len(losses) != self.world_size:
            raise ValueError(f"Expected {self.world_size} losses, got {len(losses)}")

        self.reducer.prepare_for_backward()
        for rank, loss in enumerate(losses):
            self.process_group.record("backward", f"rank {rank} backward started")
            loss.backward()
        self.reducer.synchronize()
        return list(self.process_group.trace)

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

    def max_parameter_delta_from(self, reference: nn.Module) -> float:
        max_delta = 0.0
        for ref_param, ddp_param in zip(reference.parameters(), self.replicas[0].parameters()):
            delta = (ref_param.detach().cpu() - ddp_param.detach().cpu()).abs().max().item()
            max_delta = max(max_delta, delta)
        return max_delta
