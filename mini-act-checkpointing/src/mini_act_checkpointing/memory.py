"""Small helpers for observing autograd saved tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class SavedTensorStats:
    count: int = 0
    bytes: int = 0

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        self.count += 1
        self.bytes += tensor.numel() * tensor.element_size()
        return tensor

    def unpack(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @property
    def mib(self) -> float:
        return self.bytes / (1024 * 1024)


def count_forward_saved_tensors(
    forward: Callable[[], torch.Tensor],
) -> tuple[torch.Tensor, SavedTensorStats]:
    """Run a forward pass and count tensors saved by autograd during it."""

    stats = SavedTensorStats()
    with torch.autograd.graph.saved_tensors_hooks(stats.pack, stats.unpack):
        output = forward()
    return output, stats
