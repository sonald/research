from __future__ import annotations

import argparse
import copy

import torch
from torch import nn
import torch.nn.functional as F

from .core import MiniDDP, shard_batch


class ToyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 8),
            nn.Tanh(),
            nn.Linear(8, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def train_reference(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> nn.Module:
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    opt.step()
    return model


def train_mini_ddp(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> MiniDDP:
    ddp = MiniDDP(model, world_size=2, device=device, bucket_cap_mb=0.0002)
    opts = ddp.optimizers(torch.optim.SGD, lr=0.1)
    xs = shard_batch(x.to(device), ddp.world_size)
    ys = shard_batch(y.to(device), ddp.world_size)

    for opt in opts:
        opt.zero_grad(set_to_none=True)
    losses = [F.cross_entropy(ddp(rank, xs[rank]), ys[rank]) for rank in range(ddp.world_size)]
    trace = ddp.backward(losses)
    for opt in opts:
        opt.step()

    print("Reducer trace:")
    for event in trace:
        print(f"  - {event.step}: {event.detail}")
    return ddp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="auto")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = choose_device(args.device)
    base = ToyNet()
    reference = copy.deepcopy(base)
    x = torch.randn(8, 5)
    y = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])

    reference = train_reference(reference, x, y)
    ddp = train_mini_ddp(base, x, y, device)
    ddp.assert_replicas_equal(atol=1e-5 if device.type == "mps" else 1e-6)
    print(f"\nDevice: {device}")
    print(f"Max delta from single-model baseline: {ddp.max_parameter_delta_from(reference):.8f}")
    print("Replicas stayed synchronized after local optimizer steps.")


if __name__ == "__main__":
    main()
