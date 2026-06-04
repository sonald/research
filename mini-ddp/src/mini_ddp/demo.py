from __future__ import annotations

import argparse

import torch
from torch import nn
import torch.nn.functional as F

from .core import MiniDDP, shard_batch


class TinyClassifier(nn.Module):
    def __init__(self, in_features: int = 8, hidden: int = 16, classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not available")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one tiny MiniDDP training job.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    torch.manual_seed(7)
    device = pick_device(args.device)
    model = TinyClassifier()
    ddp = MiniDDP(model, world_size=args.world_size, device=device, bucket_cap_mb=0.001)
    optimizers = ddp.optimizers(torch.optim.SGD, lr=0.08)

    for step in range(args.steps):
        x = torch.randn(12, 8, device=device)
        y = torch.randint(0, 3, (12,), device=device)
        xs = shard_batch(x, ddp.world_size)
        ys = shard_batch(y, ddp.world_size)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        losses = [
            F.cross_entropy(ddp(rank, xs[rank]), ys[rank])
            for rank in range(ddp.world_size)
        ]
        ddp.backward(losses)
        for opt in optimizers:
            opt.step()
        ddp.assert_replicas_equal(atol=1e-5 if device.type == "mps" else 1e-6)
        print(f"step={step:02d} device={device} mean_loss={torch.stack([l.detach().cpu() for l in losses]).mean().item():.4f}")

    print("MiniDDP replicas stayed synchronized.")


if __name__ == "__main__":
    main()
