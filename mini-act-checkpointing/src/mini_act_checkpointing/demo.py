"""Command-line demo for the mini checkpoint implementation."""

from __future__ import annotations

import copy

import torch
from torch import nn

from .checkpoint import checkpoint
from .memory import count_forward_saved_tensors


class TinyBlock(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(width * 2, width),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _device() -> torch.device:
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().norm().cpu())
    return total


def main() -> None:
    device = _device()
    torch.manual_seed(7)
    base = TinyBlock().to(device)
    checked = copy.deepcopy(base).to(device)
    x = torch.randn(16, 64, device=device, requires_grad=True)
    x_checked = x.detach().clone().requires_grad_(True)

    torch.manual_seed(123)
    normal_out, normal_stats = count_forward_saved_tensors(lambda: base(x).square().mean())
    normal_out.backward()

    torch.manual_seed(123)
    checked_out, checked_stats = count_forward_saved_tensors(
        lambda: checkpoint(checked, x_checked).square().mean()
    )
    checked_out.backward()

    print(f"device: {device}")
    print(f"normal saved tensors: {normal_stats.count} ({normal_stats.mib:.4f} MiB)")
    print(
        "checkpoint forward saved tensors: "
        f"{checked_stats.count} ({checked_stats.mib:.4f} MiB)"
    )
    print(f"normal grad norm: {_grad_norm(base):.6f}")
    print(f"checkpoint grad norm: {_grad_norm(checked):.6f}")
    print(
        "output close:",
        torch.allclose(normal_out.detach().cpu(), checked_out.detach().cpu(), atol=1e-6),
    )


if __name__ == "__main__":
    main()
