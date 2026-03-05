from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None,
    epoch: int,
    best_loss: float,
    config: dict[str, Any],
) -> None:
    """Save model and optimizer states with training metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "config": dict(config),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load checkpoint into model and optional optimizer."""
    path = Path(path)
    payload = torch.load(path, map_location=map_location)

    model.load_state_dict(payload["model_state"])

    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])

    return {
        "epoch": int(payload.get("epoch", 0)),
        "best_loss": float(payload.get("best_loss", float("inf"))),
        "config": payload.get("config", {}),
    }
