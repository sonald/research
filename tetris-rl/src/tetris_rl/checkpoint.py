"""Checkpoint helpers for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    algo: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    step: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algo": algo,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "step": int(step),
        "extra": extra or {},
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device)
