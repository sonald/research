from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def checkpoint_dir_for_step(checkpoints_root: str | Path, step: int) -> Path:
    return Path(checkpoints_root) / "last" / f"step-{step:04d}"


def _model_state_dict(model: Any) -> dict[str, torch.Tensor]:
    if hasattr(model, "state_dict"):
        return model.state_dict()
    raise TypeError("Model must expose state_dict()")


def save_training_checkpoint(
    checkpoints_root: str | Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    step: int,
    config: dict[str, Any],
) -> Path:
    save_dir = checkpoint_dir_for_step(checkpoints_root, step)
    save_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_dir / "adapter")

    payload = {
        "model_state": _model_state_dict(model),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step": int(step),
        "config": config,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    torch.save(payload, save_dir / "training_state.pt")
    metadata = {"step": int(step), "path": str(save_dir)}
    (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return save_dir


def resolve_resume_checkpoint(path: str | Path | None) -> Path | None:
    if path is None:
        return None

    target = Path(path)
    if target.is_file():
        return target.parent
    if (target / "training_state.pt").exists():
        return target
    last_dir = target / "last"
    if last_dir.exists():
        candidates = sorted(last_dir.glob("step-*"))
        if candidates:
            return candidates[-1]
    return None


def load_training_checkpoint(
    checkpoint_dir: str | Path,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir) / "training_state.pt"
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=False)

    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])

    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"])

    return {
        "step": int(payload.get("step", 0)),
        "config": payload.get("config", {}),
    }
