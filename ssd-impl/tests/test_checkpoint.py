from __future__ import annotations

from pathlib import Path

import torch

from ssd_impl.checkpoint import load_training_checkpoint, resolve_resume_checkpoint, save_training_checkpoint
from tests.helpers import ToyLM


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = ToyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    checkpoint_dir = save_training_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=3,
        config={"training": {"max_steps": 5}},
    )

    restored_model = ToyLM()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lr_lambda=lambda _: 1.0)

    meta = load_training_checkpoint(
        checkpoint_dir,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert meta["step"] == 3
    assert resolve_resume_checkpoint(tmp_path) == checkpoint_dir

