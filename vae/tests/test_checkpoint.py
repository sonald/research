import pytest
import torch

from vae_torch.checkpoint import load_checkpoint, save_checkpoint
from vae_torch.model import VAE


def test_checkpoint_roundtrip(tmp_path) -> None:
    model = VAE(latent_features=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ckpt_path = tmp_path / "roundtrip.pt"
    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        best_loss=1.234,
        config={"batch_size": 8},
    )

    new_model = VAE(latent_features=16)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)

    meta = load_checkpoint(ckpt_path, model=new_model, optimizer=new_optimizer)

    assert set(model.state_dict().keys()) == set(new_model.state_dict().keys())
    assert meta["epoch"] == 3
    assert meta["best_loss"] == pytest.approx(1.234)
    assert meta["config"]["batch_size"] == 8
