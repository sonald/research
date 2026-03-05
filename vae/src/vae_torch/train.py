from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from vae_torch.checkpoint import load_checkpoint, save_checkpoint
from vae_torch.data import HFDataManager, build_train_dataloader, read_config
from vae_torch.losses import vae_loss
from vae_torch.model import VAE


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_resume_file(path: str | None) -> Path | None:
    if not path:
        return None

    candidate = Path(path)
    if candidate.is_file():
        return candidate

    if candidate.is_dir():
        last_file = candidate / "last.pt"
        best_file = candidate / "best.pt"
        if last_file.exists():
            return last_file
        if best_file.exists():
            return best_file

    return candidate


def _build_model(config: dict[str, Any]) -> VAE:
    return VAE(
        latent_features=int(config.get("latent_features", 16)),
        kernel_size=3,
        strides=2,
        intermediate_features=tuple(config.get("intermediate_features", [32, 64, 128])),
    )


def train(config: dict[str, Any], device: torch.device, resume: str | None = None, max_steps: int | None = None) -> None:
    torch.manual_seed(int(config.get("seed", 42)))

    model = _build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))

    checkpoint_dir = Path("checkpoints") / str(config["output_name"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"

    start_epoch = 0
    best_loss = float("inf")

    resume_candidate = resume
    if resume_candidate is None and config.get("load_pretrained_weights", False):
        pretrained = str(config.get("pretrained_weights_path", "")).strip()
        if pretrained:
            candidate = Path(pretrained)
            if not candidate.is_absolute():
                candidate = Path("checkpoints") / pretrained
            resume_candidate = str(candidate)

    resume_file = _resolve_resume_file(resume_candidate)
    if resume_file is not None:
        if not resume_file.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_file}")
        meta = load_checkpoint(resume_file, model=model, optimizer=optimizer, map_location=device)
        start_epoch = int(meta["epoch"])
        best_loss = float(meta["best_loss"])
        print(f"Resumed from {resume_file} at epoch {start_epoch} with best_loss={best_loss:.6f}")

    manager = HFDataManager(config)
    from_disk = bool(config.get("from_disk", True))
    data_dir = config.get("data_dir", "data")
    splits = manager.load_splits(from_disk=from_disk, data_dir=data_dir)

    train_split = str(config.get("splits", ["train"])[0])
    if train_split not in splits:
        raise KeyError(f"Split '{train_split}' was not loaded. Available: {list(splits)}")

    train_loader = build_train_dataloader(
        dataset=splits[train_split],
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        height=int(config["height"]),
        width=int(config["width"]),
    )

    if len(train_loader) == 0:
        raise RuntimeError("Training dataloader is empty. Lower batch_size or disable drop_last.")

    num_epochs = int(config["num_epochs"])
    lambda_rec = float(config["lambda_rec"])
    lambda_ssim = float(config["lambda_ssim"])
    lambda_kl = float(config["lambda_kl"])

    global_step = 0
    stop_early = False

    for epoch in range(start_epoch, num_epochs):
        model.train()

        total_loss_sum = 0.0
        l1_loss_sum = 0.0
        ssim_loss_sum = 0.0
        kl_loss_sum = 0.0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for batch in progress:
            images = batch["image"].to(device)

            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(images)
            losses = vae_loss(
                x=images,
                recon=recon,
                mu=mu,
                logvar=logvar,
                lambda_rec=lambda_rec,
                lambda_ssim=lambda_ssim,
                lambda_kl=lambda_kl,
            )
            losses["total"].backward()
            optimizer.step()

            total_loss_sum += float(losses["total"].item())
            l1_loss_sum += float(losses["l1_loss"].item())
            ssim_loss_sum += float(losses["ssim_loss"].item())
            kl_loss_sum += float(losses["kl_loss"].item())
            num_batches += 1
            global_step += 1

            progress.set_postfix(loss=f"{losses['total'].item():.4f}")

            if max_steps is not None and global_step >= max_steps:
                stop_early = True
                break

        if num_batches == 0:
            raise RuntimeError("No training batches were processed.")

        mean_total = total_loss_sum / num_batches
        mean_l1 = l1_loss_sum / num_batches
        mean_ssim = ssim_loss_sum / num_batches
        mean_kl = kl_loss_sum / num_batches

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"total={mean_total:.6f} l1={mean_l1:.6f} ssim={mean_ssim:.6f} kl={mean_kl:.6f}"
        )

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            best_loss=best_loss,
            config=config,
        )

        if mean_total < best_loss:
            best_loss = mean_total
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                best_loss=best_loss,
                config=config,
            )

        if stop_early:
            print(f"Stopped early at global_step={global_step} due to --max-steps={max_steps}.")
            break

    print(f"Checkpoint(last): {last_path}")
    print(f"Checkpoint(best): {best_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PyTorch VAE.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config = read_config(args.config)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train(
        config=config,
        device=device,
        resume=args.resume,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
