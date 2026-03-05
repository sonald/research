# PyTorch Modern VAE (uv-managed)

This project reproduces the VAE architecture and training objective from:

- [Building a Modern Variational Autoencoder (VAE) from Scratch](https://maurocomi.com/blog/vae.html)
- [maurock/vaex](https://github.com/maurock/vaex)

## Features

- Conv + ResBlock + GroupNorm VAE in NCHW format.
- Reparameterization trick with latent feature maps.
- Loss: `lambda_rec * L1 + lambda_ssim * (1 - SSIM) + lambda_kl * KL`.
- Hugging Face dataset loading from remote or local `save_to_disk` artifacts.
- Resume-capable training with `last.pt` and `best.pt` checkpoints.

## Setup

```bash
uv sync
```

## Prepare Dataset

```bash
uv run vae-prepare --config configs/config.yaml
```

This downloads configured splits from Hugging Face and stores them under:

`<data_dir>/<path>/<split>`

Using the default config, that becomes: `data/zzsi/afhq64_16k/train`.

## Train

```bash
uv run vae-train --config configs/config.yaml
```

Optional smoke run:

```bash
uv run vae-train --config configs/config.yaml --max-steps 2
```

Resume from an existing checkpoint file:

```bash
uv run vae-train --config configs/config.yaml --resume checkpoints/my_weights/last.pt
```

## Test

```bash
uv run pytest -q
```
