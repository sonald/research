from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """Compute KL divergence between N(mu, sigma) and N(0, I)."""
    kl = 0.5 * torch.sum(mu.pow(2) + torch.exp(logvar) - logvar - 1.0, dim=(1, 2, 3))
    return kl.mean()


def _gaussian_kernel(filter_size: int, filter_sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    coords = torch.arange(filter_size, device=device, dtype=dtype) - (filter_size // 2)
    filt = torch.exp(-0.5 * (coords / filter_sigma).pow(2))
    filt = filt / filt.sum()
    kernel_2d = torch.outer(filt, filt)
    return kernel_2d.view(1, 1, filter_size, filter_size).repeat(channels, 1, 1, 1)


def ssim(
    a: Tensor,
    b: Tensor,
    *,
    max_val: float = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    return_map: bool = False,
) -> Tensor:
    """Compute SSIM over NCHW images using depthwise Gaussian convolution."""
    if a.shape != b.shape:
        raise ValueError(f"Input shapes must match, got {a.shape} and {b.shape}.")
    if a.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got ndim={a.ndim}.")

    channels = a.shape[1]
    kernel = _gaussian_kernel(filter_size, filter_sigma, channels, a.device, a.dtype)

    def conv(img: Tensor) -> Tensor:
        return F.conv2d(img, kernel, stride=1, padding=0, groups=channels)

    mu_a = conv(a)
    mu_b = conv(b)

    sigma_a_sq = conv(a * a) - mu_a * mu_a
    sigma_b_sq = conv(b * b) - mu_b * mu_b
    sigma_ab = conv(a * b) - mu_a * mu_b

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a_sq + sigma_b_sq + c2)

    ssim_map = numerator / (denominator + 1e-12)

    if return_map:
        return ssim_map
    return ssim_map.mean(dim=(1, 2, 3))


def vae_loss(
    x: Tensor,
    recon: Tensor,
    mu: Tensor,
    logvar: Tensor,
    lambda_rec: float,
    lambda_ssim: float,
    lambda_kl: float,
) -> dict[str, Tensor]:
    """Return weighted VAE loss terms and total."""
    l1_term = F.l1_loss(recon, x, reduction="mean")
    ssim_term = 1.0 - ssim(x, recon).mean()
    kl_term = kl_divergence(mu, logvar)
    total = lambda_rec * l1_term + lambda_ssim * ssim_term + lambda_kl * kl_term

    return {
        "total": total,
        "l1_loss": l1_term,
        "ssim_loss": ssim_term,
        "kl_loss": kl_term,
    }
