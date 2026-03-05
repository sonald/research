from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """Apply the reparameterization trick for latent sampling."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


class ResBlock(nn.Module):
    """Residual block with GroupNorm + SiLU + Conv repeated twice."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding)

        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        return x + residual


class Encoder(nn.Module):
    """Encoder matching the architecture from the reference implementation."""

    def __init__(
        self,
        latent_features: int,
        kernel_size: int = 3,
        strides: int = 2,
        intermediate_features: Iterable[int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        f0, f1, f2 = tuple(intermediate_features)
        padding = kernel_size // 2

        self.conv0 = nn.Conv2d(3, f0, kernel_size=kernel_size, stride=1, padding=padding)
        self.block1 = ResBlock(f0, kernel_size=kernel_size)

        self.conv2 = nn.Conv2d(f0, f1, kernel_size=kernel_size, stride=strides, padding=padding)
        self.block3 = ResBlock(f1, kernel_size=kernel_size)

        self.conv4 = nn.Conv2d(f1, f2, kernel_size=kernel_size, stride=strides, padding=padding)
        self.block5 = ResBlock(f2, kernel_size=kernel_size)

        self.conv6 = nn.Conv2d(f2, 2 * latent_features, kernel_size=kernel_size, stride=strides, padding=padding)
        self.latent_features = latent_features

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv0(x)
        x = self.block1(x)

        x = self.conv2(x)
        x = self.block3(x)

        x = self.conv4(x)
        x = self.block5(x)

        x = self.conv6(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder mirroring encoder depth with transposed convolutions."""

    def __init__(
        self,
        latent_features: int,
        kernel_size: int = 3,
        strides: int = 2,
        intermediate_features: Iterable[int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        f0, f1, f2 = tuple(intermediate_features)
        padding = kernel_size // 2
        output_padding = 1 if strides > 1 else 0

        self.deconv0 = nn.ConvTranspose2d(
            latent_features,
            f2,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
            output_padding=output_padding,
        )
        self.block1 = ResBlock(f2, kernel_size=kernel_size)

        self.deconv2 = nn.ConvTranspose2d(
            f2,
            f1,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
            output_padding=output_padding,
        )
        self.block3 = ResBlock(f1, kernel_size=kernel_size)

        self.deconv4 = nn.ConvTranspose2d(
            f1,
            f0,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
            output_padding=output_padding,
        )
        self.block5 = ResBlock(f0, kernel_size=kernel_size)

        self.deconv6 = nn.ConvTranspose2d(
            f0,
            3,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

    def forward(self, z: Tensor) -> Tensor:
        x = self.deconv0(z)
        x = self.block1(x)

        x = self.deconv2(x)
        x = self.block3(x)

        x = self.deconv4(x)
        x = self.block5(x)

        logits = self.deconv6(x)
        return torch.sigmoid(logits)


class VAE(nn.Module):
    """Variational Autoencoder returning reconstruction, mu, and logvar."""

    def __init__(
        self,
        latent_features: int,
        kernel_size: int = 3,
        strides: int = 2,
        intermediate_features: Iterable[int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            latent_features=latent_features,
            kernel_size=kernel_size,
            strides=strides,
            intermediate_features=intermediate_features,
        )
        self.decoder = Decoder(
            latent_features=latent_features,
            kernel_size=kernel_size,
            strides=strides,
            intermediate_features=intermediate_features,
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
