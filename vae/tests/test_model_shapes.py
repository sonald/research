import torch

from vae_torch.model import VAE


def test_vae_shapes_for_256_input() -> None:
    model = VAE(latent_features=16, intermediate_features=(32, 64, 128))
    x = torch.randn(2, 3, 256, 256)

    recon, mu, logvar = model(x)

    assert recon.shape == x.shape
    assert mu.shape == (2, 16, 32, 32)
    assert logvar.shape == (2, 16, 32, 32)
