import torch

from vae_torch.losses import vae_loss


def test_vae_loss_outputs_are_finite_and_differentiable() -> None:
    x = torch.rand(2, 3, 256, 256)
    recon = torch.rand(2, 3, 256, 256, requires_grad=True)
    mu = torch.randn(2, 16, 32, 32, requires_grad=True)
    logvar = torch.randn(2, 16, 32, 32, requires_grad=True)

    losses = vae_loss(
        x=x,
        recon=recon,
        mu=mu,
        logvar=logvar,
        lambda_rec=1.0,
        lambda_ssim=0.2,
        lambda_kl=1e-5,
    )

    assert set(losses.keys()) == {"total", "l1_loss", "ssim_loss", "kl_loss"}
    assert losses["total"].requires_grad

    for value in losses.values():
        assert torch.isfinite(value)

    losses["total"].backward()
    assert recon.grad is not None
