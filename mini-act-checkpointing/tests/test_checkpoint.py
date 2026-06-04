from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from mini_act_checkpointing import checkpoint, count_forward_saved_tensors


def devices() -> list[torch.device]:
    result = [torch.device("cpu")]
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        result.append(torch.device("mps"))
    return result


class Block(nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        self.lin1 = nn.Linear(width, width * 2)
        self.lin2 = nn.Linear(width * 2, width)

    def forward(self, x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.lin1(x))
        hidden = torch.nn.functional.dropout(hidden, p=0.25, training=True)
        return self.lin2(hidden).tanh() * scale


@pytest.mark.parametrize("device", devices())
def test_checkpoint_matches_regular_forward_and_gradients(device: torch.device) -> None:
    torch.manual_seed(1234)
    model = Block().to(device)
    model_checked = copy.deepcopy(model).to(device)
    x = torch.randn(8, 16, device=device, requires_grad=True)
    x_checked = x.detach().clone().requires_grad_(True)

    torch.manual_seed(99)
    y = model(x, scale=0.7).square().mean()
    y.backward()

    torch.manual_seed(99)
    y_checked = checkpoint(model_checked, x_checked, scale=0.7).square().mean()
    y_checked.backward()

    torch.testing.assert_close(y_checked.detach().cpu(), y.detach().cpu(), rtol=0, atol=1e-6)
    torch.testing.assert_close(x_checked.grad.cpu(), x.grad.cpu(), rtol=1e-5, atol=1e-6)
    for p_checked, p in zip(model_checked.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(p_checked.grad.cpu(), p.grad.cpu(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", devices())
def test_checkpoint_supports_nested_args_kwargs_and_static_outputs(
    device: torch.device,
) -> None:
    x = torch.randn(4, 3, device=device, requires_grad=True)
    bias = torch.randn(4, 3, device=device, requires_grad=True)

    def run(payload, *, bias, scale):
        y = (payload["x"] + bias).sin() * scale
        return {"loss_input": y, "label": "kept as a static Python output"}

    result = checkpoint(run, {"x": x}, bias=bias, scale=1.25)
    assert result["label"] == "kept as a static Python output"
    result["loss_input"].sum().backward()

    expected_x = x.detach().clone().requires_grad_(True)
    expected_bias = bias.detach().clone().requires_grad_(True)
    expected = run({"x": expected_x}, bias=expected_bias, scale=1.25)["loss_input"]
    expected.sum().backward()

    torch.testing.assert_close(x.grad.cpu(), expected_x.grad.cpu(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        bias.grad.cpu(), expected_bias.grad.cpu(), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("device", devices())
def test_rng_is_replayed_during_recomputation(device: torch.device) -> None:
    x = torch.ones(32, device=device, requires_grad=True)
    masks: list[torch.Tensor] = []

    def stochastic(inp: torch.Tensor) -> torch.Tensor:
        mask = (torch.rand_like(inp) > 0.5).to(inp.dtype)
        masks.append(mask.detach().cpu())
        return inp * mask

    torch.manual_seed(2024)
    checkpoint(stochastic, x).sum().backward()

    assert len(masks) == 2
    torch.testing.assert_close(masks[0], masks[1], rtol=0, atol=0)


def test_checkpoint_reduces_forward_saved_tensor_bytes_on_cpu() -> None:
    device = torch.device("cpu")
    torch.manual_seed(11)
    block = nn.Sequential(
        nn.Linear(32, 128),
        nn.GELU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
    ).to(device)
    x = torch.randn(16, 32, device=device, requires_grad=True)

    _normal_out, normal_stats = count_forward_saved_tensors(
        lambda: block(x).square().mean()
    )
    _checked_out, checked_stats = count_forward_saved_tensors(
        lambda: checkpoint(block, x).square().mean()
    )

    assert checked_stats.count < normal_stats.count
    assert checked_stats.bytes < normal_stats.bytes
