"""Tests for :class:`moe.layer.MoELayer`.

The layer is the integration point. These tests confirm end-to-end shape,
full gradient flow (router, every expert, and the input), finite auxiliary
losses, residual compatibility, eval determinism, and that all three presets run.
"""

from __future__ import annotations

import pytest
import torch

from moe.config import MoEConfig
from moe.layer import MoELayer


def test_end_to_end_shape() -> None:
    """The layer preserves the ``[batch, seq, d_model]`` shape."""
    layer = MoELayer(MoEConfig(d_model=16, d_ff=32, num_experts=4, top_k=2))
    out = layer(torch.randn(3, 7, 16))
    assert out.output.shape == (3, 7, 16)


def test_gradient_flow_end_to_end() -> None:
    """Backprop reaches the router, every expert, and the input.

    To guarantee every expert is exercised we route to all experts
    (``top_k == num_experts``); a real sparse layer would leave unused experts
    without a gradient, which is expected behaviour, not a bug.
    """
    config = MoEConfig(d_model=16, d_ff=32, num_experts=4, top_k=4)
    layer = MoELayer(config)
    x = torch.randn(2, 8, 16, requires_grad=True)

    out = layer(x)
    out.output.sum().backward()

    gate_grad = layer.router.w_gate.weight.grad
    assert gate_grad is not None and torch.any(gate_grad != 0)
    for expert in layer.expert_bank.experts:
        assert expert.w1.weight.grad is not None
        assert torch.any(expert.w1.weight.grad != 0)
    assert x.grad is not None and torch.any(x.grad != 0)


def test_losses_are_finite() -> None:
    """Auxiliary and z losses are finite scalars (never NaN/Inf)."""
    layer = MoELayer(MoEConfig(d_model=16, d_ff=32, num_experts=4, top_k=2))
    out = layer(torch.randn(2, 8, 16))
    assert torch.isfinite(out.aux_loss).item()
    assert torch.isfinite(out.z_loss).item()


def test_residual_compatibility() -> None:
    """The output can be added to a residual of the same shape and stays finite."""
    layer = MoELayer(MoEConfig(d_model=16, d_ff=32, num_experts=4, top_k=2))
    x = torch.randn(2, 8, 16)
    residual = x + layer(x).output
    assert residual.shape == x.shape
    assert torch.isfinite(residual).all()


def test_eval_determinism() -> None:
    """In eval mode the same input gives the same output despite noise/dropout."""
    config = MoEConfig(
        d_model=16,
        d_ff=32,
        num_experts=4,
        top_k=2,
        use_noisy_gating=True,
        expert_dropout=0.3,
        jitter_noise=0.1,
    )
    layer = MoELayer(config).eval()
    x = torch.randn(2, 8, 16)
    assert torch.equal(layer(x).output, layer(x).output)


@pytest.mark.parametrize(
    "config",
    [
        MoEConfig.switch_transformer(num_experts=4, d_model=32),
        MoEConfig.mixtral_style(num_experts=4, d_model=32),
        MoEConfig.gpt4_style(num_experts=4, d_model=32),
    ],
)
def test_preset_configs_run(config: MoEConfig) -> None:
    """Each published preset completes a forward pass without error."""
    layer = MoELayer(config)
    out = layer(torch.randn(2, 8, config.d_model))
    assert out.output.shape == (2, 8, config.d_model)


def test_routing_stats_lifecycle() -> None:
    """``get_routing_stats`` is empty before a forward and populated after."""
    layer = MoELayer(MoEConfig(d_model=16, d_ff=32, num_experts=4, top_k=2))
    assert layer.get_routing_stats() == {}

    layer(torch.randn(2, 8, 16))
    stats = layer.get_routing_stats()
    for key in (
        "entropy_mean",
        "imbalance_ratio",
        "overflow_fraction",
        "dead_experts",
        "expert_utilisation",
    ):
        assert key in stats
    assert len(stats["expert_utilisation"]) == 4


def test_dmodel_mismatch_raises() -> None:
    """An input whose trailing dim != d_model raises an informative error."""
    layer = MoELayer(MoEConfig(d_model=16, d_ff=32, num_experts=4, top_k=2))
    with pytest.raises(ValueError, match="must equal d_model"):
        layer(torch.randn(2, 8, 24))


def test_extra_repr_summary() -> None:
    """The layer's ``extra_repr`` surfaces the config summary."""
    layer = MoELayer(MoEConfig(d_model=16, d_ff=32, num_experts=4, top_k=2))
    assert "MoEConfig(" in layer.extra_repr()
