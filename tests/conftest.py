"""Shared pytest fixtures for the MoE test suite.

These fixtures provide reproducible configs, inputs and a lightly-trained layer so
individual tests stay short and deterministic. All randomness flows through an
explicit :class:`torch.Generator` or a locally-scoped ``manual_seed`` so tests do
not leak global RNG state into one another.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from moe import MoEConfig, MoELayer

#: Seed used by the reproducible-input fixture.
INPUT_SEED: int = 42


@pytest.fixture
def small_config() -> MoEConfig:
    """A tiny, fast config for unit tests: 4 experts, top-1, d_model=32."""
    return MoEConfig(
        d_model=32,
        d_ff=64,
        num_experts=4,
        top_k=1,
        capacity_factor=1.25,
        router_type="topk",
        activation="gelu",
    )


@pytest.fixture
def standard_config() -> MoEConfig:
    """A realistic config for integration-style tests: 8 experts, top-2."""
    return MoEConfig(
        d_model=512,
        d_ff=2048,
        num_experts=8,
        top_k=2,
        capacity_factor=1.25,
        router_type="topk",
        activation="gelu",
    )


@pytest.fixture
def random_input() -> Callable[..., torch.Tensor]:
    """Return a factory producing a reproducible ``[batch, seq, d_model]`` tensor.

    The factory seeds a private :class:`torch.Generator` so repeated calls with
    the same arguments yield identical tensors without disturbing global RNG.
    """

    def _make(config: MoEConfig, batch: int = 2, seq: int = 16) -> torch.Tensor:
        generator = torch.Generator().manual_seed(INPUT_SEED)
        return torch.randn(batch, seq, config.d_model, generator=generator)

    return _make


@pytest.fixture
def trained_layer(standard_config: MoEConfig) -> MoELayer:
    """A :class:`MoELayer` after 10 gradient steps (non-random-init state)."""
    torch.manual_seed(0)
    layer = MoELayer(standard_config)
    optimiser = torch.optim.Adam(layer.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(2, 16, standard_config.d_model, generator=generator)
    target = torch.randn(2, 16, standard_config.d_model, generator=generator)
    head = nn.Identity()
    for _ in range(10):
        out = layer(x)
        task = ((head(out.output) - target) ** 2).mean()
        loss = task + out.aux_loss + out.z_loss
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    layer.eval()
    return layer
