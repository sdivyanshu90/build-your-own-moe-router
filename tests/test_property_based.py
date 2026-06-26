"""Property-based tests (Hypothesis) for router invariants.

Worked examples pin specific cases; these check that core invariants hold across
a wide swathe of randomly-generated valid configurations and inputs — the kind of
coverage that catches dimension- or dtype-dependent bugs unit tests miss.
"""

from __future__ import annotations

import hypothesis.strategies as st
import torch
from hypothesis import given, settings

from moe.config import MoEConfig
from moe.losses import auxiliary_load_balancing_loss, router_z_loss
from moe.router import TopKRouter

# Building real nn.Modules per example is not free; cap examples and drop the
# per-example deadline so slow-but-valid draws do not flake.
_SETTINGS = settings(max_examples=25, deadline=None)


@st.composite
def _router_case(draw: st.DrawFn) -> tuple[MoEConfig, torch.Tensor]:
    """Draw a valid config plus a matching random input tensor."""
    num_experts = draw(st.integers(min_value=2, max_value=8))
    top_k = draw(st.integers(min_value=1, max_value=num_experts))
    d_model = draw(st.integers(min_value=16, max_value=64))
    batch = draw(st.integers(min_value=1, max_value=4))
    seq = draw(st.integers(min_value=1, max_value=16))
    config = MoEConfig(
        d_model=d_model,
        d_ff=2 * d_model,
        num_experts=num_experts,
        top_k=top_k,
    )
    x = torch.randn(batch, seq, d_model)
    return config, x


@given(case=_router_case())
@_SETTINGS
def test_router_outputs_are_finite(case: tuple[MoEConfig, torch.Tensor]) -> None:
    """For any valid config/input, all router outputs are finite."""
    config, x = case
    out = TopKRouter(config)(x)
    assert torch.isfinite(out.router_logits).all()
    assert torch.isfinite(out.dispatch_weights).all()
    assert torch.isfinite(out.router_probs).all()
    assert torch.isfinite(out.aux_loss).all()


@given(case=_router_case())
@_SETTINGS
def test_aux_loss_non_negative(case: tuple[MoEConfig, torch.Tensor]) -> None:
    """The auxiliary load-balancing loss is always non-negative."""
    config, x = case
    out = TopKRouter(config)(x)
    assert out.aux_loss.item() >= 0.0


@given(
    rows=st.integers(min_value=1, max_value=16),
    cols=st.integers(min_value=2, max_value=8),
    scale=st.floats(min_value=0.0, max_value=1e3),
)
@_SETTINGS
def test_z_loss_non_negative(rows: int, cols: int, scale: float) -> None:
    """The z-loss is a mean of squares and is non-negative for any logits."""
    logits = torch.randn(rows, cols) * scale
    assert router_z_loss(logits).item() >= 0.0


@given(case=_router_case())
@_SETTINGS
def test_topk_sparsity_invariant(case: tuple[MoEConfig, torch.Tensor]) -> None:
    """Exactly ``min(top_k, num_experts)`` experts are selected per token."""
    config, x = case
    out = TopKRouter(config)(x)
    k = min(config.top_k, config.num_experts)
    nonzero = (out.dispatch_weights.squeeze(-1) != 0).sum(dim=-1)
    assert torch.all(nonzero == k)


@given(
    num_experts=st.integers(min_value=2, max_value=12),
    tokens=st.integers(min_value=1, max_value=32),
)
@_SETTINGS
def test_aux_loss_function_non_negative(num_experts: int, tokens: int) -> None:
    """The standalone aux-loss function is non-negative for random inputs."""
    probs = torch.softmax(torch.randn(tokens, num_experts), dim=-1)
    indices = torch.randint(0, num_experts, (tokens, 1))
    loss = auxiliary_load_balancing_loss(probs, indices, num_experts)
    assert loss.item() >= 0.0
