"""Tests for :mod:`moe.experts`.

These verify that experts are independent (perturbing an unused expert cannot
change the output), that dropout is training-only, and — most importantly — that
the readable naive dispatch and the fused batched dispatch produce identical
results, since the batched path is the one that runs in production.
"""

from __future__ import annotations

import pytest
import torch

from moe.config import MoEConfig
from moe.experts import Expert, ExpertBank, expert_ffn
from moe.router import TopKRouter


def _all_to_expert0(
    num_tokens: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dispatch weights/indices that route every token to expert 0."""
    dispatch = torch.zeros(num_tokens, num_experts, 1)
    dispatch[:, 0, 0] = 1.0
    indices = torch.zeros(num_tokens, 1, dtype=torch.long)
    return dispatch, indices


def test_expert_output_shape() -> None:
    """A single expert maps ``[N, d_model]`` to ``[N, d_model]``."""
    expert = Expert(MoEConfig(d_model=8, d_ff=16))
    out = expert(torch.randn(5, 8))
    assert out.shape == (5, 8)


@pytest.mark.parametrize("activation", ["gelu", "swiglu", "relu2"])
def test_activation_functions(activation: str) -> None:
    """Every supported activation yields a finite forward pass."""
    expert = Expert(MoEConfig(d_model=8, d_ff=16, activation=activation))
    out = expert(torch.randn(4, 8))
    assert out.shape == (4, 8)
    assert torch.isfinite(out).all()


def test_expert_independence() -> None:
    """Perturbing an *unused* expert must not change the output.

    Why it matters: experts must be truly independent; a shared buffer or a stray
    in-place op would couple them and this test would catch it.
    """
    config = MoEConfig(d_model=8, d_ff=16, num_experts=4, top_k=1, drop_tokens=False)
    bank = ExpertBank(config).eval()
    x = torch.randn(6, 8)
    dispatch, indices = _all_to_expert0(6, 4)

    before = bank(x, dispatch, indices)
    with torch.no_grad():
        bank.experts[1].w1.weight.add_(100.0)  # expert 1 receives no tokens
    after = bank(x, dispatch, indices)
    assert torch.equal(before, after)


def test_dropout_training_only() -> None:
    """Expert dropout makes the output stochastic in train mode, fixed in eval."""
    config = MoEConfig(
        d_model=8,
        d_ff=16,
        num_experts=2,
        top_k=1,
        expert_dropout=0.5,
        drop_tokens=False,
    )
    bank = ExpertBank(config)
    x = torch.randn(10, 8)
    dispatch, indices = _all_to_expert0(10, 2)

    bank.train()
    train_runs = torch.stack([bank(x, dispatch, indices) for _ in range(50)])
    bank.eval()
    eval_runs = torch.stack([bank(x, dispatch, indices) for _ in range(50)])

    assert train_runs.std(dim=0).max().item() > 0.0
    assert eval_runs.std(dim=0).max().item() == 0.0


@pytest.mark.parametrize(
    "activation, use_bias",
    [("gelu", True), ("swiglu", False), ("swiglu", True), ("relu2", True)],
)
def test_dispatch_strategies_agree(activation: str, use_bias: bool) -> None:
    """Naive and batched dispatch agree to 1e-4 (covers SwiGLU and bias-free).

    Why it matters: production uses the batched path; it must be a numerically
    faithful reimplementation of the readable naive reference.
    """
    config = MoEConfig(
        d_model=16,
        d_ff=32,
        num_experts=4,
        top_k=2,
        activation=activation,
        use_bias=use_bias,
        drop_tokens=False,
        expert_dropout=0.0,
    )
    bank = ExpertBank(config).eval()
    router = TopKRouter(config).eval()
    x = torch.randn(20, 16)
    routed = router(x)

    config.dispatch_strategy = "naive"
    out_naive = bank(x, routed.dispatch_weights, routed.expert_indices)
    config.dispatch_strategy = "batch"
    out_batch = bank(x, routed.dispatch_weights, routed.expert_indices)

    assert torch.allclose(out_naive, out_batch, atol=1e-4)


def test_batch_dispatch_reports_overflow() -> None:
    """The bank records dropped assignments when capacity is exceeded."""
    config = MoEConfig(
        d_model=4,
        d_ff=8,
        num_experts=2,
        top_k=1,
        capacity_factor=1.0,
        drop_tokens=True,
        dispatch_strategy="batch",
    )
    bank = ExpertBank(config).eval()
    x = torch.randn(10, 4)
    # Route all 10 tokens to expert 0; capacity = ceil(1.0*1*10/2) = 5 => 5 drop.
    dispatch, indices = _all_to_expert0(10, 2)
    bank(x, dispatch, indices)
    assert bank.last_overflow_tokens == 5
    assert bank.last_dispatched_tokens == 10


def test_expert_ffn_input_validation() -> None:
    """The shared kernel rejects unknown activations and SwiGLU without a gate."""
    x = torch.randn(3, 4)
    w1 = torch.randn(8, 4)
    w2 = torch.randn(4, 8)
    with pytest.raises(ValueError, match="Unknown activation"):
        expert_ffn(x, w1, None, w2, None, None, None, "tanh", 0.0, False)
    with pytest.raises(ValueError, match="requires a gate weight"):
        expert_ffn(x, w1, None, w2, None, None, None, "swiglu", 0.0, False)


def test_expert_bank_rejects_non_2d() -> None:
    """The bank requires a 2-D ``[num_tokens, d_model]`` input."""
    bank = ExpertBank(MoEConfig(d_model=8, d_ff=16, num_experts=2, top_k=1))
    dispatch, indices = _all_to_expert0(3, 2)
    with pytest.raises(ValueError, match="\\[num_tokens, d_model\\]"):
        bank(torch.randn(1, 3, 8), dispatch, indices)


def test_extra_repr_strings() -> None:
    """``extra_repr`` summaries mention the key hyperparameters."""
    config = MoEConfig(d_model=8, d_ff=16, num_experts=3, top_k=1)
    assert "d_ff=16" in Expert(config).extra_repr()
    assert "num_experts=3" in ExpertBank(config).extra_repr()
