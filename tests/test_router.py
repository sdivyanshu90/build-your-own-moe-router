"""Tests for the routers in :mod:`moe.router`.

Routing correctness is the crux of an MoE layer: wrong shapes, wrong sparsity,
leaked train-time noise into eval, or NaNs from large logits would each silently
corrupt training. These tests pin down all of those.
"""

from __future__ import annotations

import pytest
import torch

from moe.config import MoEConfig
from moe.router import (
    ExpertChoiceRouter,
    SwitchRouter,
    TopKRouter,
    build_router,
)

BATCH, SEQ = 2, 8


def _config(**kw: object) -> MoEConfig:
    base: dict[str, object] = {
        "d_model": 16,
        "d_ff": 32,
        "num_experts": 4,
        "top_k": 2,
    }
    base.update(kw)
    return MoEConfig(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("router_type", ["topk", "switch", "expert_choice"])
def test_output_shapes(router_type: str) -> None:
    """Every router emits the documented shapes for a ``[B, T, d]`` input.

    Why it matters: the expert bank and losses index these tensors by exact
    shape; a wrong dimension would broadcast silently rather than error.
    """
    top_k = 1 if router_type == "switch" else 2
    config = _config(router_type=router_type, top_k=top_k)
    router = build_router(config)
    out = router(torch.randn(BATCH, SEQ, config.d_model))
    n_tokens = BATCH * SEQ
    assert out.dispatch_weights.shape == (n_tokens, config.num_experts, 1)
    assert out.combine_weights.shape == (n_tokens, config.num_experts, 1)
    assert out.expert_indices.shape == (n_tokens, top_k)
    assert out.router_logits.shape == (n_tokens, config.num_experts)
    assert out.router_probs.shape == (n_tokens, config.num_experts)
    assert out.aux_loss.shape == ()


def test_topk_sparsity_exactly_k() -> None:
    """For top_k=2 exactly two experts are non-zero per token."""
    router = TopKRouter(_config(top_k=2))
    out = router(torch.randn(BATCH, SEQ, 16))
    nonzero_per_token = torch.count_nonzero(out.combine_weights.squeeze(-1), dim=-1)
    assert torch.all(nonzero_per_token == 2)


def test_combine_weights_sum_le_one() -> None:
    """Per-token combine weights sum to <= 1 (masked full-softmax convention)."""
    router = TopKRouter(_config(top_k=2, normalize_router_weights=False))
    out = router(torch.randn(BATCH, SEQ, 16))
    sums = out.combine_weights.squeeze(-1).sum(dim=-1)
    assert torch.all(sums <= 1.0 + 1e-6)


def test_normalised_weights_sum_to_one() -> None:
    """With renormalisation the selected top-K weights sum to 1 per token."""
    router = TopKRouter(_config(top_k=2, normalize_router_weights=True))
    out = router(torch.randn(BATCH, SEQ, 16))
    sums = out.combine_weights.squeeze(-1).sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_train_vs_eval_noise() -> None:
    """Noisy gating varies the routing in train mode but is fixed in eval mode.

    Why it matters: noise must aid exploration during training yet be perfectly
    deterministic at inference, or eval metrics become irreproducible.
    """
    router = TopKRouter(_config(use_noisy_gating=True))
    x = torch.randn(BATCH, SEQ, 16)

    router.train()
    train_runs = torch.stack([router(x).combine_weights for _ in range(50)])
    router.eval()
    eval_runs = torch.stack([router(x).combine_weights for _ in range(50)])

    assert train_runs.std(dim=0).max().item() > 0.0  # varies in train
    assert eval_runs.std(dim=0).max().item() == 0.0  # identical in eval


def test_gradient_flow_through_router() -> None:
    """``aux_loss.backward()`` produces non-zero gradients for W_g and W_noise."""
    router = TopKRouter(_config(use_noisy_gating=True, alpha=1e-2))
    out = router(torch.randn(BATCH, SEQ, 16))
    out.aux_loss.backward()

    assert router.w_gate.weight.grad is not None
    assert torch.any(router.w_gate.weight.grad != 0)
    assert router.w_noise is not None
    assert router.w_noise.weight.grad is not None
    assert torch.any(router.w_noise.weight.grad != 0)


def test_numerical_stability_large_logits() -> None:
    """Inputs scaled 1000x must not produce NaN/Inf anywhere in the output."""
    router = TopKRouter(_config(use_noisy_gating=True))
    x = torch.randn(BATCH, SEQ, 16) * 1000.0
    out = router(x)
    for tensor in (
        out.dispatch_weights,
        out.combine_weights,
        out.router_logits,
        out.router_probs,
        out.aux_loss,
    ):
        assert torch.isfinite(tensor).all()


def test_switch_router_capacity_overflow() -> None:
    """Switch routing drops exactly the tokens that overflow expert capacity.

    We craft a gating head and one-hot inputs so 90 of 100 tokens route to
    expert 0. With capacity_factor=1.0 and 4 experts the capacity is 25, so
    ``90 - 25 = 65`` tokens must be dropped (their combine weights become zero).
    """
    n_tokens, num_experts = 100, 4
    config = MoEConfig(
        d_model=num_experts,
        d_ff=8,
        num_experts=num_experts,
        top_k=1,
        router_type="switch",
        capacity_factor=1.0,
        drop_tokens=True,
    )
    router = SwitchRouter(config).eval()
    with torch.no_grad():
        # Identity-scaled head => argmax(logits) == argmax(input one-hot).
        router.w_gate.weight.copy_(torch.eye(num_experts) * 50.0)

    x = torch.zeros(n_tokens, num_experts)
    x[:90, 0] = 1.0  # 90 tokens prefer expert 0
    for i in range(10):  # remaining 10 spread over experts 1..3
        x[90 + i, (i % (num_experts - 1)) + 1] = 1.0

    out = router(x)
    dropped = int((out.combine_weights.squeeze(-1).sum(dim=-1) == 0).sum())
    capacity = config.capacity(n_tokens)
    assert dropped == max(0, 90 - capacity) == 65


def test_switch_requires_top_1() -> None:
    """Constructing a Switch router with top_k != 1 raises immediately."""
    with pytest.raises(ValueError, match="top_k == 1"):
        SwitchRouter(MoEConfig(num_experts=4, top_k=2, router_type="switch"))


def test_expert_choice_no_token_exceeds_one() -> None:
    """Expert-Choice combine weights still sum to <= 1 per token."""
    router = ExpertChoiceRouter(_config(router_type="expert_choice", top_k=2))
    out = router(torch.randn(BATCH, SEQ, 16))
    sums = out.combine_weights.squeeze(-1).sum(dim=-1)
    assert torch.all(sums <= 1.0 + 1e-6)


def test_flatten_accepts_2d_and_rejects_bad_rank() -> None:
    """The router accepts flat 2-D inputs and rejects 1-D/4-D inputs."""
    router = TopKRouter(_config())
    out = router(torch.randn(5, 16))  # already-flat [tokens, d_model]
    assert out.router_logits.shape == (5, 4)
    with pytest.raises(ValueError, match="2-D .* or 3-D"):
        router(torch.randn(16))


def test_jitter_and_zscore_paths_run() -> None:
    """Input jitter and z-score logit normalisation execute in train and eval.

    These optional preprocessing paths are off by default; this exercises both
    so a regression in either is caught (and confirms eval stays finite).
    """
    config = _config(jitter_noise=0.1, router_z_score_norm=True)
    router = TopKRouter(config)
    router.train()
    train_out = router(torch.randn(BATCH, SEQ, 16))
    router.eval()
    eval_out = router(torch.randn(BATCH, SEQ, 16))
    assert torch.isfinite(train_out.router_logits).all()
    assert torch.isfinite(eval_out.router_logits).all()


def test_router_extra_repr() -> None:
    """The router's ``extra_repr`` surfaces its key hyperparameters."""
    repr_str = TopKRouter(_config(top_k=2, use_noisy_gating=True)).extra_repr()
    assert "num_experts=4" in repr_str
    assert "top_k=2" in repr_str
    assert "noisy=True" in repr_str


def test_build_router_unknown_type() -> None:
    """``build_router`` validates ``router_type`` before construction."""
    config = MoEConfig(num_experts=4, top_k=2)
    object.__setattr__(config, "router_type", "bogus")  # bypass validate
    with pytest.raises(ValueError, match="Unknown router_type"):
        build_router(config)
