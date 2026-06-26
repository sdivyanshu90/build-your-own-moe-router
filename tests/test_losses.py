"""Tests for :mod:`moe.losses`.

The auxiliary and z-losses are what make sparse routing trainable. These tests
pin their exact values at the analytically-known extremes (perfect balance, full
collapse, zero logits), confirm numerical stability, and verify the detach/attach
gradient pattern that lets a non-differentiable counter shape a differentiable
loss.
"""

from __future__ import annotations

import math

import pytest
import torch

from moe.losses import (
    MoELoss,
    auxiliary_load_balancing_loss,
    router_z_loss,
)

ALPHA = 1e-2
BETA = 1e-3


def test_aux_loss_perfect_balance() -> None:
    r"""Perfectly balanced routing gives ``L_aux = alpha``.

    Derivation: with ``N`` experts, ``f_i = P_i = 1/N`` for all ``i``, so
    ``sum_i f_i P_i = N * (1/N)(1/N) = 1/N`` and
    ``L_aux = alpha * N * (1/N) = alpha``.
    """
    num_experts = 4
    probs = torch.full((8, num_experts), 1.0 / num_experts)
    # Round-robin indices => every expert receives an equal share.
    indices = torch.arange(8).remainder(num_experts).unsqueeze(1)
    loss = auxiliary_load_balancing_loss(probs, indices, num_experts, alpha=ALPHA)
    assert loss.item() == pytest.approx(ALPHA, abs=1e-7)


def test_aux_loss_collapsed() -> None:
    r"""Fully collapsed routing gives ``L_aux = alpha * N``.

    Derivation: all mass on expert 0 means ``f_0 = P_0 = 1`` and every other
    term is 0, so ``sum_i f_i P_i = 1`` and ``L_aux = alpha * N * 1 = alpha N``.
    """
    num_experts = 4
    probs = torch.zeros(8, num_experts)
    probs[:, 0] = 1.0
    indices = torch.zeros(8, 1, dtype=torch.long)
    loss = auxiliary_load_balancing_loss(probs, indices, num_experts, alpha=ALPHA)
    assert loss.item() == pytest.approx(ALPHA * num_experts, abs=1e-7)


def test_aux_loss_rejects_bad_shape() -> None:
    """Non-2-D probs or a mismatched expert dim raise informative errors."""
    with pytest.raises(ValueError, match="must be 2-D"):
        auxiliary_load_balancing_loss(torch.rand(4), torch.zeros(4, 1), 4)
    with pytest.raises(ValueError, match="must equal num_experts"):
        auxiliary_load_balancing_loss(torch.rand(4, 3), torch.zeros(4, 1), 4)


def test_z_loss_at_origin() -> None:
    r"""Zero logits give ``L_z = beta * (log N)^2``.

    Derivation: ``logsumexp(0,...,0) = log N`` for ``N`` experts, squared and
    averaged over tokens is ``(log N)^2``.
    """
    num_experts = 4
    logits = torch.zeros(3, num_experts)
    loss = router_z_loss(logits, beta=BETA)
    assert loss.item() == pytest.approx(BETA * math.log(num_experts) ** 2, abs=1e-6)


def test_z_loss_numerical_stability() -> None:
    """Logits of magnitude 1e4 stay finite thanks to ``logsumexp``."""
    logits = torch.full((4, 8), 1e4)
    loss = router_z_loss(logits, beta=BETA)
    assert torch.isfinite(loss)


def test_z_loss_rejects_bad_shape() -> None:
    """A non-2-D logits tensor raises an informative error."""
    with pytest.raises(ValueError, match="must be 2-D"):
        router_z_loss(torch.zeros(4))


def test_z_loss_non_negative() -> None:
    """The z-loss is a mean of squares and is therefore non-negative."""
    loss = router_z_loss(torch.randn(10, 6) * 3.0)
    assert loss.item() >= 0.0


def test_moeloss_total_is_sum_of_parts() -> None:
    """``total == task + alpha*aux + beta*z`` to within tolerance."""
    crit = MoELoss(alpha=ALPHA, beta=BETA, num_experts=4)
    probs = torch.softmax(torch.randn(8, 4), dim=-1)
    logits = torch.randn(8, 4)
    indices = torch.randint(0, 4, (8, 2))
    task = torch.tensor(2.5)

    out = crit(task, probs, indices, logits)
    expected = task + ALPHA * out.aux_loss + BETA * out.z_loss
    assert out.total_loss.item() == pytest.approx(expected.item(), abs=1e-5)


def test_moeloss_gradient_only_through_probs() -> None:
    """Gradient flows through P_i (router_probs) but not the detached f_i counter.

    We make the logits a leaf and derive both ``router_probs`` (the
    differentiable P_i) and ``expert_indices`` (the detached f_i counter) from a
    *separate* leaf. After backward, only the probs' leaf has a gradient; the
    index source has none, proving the counter is detached.
    """
    prob_logits = torch.randn(8, 4, requires_grad=True)
    index_source = torch.randn(8, 4, requires_grad=True)

    probs = torch.softmax(prob_logits, dim=-1)
    # argmax/topk over index_source is non-differentiable AND detached inside the
    # loss; index_source must therefore receive no gradient.
    indices = index_source.topk(2, dim=-1).indices

    loss = auxiliary_load_balancing_loss(probs, indices, num_experts=4, alpha=1.0)
    loss.backward()

    assert prob_logits.grad is not None
    assert torch.any(prob_logits.grad != 0)
    assert index_source.grad is None  # f_i path is fully detached


def test_moeloss_init_validation_and_helpers() -> None:
    """Negative weights / bad expert count raise; helpers behave as documented."""
    with pytest.raises(ValueError, match="non-negative"):
        MoELoss(alpha=-1.0, beta=0.0, num_experts=4)
    with pytest.raises(ValueError, match="num_experts"):
        MoELoss(alpha=0.0, beta=0.0, num_experts=0)

    crit = MoELoss(alpha=ALPHA, beta=BETA, num_experts=4)
    # log_metrics before any forward returns only the step.
    assert crit.log_metrics(step=0) == {"step": 0}

    probs = torch.softmax(torch.randn(8, 4), dim=-1)
    crit(torch.tensor(1.0), probs, torch.randint(0, 4, (8, 2)), torch.randn(8, 4))
    metrics = crit.log_metrics(step=5)
    assert metrics["step"] == 5
    assert "loss/total" in metrics
    assert "loss/aux_fraction" in metrics
    assert "alpha=" in crit.extra_repr()


def test_moeloss_from_config() -> None:
    """``from_config`` wires alpha/beta/num_experts from an MoEConfig."""
    from moe.config import MoEConfig

    crit = MoELoss.from_config(MoEConfig(num_experts=8, alpha=ALPHA, beta=BETA))
    assert crit.num_experts == 8
    assert crit.alpha == ALPHA
    assert crit.beta == BETA
