"""Composable load-balancing and regularisation losses for MoE routing.

This module implements the two auxiliary objectives that keep a sparse MoE layer
trainable:

* :func:`auxiliary_load_balancing_loss` — the Switch Transformer load-balancing
  loss that fights expert collapse.
* :func:`router_z_loss` — the ST-MoE z-loss that keeps router logits small and
  numerically well behaved.

They are exposed both as pure functions (easy to unit-test in isolation) and as a
single :class:`MoELoss` ``nn.Module`` that combines them with a task loss and
emits a logging dictionary.

Example:
    >>> import torch
    >>> probs = torch.full((8, 4), 0.25)          # 8 tokens, 4 experts, balanced
    >>> idx = torch.arange(8).remainder(4).unsqueeze(1)  # round-robin top-1
    >>> float(auxiliary_load_balancing_loss(probs, idx, num_experts=4))
    1.0
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

#: Default weights used when a caller invokes the pure loss functions without an
#: explicit weight. A value of 1.0 returns the *raw* (unweighted) loss term so
#: that :class:`MoELoss` can apply the configured ``alpha``/``beta`` itself and
#: still report the raw terms for monitoring.
DEFAULT_WEIGHT: float = 1.0


def auxiliary_load_balancing_loss(
    router_probs: Tensor,
    expert_indices: Tensor,
    num_experts: int,
    alpha: float = DEFAULT_WEIGHT,
) -> Tensor:
    r"""Switch Transformer auxiliary load-balancing loss.

    Implements

    .. math::
        L_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i

    where :math:`N` is the number of experts, :math:`f_i` is the fraction of
    routed slots that landed on expert :math:`i` (a *detached*, non-differentiable
    counter), and :math:`P_i` is the mean softmax routing probability assigned to
    expert :math:`i` over the batch (differentiable).

    Args:
        router_probs: Full softmax routing distribution, shape
            ``[num_tokens, num_experts]``. Each row sums to 1. This term carries
            the gradient (it is the differentiable :math:`P_i`).
        expert_indices: Selected expert ids per token, shape
            ``[num_tokens, top_k]`` (or ``[num_tokens]``). Used only to count
            dispatch fractions :math:`f_i`; it is detached and never receives a
            gradient.
        num_experts: The number of experts :math:`N`.
        alpha: Scalar weight :math:`\alpha`. Defaults to 1.0, i.e. the raw term.

    Returns:
        A scalar tensor. At perfect balance (:math:`f_i = P_i = 1/N`) it equals
        ``alpha``; under full collapse (all mass on one expert) it approaches
        ``alpha * num_experts``.

    Raises:
        ValueError: If ``router_probs`` is not 2-D or its expert dimension does
            not equal ``num_experts``.

    Notes:
        **Why detach/attach works.** The product :math:`f_i \cdot P_i` multiplies a
        constant (detached count) by a differentiable probability. Back-prop
        therefore flows only through :math:`P_i`, and the per-expert gradient is
        proportional to :math:`f_i`. Over-used experts (large :math:`f_i`) get a
        larger downward push on their probability mass, which redistributes
        tokens without ever differentiating through the discrete ``argmax`` that
        produced ``expert_indices``.

    Example:
        >>> import torch
        >>> probs = torch.full((4, 4), 0.25)
        >>> idx = torch.tensor([[0], [1], [2], [3]])
        >>> float(auxiliary_load_balancing_loss(probs, idx, 4))
        1.0
    """
    if router_probs.dim() != 2:
        raise ValueError(
            f"router_probs must be 2-D [num_tokens, num_experts], got shape "
            f"{tuple(router_probs.shape)}."
        )
    if router_probs.size(-1) != num_experts:
        raise ValueError(
            f"router_probs last dim ({router_probs.size(-1)}) must equal "
            f"num_experts ({num_experts})."
        )

    # P_i: mean gating probability per expert, differentiable. Cast to float32
    # for a stable mean under bfloat16/float16 autocast.
    mean_prob = router_probs.float().mean(dim=0)  # [num_experts]

    # f_i: fraction of dispatch slots per expert. ``bincount`` counts the integer
    # expert ids; we DETACH (it is a non-differentiable counter) and normalise by
    # the total number of slots so that ``sum_i f_i == 1``.
    flat_idx = expert_indices.reshape(-1).detach().to(torch.long)
    counts = torch.bincount(flat_idx, minlength=num_experts).to(mean_prob.dtype)
    dispatch_fraction = counts / counts.sum().clamp_min(1.0)  # [num_experts]

    # L_aux = alpha * N * <f, P>. The dot product is the load-imbalance signal.
    loss = alpha * num_experts * torch.sum(dispatch_fraction * mean_prob)
    return loss


def router_z_loss(router_logits: Tensor, beta: float = DEFAULT_WEIGHT) -> Tensor:
    r"""ST-MoE router z-loss (Zoph et al. 2022).

    Implements

    .. math::
        L_z = \beta \cdot \frac{1}{B} \sum_{b=1}^{B}
              \Bigl(\log \sum_{i=1}^{N} e^{z_{b,i}}\Bigr)^2

    where :math:`z_{b,i}` is the pre-softmax gating logit of token :math:`b` for
    expert :math:`i`. The loss penalises large log-partition values, which keeps
    logits small, improves softmax numerical stability and acts as an
    architecture-independent regulariser on the router.

    Args:
        router_logits: Pre-softmax gating logits, shape
            ``[num_tokens, num_experts]``.
        beta: Scalar weight :math:`\beta`. Defaults to 1.0 (raw term).

    Returns:
        A non-negative scalar tensor (a mean of squares, hence ``>= 0``).

    Raises:
        ValueError: If ``router_logits`` is not 2-D.

    Notes:
        We use :func:`torch.logsumexp`, which internally subtracts the row max
        before exponentiating: :math:`\log\sum_i e^{z_i} = m + \log\sum_i
        e^{z_i - m}` with :math:`m = \max_i z_i`. Computing ``log(sum(exp(z)))``
        naively would overflow ``exp`` for logits above ~88 (float32); the
        log-sum-exp identity makes the computation exact and overflow-free.

    Example:
        >>> import torch, math
        >>> logits = torch.zeros(3, 4)
        >>> abs(float(router_z_loss(logits)) - math.log(4) ** 2) < 1e-6
        True
    """
    if router_logits.dim() != 2:
        raise ValueError(
            f"router_logits must be 2-D [num_tokens, num_experts], got shape "
            f"{tuple(router_logits.shape)}."
        )
    # logsumexp over experts → per-token log-partition, in float32 for stability.
    log_z = torch.logsumexp(router_logits.float(), dim=-1)  # [num_tokens]
    return beta * torch.mean(log_z**2)


class LossOutput(NamedTuple):
    """Bundle of scalar losses returned by :class:`MoELoss`.

    Attributes:
        total_loss: ``task_loss + alpha * aux_loss + beta * z_loss``.
        aux_loss: The raw (unweighted) load-balancing term.
        z_loss: The raw (unweighted) z-loss term.
        task_loss: The user-supplied primary objective.
    """

    total_loss: Tensor
    aux_loss: Tensor
    z_loss: Tensor
    task_loss: Tensor


class MoELoss(nn.Module):
    """Combine the task loss with the MoE auxiliary and z-losses.

    The module stores ``alpha`` and ``beta`` and, on every forward call, computes
    the *raw* auxiliary and z-loss terms, scales them and adds them to the task
    loss. The raw terms are returned (and cached for :meth:`log_metrics`) so that
    monitoring can track them independently of their weights.

    Args:
        alpha: Weight of the auxiliary load-balancing loss.
        beta: Weight of the router z-loss.
        num_experts: Number of experts, needed to compute :math:`f_i`/:math:`P_i`.

    Raises:
        ValueError: If ``alpha`` or ``beta`` is negative, or ``num_experts < 1``.
    """

    def __init__(self, alpha: float, beta: float, num_experts: int) -> None:
        super().__init__()
        if alpha < 0.0 or beta < 0.0:
            raise ValueError(
                f"alpha and beta must be non-negative, got alpha={alpha}, beta={beta}."
            )
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}.")
        self.alpha = alpha
        self.beta = beta
        self.num_experts = num_experts
        # Cached scalar values (plain Python floats) for logging. Instance-level,
        # not class-level, so no cross-instance sharing and no lock required.
        self._last: dict[str, float] = {}

    @classmethod
    def from_config(cls, config: Any) -> MoELoss:
        """Build an :class:`MoELoss` from an :class:`~moe.config.MoEConfig`."""
        return cls(alpha=config.alpha, beta=config.beta, num_experts=config.num_experts)

    def forward(
        self,
        task_loss: Tensor,
        router_probs: Tensor,
        expert_indices: Tensor,
        router_logits: Tensor,
    ) -> LossOutput:
        """Combine the task loss with the MoE regularisers.

        Args:
            task_loss: Scalar primary objective (e.g. cross-entropy / MSE).
            router_probs: Full softmax routing distribution
                ``[num_tokens, num_experts]``.
            expert_indices: Selected expert ids ``[num_tokens, top_k]``.
            router_logits: Pre-softmax gating logits ``[num_tokens, num_experts]``.

        Returns:
            A :class:`LossOutput` with the total and the three component losses.
            The component ``aux_loss``/``z_loss`` are the *raw* (unweighted) terms.

        Example:
            >>> import torch
            >>> crit = MoELoss(alpha=0.01, beta=0.001, num_experts=4)
            >>> probs = torch.full((4, 4), 0.25)
            >>> logits = torch.zeros(4, 4)
            >>> idx = torch.tensor([[0], [1], [2], [3]])
            >>> out = crit(torch.tensor(2.0), probs, idx, logits)
            >>> out.total_loss.item() > 0
            True
        """
        raw_aux = auxiliary_load_balancing_loss(
            router_probs, expert_indices, self.num_experts, alpha=DEFAULT_WEIGHT
        )
        raw_z = router_z_loss(router_logits, beta=DEFAULT_WEIGHT)
        total = task_loss + self.alpha * raw_aux + self.beta * raw_z

        # Cache detached floats for logging; never participates in autograd.
        self._last = {
            "loss/total": float(total.detach()),
            "loss/task": float(task_loss.detach()),
            "loss/aux_raw": float(raw_aux.detach()),
            "loss/z_raw": float(raw_z.detach()),
            "loss/aux_weighted": float((self.alpha * raw_aux).detach()),
            "loss/z_weighted": float((self.beta * raw_z).detach()),
        }
        return LossOutput(total, raw_aux, raw_z, task_loss)

    def log_metrics(self, step: int) -> dict[str, float | int]:
        """Return a flat dict of the most recent losses for W&B / TensorBoard.

        Args:
            step: The global training step, echoed back under the ``"step"`` key.

        Returns:
            A dict of scalar metrics from the last :meth:`forward` call. If
            ``forward`` has not been called yet, only ``{"step": step}`` is
            returned.
        """
        metrics: dict[str, float | int] = {"step": step}
        metrics.update(self._last)
        if "loss/total" in self._last and self._last["loss/total"] != 0.0:
            # Fraction of the total loss coming from the aux term; > ~5% means
            # alpha is probably too large.
            metrics["loss/aux_fraction"] = (
                self._last["loss/aux_weighted"] / self._last["loss/total"]
            )
        return metrics

    def extra_repr(self) -> str:
        """One-line summary of the loss weights for ``print(module)``."""
        return f"alpha={self.alpha}, beta={self.beta}, num_experts={self.num_experts}"
