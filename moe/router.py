"""Gating networks (routers) for the MoE layer.

Three routers are provided, all returning a common :class:`RouterOutput` so they
are drop-in interchangeable inside :class:`~moe.layer.MoELayer`:

* :class:`TopKRouter` — softmax / noisy top-K gating (Shazeer 2017).
* :class:`SwitchRouter` — top-1 routing with a capacity buffer (Fedus 2022).
* :class:`ExpertChoiceRouter` — experts pick their top-C tokens (Zhou 2022).

The default combine-weight convention is the *masked full softmax*: a softmax is
taken over all experts and then masked to the top-K, so per-token weights sum to
``<= 1`` (Switch convention). Set ``config.normalize_router_weights = True`` to
renormalise the selected subset to sum to 1 (Mixtral convention).

Example:
    >>> import torch
    >>> from moe.config import MoEConfig
    >>> router = TopKRouter(MoEConfig(d_model=16, num_experts=4, top_k=2))
    >>> out = router(torch.randn(2, 8, 16))   # [batch, seq, d_model]
    >>> out.combine_weights.shape
    torch.Size([16, 4, 1])
"""

from __future__ import annotations

from typing import NamedTuple, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import MoEConfig
from .losses import auxiliary_load_balancing_loss

#: Momentum for the optional running-std buffer used by z-score normalisation.
_ZSCORE_MOMENTUM: float = 0.99


class RouterOutput(NamedTuple):
    """Everything a router produces, consumed by the expert bank and losses.

    Attributes:
        dispatch_weights: ``[num_tokens, num_experts, 1]`` weights used to send
            tokens to experts (non-zero only for selected experts).
        combine_weights: ``[num_tokens, num_experts, 1]`` weights used to combine
            the experts' outputs. Identical to ``dispatch_weights`` in these
            routers, kept separate to mirror the GShard dispatch/combine split.
        expert_indices: ``[num_tokens, top_k]`` selected expert ids.
        router_logits: ``[num_tokens, num_experts]`` pre-softmax gating logits
            (the values actually used for routing), needed by the z-loss.
        aux_loss: Scalar weighted auxiliary load-balancing loss.
        router_probs: ``[num_tokens, num_experts]`` full softmax distribution
            over all experts, used for monitoring and the aux loss.
    """

    dispatch_weights: Tensor
    combine_weights: Tensor
    expert_indices: Tensor
    router_logits: Tensor
    aux_loss: Tensor
    router_probs: Tensor


class TopKRouter(nn.Module):
    """Softmax / noisy top-K gating network.

    Args:
        config: The :class:`~moe.config.MoEConfig`.

    Notes:
        The gating projection ``W_g`` has **no bias** on purpose. A per-expert
        bias is a token-independent preference that directly biases the partition
        of tokens across experts, working against the load-balancing loss; and it
        is redundant because, per token, softmax is invariant to a constant added
        to all logits. Routing should therefore be a pure function of the input.
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        # Gating head: d_model -> num_experts, no bias (see class docstring).
        self.w_gate = nn.Linear(config.d_model, config.num_experts, bias=False)

        # Optional learnable noise scale for noisy top-K gating (Shazeer 2017).
        self.w_noise: nn.Linear | None
        if config.use_noisy_gating:
            self.w_noise = nn.Linear(config.d_model, config.num_experts, bias=False)
            # Small init so noise starts modest and grows only if useful.
            nn.init.normal_(self.w_noise.weight, std=config.noise_std_init * 1e-2)
        else:
            self.w_noise = None

        # Running std for optional z-score logit normalisation (eval-time stat).
        self.register_buffer("running_logit_std", torch.ones(()), persistent=True)

    # -- Logit computation --------------------------------------------------

    def _flatten(self, x: Tensor) -> Tensor:
        """Reshape ``[batch, seq, d_model]`` (or already-flat) to ``[N, d_model]``."""
        if x.dim() == 3:
            return x.reshape(-1, x.size(-1))
        if x.dim() == 2:
            return x
        raise ValueError(
            f"Router expects a 2-D [tokens, d_model] or 3-D [batch, seq, d_model] "
            f"input, got shape {tuple(x.shape)}."
        )

    def _routing_logits(self, x_flat: Tensor) -> Tensor:
        """Compute the gating logits used for routing.

        Applies, in order: optional training-time input jitter, the gating
        projection, optional z-score normalisation, and optional noisy-gating
        Gaussian noise (training only).

        Args:
            x_flat: Flattened inputs ``[num_tokens, d_model]``.

        Returns:
            Routing logits ``[num_tokens, num_experts]``.

        Notes:
            Logits are *not* force-cast to float32 here; under ``torch.autocast``
            the downstream softmax runs in float32 automatically (it is on the
            autocast fp32 policy), keeping routing probabilities stable while the
            expert matmuls stay in the low-precision compute dtype.
        """
        # Switch-style multiplicative input jitter, training only.
        if self.training and self.config.jitter_noise > 0.0:
            jitter = self.config.jitter_noise
            # Uniform in [1 - jitter, 1 + jitter]; created on x's device/dtype.
            scale = torch.empty_like(x_flat).uniform_(1.0 - jitter, 1.0 + jitter)
            x_flat = x_flat * scale

        # Annotate: ``nn.Module.__call__`` is typed to return ``Any``.
        logits: Tensor = self.w_gate(x_flat)  # [num_tokens, num_experts]

        if self.config.router_z_score_norm:
            logits = self._z_score_normalise(logits)

        # Noisy top-K gating: add input-dependent, strictly-positive-scale noise.
        if self.training and self.w_noise is not None:
            # softplus keeps the noise std strictly positive and smooth.
            noise_std = F.softplus(self.w_noise(x_flat))
            noise = noise_std * torch.randn_like(logits)
            logits = logits + noise

        return logits

    def _z_score_normalise(self, logits: Tensor) -> Tensor:
        """Divide logits by their (running) standard deviation before softmax.

        In training the current batch std is used and the running buffer is
        updated; in eval the stored running std is used. This is an alternative
        to the z-loss for controlling logit scale.
        """
        # ``nn.Module.__getattr__`` types buffer access as ``Tensor | Module``;
        # narrow it so the arithmetic below is typed as ``Tensor``.
        running_std = cast(Tensor, self.running_logit_std)
        if self.training:
            batch_std = logits.detach().float().std().clamp_min(self.config.eps)
            # EMA update of the running statistic (no autograd through the buffer).
            running_std.mul_(_ZSCORE_MOMENTUM).add_(
                batch_std * (1.0 - _ZSCORE_MOMENTUM)
            )
            denom = batch_std
        else:
            denom = running_std.clamp_min(self.config.eps)
        return logits / denom.to(logits.dtype)

    # -- Weight assembly ----------------------------------------------------

    def _combine_weights(
        self, routing_logits: Tensor, router_probs: Tensor, expert_indices: Tensor
    ) -> Tensor:
        """Build the ``[num_tokens, num_experts]`` combine weights.

        Args:
            routing_logits: Routing logits (used if renormalising).
            router_probs: Full softmax distribution over all experts.
            expert_indices: Selected expert ids ``[num_tokens, top_k]``.

        Returns:
            Combine weights, zero for non-selected experts. By default these are
            the masked full-softmax probabilities (sum ``<= 1`` per token); with
            ``normalize_router_weights`` they are renormalised to sum to 1.
        """
        # Boolean mask of the top-K experts per token.
        topk_mask = torch.zeros_like(router_probs, dtype=torch.bool)
        topk_mask.scatter_(-1, expert_indices, True)

        if self.config.normalize_router_weights:
            # KeepTopK then softmax: set non-top-K logits to -inf so the softmax
            # renormalises over exactly the selected experts (sums to 1).
            neg_inf = torch.finfo(routing_logits.dtype).min
            masked = routing_logits.masked_fill(~topk_mask, neg_inf)
            return F.softmax(masked, dim=-1)
        # Masked full softmax (not renormalised); per-token sum <= 1.
        return router_probs * topk_mask.to(router_probs.dtype)

    def forward(self, x: Tensor) -> RouterOutput:
        """Route tokens to their top-K experts.

        Args:
            x: Inputs ``[batch, seq, d_model]`` or flattened ``[num_tokens,
                d_model]``.

        Returns:
            A :class:`RouterOutput`.

        Example:
            >>> import torch
            >>> from moe.config import MoEConfig
            >>> r = TopKRouter(MoEConfig(d_model=8, num_experts=4, top_k=2))
            >>> out = r(torch.randn(3, 8))
            >>> int((out.combine_weights.squeeze(-1) != 0).sum(-1)[0])
            2
        """
        x_flat = self._flatten(x)
        routing_logits = self._routing_logits(x_flat)

        # Full softmax distribution (monitoring + aux loss). Sums to 1 per token.
        router_probs = F.softmax(routing_logits, dim=-1)

        # Top-K selection on the routing logits.
        k = min(self.top_k, self.num_experts)
        expert_indices = routing_logits.topk(k, dim=-1).indices  # [N, k]

        combine = self._combine_weights(routing_logits, router_probs, expert_indices)
        weights = combine.unsqueeze(-1)  # [N, E, 1]

        aux_loss = auxiliary_load_balancing_loss(
            router_probs, expert_indices, self.num_experts, alpha=self.config.alpha
        )

        return RouterOutput(
            dispatch_weights=weights,
            combine_weights=weights,
            expert_indices=expert_indices,
            router_logits=routing_logits,
            aux_loss=aux_loss,
            router_probs=router_probs,
        )

    def extra_repr(self) -> str:
        """One-line summary for ``print(module)``."""
        return (
            f"d_model={self.w_gate.in_features}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, noisy={self.w_noise is not None}"
        )


class SwitchRouter(TopKRouter):
    """Top-1 Switch routing with an explicit capacity buffer (Fedus 2022).

    Switch routing always selects a single expert per token, which makes the
    dispatch trivially simple and stable, at the cost of needing a capacity
    buffer to bound per-expert load. Tokens that overflow an expert's capacity
    are dropped: their combine weight is set to zero, so they contribute nothing
    to the layer output (they rely on the residual connection).

    Args:
        config: The :class:`~moe.config.MoEConfig`. ``top_k`` must be 1.
    """

    def __init__(self, config: MoEConfig) -> None:
        if config.top_k != 1:
            raise ValueError(
                f"SwitchRouter requires top_k == 1, got {config.top_k}. Use "
                "TopKRouter for top_k > 1."
            )
        super().__init__(config)

    def forward(self, x: Tensor) -> RouterOutput:
        """Route each token to its single best expert, enforcing capacity.

        Args:
            x: Inputs ``[batch, seq, d_model]`` or ``[num_tokens, d_model]``.

        Returns:
            A :class:`RouterOutput`. Dropped (overflowing) tokens have all-zero
            combine weights; the number dropped is observable as
            ``(combine_weights.squeeze(-1).sum(-1) == 0).sum()``.
        """
        x_flat = self._flatten(x)
        routing_logits = self._routing_logits(x_flat)
        router_probs = F.softmax(routing_logits, dim=-1)

        num_tokens = x_flat.size(0)
        # Top-1 selection via argmax (cheaper than topk for k=1).
        expert_idx = routing_logits.argmax(dim=-1)  # [num_tokens]
        # Combine weight = probability of the chosen expert (not renormalised).
        gate = router_probs.gather(-1, expert_idx.unsqueeze(-1)).squeeze(-1)  # [N]

        # Capacity buffer: rank each token within its expert (FIFO by position).
        one_hot = F.one_hot(expert_idx, self.num_experts).to(torch.long)  # [N, E]
        position = one_hot.cumsum(dim=0) - 1  # [N, E]
        token_pos = position.gather(-1, expert_idx.unsqueeze(-1)).squeeze(-1)  # [N]
        capacity = self.config.capacity(num_tokens)
        keep = token_pos < capacity  # [num_tokens]
        gate = gate * keep.to(gate.dtype)  # zero out dropped tokens

        # Scatter the per-token gate into a dense [N, E] combine matrix.
        combine = torch.zeros_like(router_probs)
        combine.scatter_(-1, expert_idx.unsqueeze(-1), gate.unsqueeze(-1))
        weights = combine.unsqueeze(-1)
        expert_indices = expert_idx.unsqueeze(-1)  # [N, 1]

        aux_loss = auxiliary_load_balancing_loss(
            router_probs, expert_indices, self.num_experts, alpha=self.config.alpha
        )
        return RouterOutput(
            dispatch_weights=weights,
            combine_weights=weights,
            expert_indices=expert_indices,
            router_logits=routing_logits,
            aux_loss=aux_loss,
            router_probs=router_probs,
        )


class ExpertChoiceRouter(TopKRouter):
    """Expert Choice routing: each expert selects its top-C tokens (Zhou 2022).

    Standard routing asks "which experts does this token want?"; Expert Choice
    inverts the question to "which tokens does this expert want?". Each expert
    independently keeps its top-``C`` tokens by affinity, which guarantees every
    expert is exactly full and eliminates dropped-token starvation — at the cost
    that some tokens may be chosen by many experts and some by none. Gradients
    flow only through the (token, expert) pairs that were actually selected.

    Args:
        config: The :class:`~moe.config.MoEConfig`.
    """

    def forward(self, x: Tensor) -> RouterOutput:
        """Route from the experts' perspective.

        Args:
            x: Inputs ``[batch, seq, d_model]`` or ``[num_tokens, d_model]``.

        Returns:
            A :class:`RouterOutput`. ``combine_weights`` is non-zero only for the
            (token, expert) pairs that experts selected.

        Notes:
            ``expert_indices`` is filled with each token's top-``top_k`` experts
            by affinity purely to satisfy the common interface (and to give the
            aux loss a per-token counter); the *actual* dispatch is governed by
            ``combine_weights``, computed from the expert-side top-C selection.
        """
        x_flat = self._flatten(x)
        routing_logits = self._routing_logits(x_flat)
        router_probs = F.softmax(routing_logits, dim=-1)  # [N, E] affinity matrix

        num_tokens = x_flat.size(0)
        # Capacity C: number of tokens each expert keeps (>= 1, <= num_tokens).
        capacity = min(self.config.capacity(num_tokens), num_tokens)

        # Each expert selects its top-C tokens along the token axis.
        topk_scores, topk_tokens = router_probs.topk(capacity, dim=0)  # [C, E]

        # Scatter selected affinities into a dense [N, E] combine matrix.
        combine = torch.zeros_like(router_probs)
        expert_axis = torch.arange(self.num_experts, device=x_flat.device)
        combine[topk_tokens, expert_axis.unsqueeze(0).expand_as(topk_tokens)] = (
            topk_scores
        )
        weights = combine.unsqueeze(-1)

        # Interface-only per-token indices (see Notes).
        k = min(self.top_k, self.num_experts)
        expert_indices = router_probs.topk(k, dim=-1).indices  # [N, k]

        aux_loss = auxiliary_load_balancing_loss(
            router_probs, expert_indices, self.num_experts, alpha=self.config.alpha
        )
        return RouterOutput(
            dispatch_weights=weights,
            combine_weights=weights,
            expert_indices=expert_indices,
            router_logits=routing_logits,
            aux_loss=aux_loss,
            router_probs=router_probs,
        )


def build_router(config: MoEConfig) -> TopKRouter:
    """Factory that instantiates the router named by ``config.router_type``.

    Args:
        config: The :class:`~moe.config.MoEConfig`.

    Returns:
        A router instance (:class:`TopKRouter`, :class:`SwitchRouter` or
        :class:`ExpertChoiceRouter`).

    Raises:
        ValueError: If ``config.router_type`` is unrecognised.
    """
    if config.router_type == "topk":
        return TopKRouter(config)
    if config.router_type == "switch":
        return SwitchRouter(config)
    if config.router_type == "expert_choice":
        return ExpertChoiceRouter(config)
    raise ValueError(
        f"Unknown router_type {config.router_type!r}; expected one of "
        "'topk', 'switch', 'expert_choice'."
    )
