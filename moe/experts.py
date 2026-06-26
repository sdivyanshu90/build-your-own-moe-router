"""Expert feed-forward networks and the dispatching expert bank.

This module provides :class:`Expert` (a single two-layer MLP expert) and
:class:`ExpertBank` (a collection of experts plus the dispatch/combine logic that
sends each token to its selected experts and gathers the weighted result).

Two dispatch strategies are implemented and are guaranteed to produce identical
outputs (up to floating-point reordering) because they share a single functional
FFN kernel, :func:`expert_ffn`:

* ``"naive"`` — a readable per-expert Python loop.
* ``"batch"`` — a fixed-capacity ``[num_experts, capacity, d_model]`` batched
  dispatch using a batched ``matmul`` over stacked expert weights.

Example:
    >>> import torch
    >>> from moe.config import MoEConfig
    >>> cfg = MoEConfig(d_model=8, d_ff=16, num_experts=2, top_k=1)
    >>> bank = ExpertBank(cfg)
    >>> x = torch.randn(4, 8)
    >>> dw = torch.zeros(4, 2, 1); dw[:, 0, 0] = 1.0   # everything to expert 0
    >>> idx = torch.zeros(4, 1, dtype=torch.long)
    >>> bank(x, dw, idx).shape
    torch.Size([4, 8])
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import MoEConfig

#: ``relu2`` squares the ReLU output; the exponent is named to avoid a bare
#: literal in the kernel.
_SQUARE: int = 2

#: Return type of :meth:`ExpertBank._stacked_weights`: the stacked first-layer
#: weight/bias, second-layer weight/bias, and optional SwiGLU gate weight/bias.
_StackedWeights = tuple[
    Tensor, Tensor | None, Tensor, Tensor | None, Tensor | None, Tensor | None
]


def _batched_linear(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    """Apply a per-expert linear layer to a batched ``[E, C, in]`` input.

    Equivalent to ``einsum("eci,eoi->eco", x, weight)`` plus bias, but built from
    :func:`torch.matmul` so it is *autocast-eligible*: under ``torch.autocast`` it
    runs in the low-precision compute dtype exactly like :func:`F.linear` on the
    single-expert path. That parity is what keeps the naive and batched dispatch
    strategies' output dtype identical under mixed precision.

    Args:
        x: Inputs ``[num_experts, capacity, in_features]``.
        weight: Per-expert weight ``[num_experts, out_features, in_features]``.
        bias: Per-expert bias ``[num_experts, out_features]`` or ``None``.

    Returns:
        Output ``[num_experts, capacity, out_features]``.

    Notes:
        The bias is cast to the matmul's output dtype before the add; otherwise a
        float32 bias parameter would silently re-promote a bfloat16 result back to
        float32 under autocast.
    """
    # matmul contracts the last dim of x with the last dim of weight^T.
    out = torch.matmul(x, weight.transpose(-1, -2))
    if bias is not None:
        out = out + bias.unsqueeze(1).to(out.dtype)
    return out


def expert_ffn(
    x: Tensor,
    w1: Tensor,
    b1: Tensor | None,
    w2: Tensor,
    b2: Tensor | None,
    w3: Tensor | None,
    b3: Tensor | None,
    activation: str,
    dropout_p: float,
    training: bool,
) -> Tensor:
    """Apply a two-layer expert FFN, supporting both single and batched weights.

    This single kernel backs both :class:`Expert` (2-D weights, one expert) and
    the batched dispatch path (3-D weights, one set per expert). Sharing the
    kernel is what guarantees the two :class:`ExpertBank` strategies agree.

    Args:
        x: Inputs. Either ``[num_tokens, d_model]`` (unbatched) or
            ``[num_experts, capacity, d_model]`` (batched).
        w1: First-layer weight. ``[d_ff, d_model]`` (unbatched) or
            ``[num_experts, d_ff, d_model]`` (batched).
        b1: First-layer bias or ``None``.
        w2: Second-layer weight. ``[d_model, d_ff]`` or
            ``[num_experts, d_model, d_ff]``.
        b2: Second-layer bias or ``None``.
        w3: SwiGLU gate weight (same shape as ``w1``) or ``None`` for non-SwiGLU.
        b3: SwiGLU gate bias or ``None``.
        activation: ``"gelu"``, ``"swiglu"`` or ``"relu2"``.
        dropout_p: Hidden-layer dropout probability.
        training: Whether to apply dropout (it is a no-op when ``False``).

    Returns:
        The FFN output, same leading shape as ``x`` and trailing dim ``d_model``.

    Raises:
        ValueError: If ``activation`` is unknown or SwiGLU is requested without
            a gate weight.

    Notes:
        For batched weights we use :func:`_batched_linear` (a ``matmul``) rather
        than :func:`F.linear`, because ``F.linear`` cannot broadcast a per-expert
        weight stack. ``matmul`` is autocast-eligible, so the batched path runs in
        the same low-precision dtype as the single-expert ``F.linear`` path under
        ``torch.autocast`` — keeping the two dispatch strategies consistent.
    """
    batched = w1.dim() == _SQUARE + 1  # 3-D weights => batched per-expert path

    # ---- First (up) projection -------------------------------------------
    if batched:
        hidden = _batched_linear(x, w1, b1)
    else:
        hidden = F.linear(x, w1, b1)

    # ---- Non-linearity ----------------------------------------------------
    if activation == "swiglu":
        if w3 is None:
            raise ValueError("activation='swiglu' requires a gate weight w3.")
        if batched:
            gate = _batched_linear(x, w3, b3)
        else:
            gate = F.linear(x, w3, b3)
        # SwiGLU: SiLU(gate) elementwise-gates the up-projection.
        hidden = F.silu(gate) * hidden
    elif activation == "gelu":
        hidden = F.gelu(hidden)
    elif activation == "relu2":
        # ReLU squared (a.k.a. squared ReLU): relu(x) ** 2.
        relu = F.relu(hidden)
        hidden = relu.pow(_SQUARE)
    else:
        raise ValueError(
            f"Unknown activation {activation!r}; expected gelu, swiglu or relu2."
        )

    # ---- Dropout (training only) -----------------------------------------
    if dropout_p > 0.0 and training:
        hidden = F.dropout(hidden, p=dropout_p, training=True)

    # ---- Second (down) projection ----------------------------------------
    if batched:
        out = _batched_linear(hidden, w2, b2)
    else:
        out = F.linear(hidden, w2, b2)
    return out


class Expert(nn.Module):
    """A single feed-forward expert: ``Linear -> activation -> Linear``.

    Args:
        config: The :class:`~moe.config.MoEConfig`. Uses ``d_model``, ``d_ff``,
            ``activation``, ``expert_dropout`` and ``use_bias``.

    Notes:
        Weight initialisation follows the prompt: ``kaiming_uniform_`` in
        ``fan_in`` mode for the up-projection (and SwiGLU gate), zeros for the
        down-projection bias. ``fan_in`` mode scales the variance by the number
        of *input* units, which keeps the forward-pass activation variance
        roughly constant as ``d_model`` grows — the appropriate choice for a
        layer whose output feeds a residual stream.
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.activation = config.activation
        self.dropout_p = config.expert_dropout

        bias = config.use_bias
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=bias)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=bias)
        # SwiGLU needs an extra gate projection of the same shape as ``w1``.
        self.w3: nn.Linear | None
        if config.activation == "swiglu":
            self.w3 = nn.Linear(config.d_model, config.d_ff, bias=bias)
        else:
            self.w3 = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """(Re)initialise expert weights with the documented scheme."""
        nn.init.kaiming_uniform_(self.w1.weight, mode="fan_in", nonlinearity="relu")
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w3 is not None:
            nn.init.kaiming_uniform_(self.w3.weight, mode="fan_in", nonlinearity="relu")
            if self.w3.bias is not None:
                nn.init.zeros_(self.w3.bias)
        # Down-projection: kaiming weight, zeroed bias so the expert starts as a
        # near-identity perturbation on the residual stream.
        nn.init.kaiming_uniform_(self.w2.weight, mode="fan_in", nonlinearity="relu")
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the expert to a batch of token vectors.

        Args:
            x: Token vectors, shape ``[num_tokens, d_model]``.

        Returns:
            Output of shape ``[num_tokens, d_model]``, same dtype/device as ``x``.

        Example:
            >>> import torch
            >>> from moe.config import MoEConfig
            >>> e = Expert(MoEConfig(d_model=4, d_ff=8))
            >>> e(torch.randn(3, 4)).shape
            torch.Size([3, 4])
        """
        return expert_ffn(
            x,
            self.w1.weight,
            self.w1.bias,
            self.w2.weight,
            self.w2.bias,
            None if self.w3 is None else self.w3.weight,
            None if self.w3 is None else self.w3.bias,
            self.activation,
            self.dropout_p,
            self.training,
        )

    def extra_repr(self) -> str:
        """One-line summary of the expert's dimensions and activation."""
        return (
            f"d_model={self.w1.in_features}, d_ff={self.w1.out_features}, "
            f"activation={self.activation}, dropout={self.dropout_p}"
        )


class ExpertBank(nn.Module):
    """A bank of ``num_experts`` experts plus dispatch/combine logic.

    Args:
        config: The :class:`~moe.config.MoEConfig`.

    Attributes:
        experts: ``nn.ModuleList`` of :class:`Expert`.
        last_overflow_tokens: Number of (token, expert) assignments dropped due to
            capacity overflow on the most recent forward call. Instance-level
            state used only for monitoring (never participates in autograd).
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList(Expert(config) for _ in range(config.num_experts))
        self.last_overflow_tokens: int = 0
        self.last_dispatched_tokens: int = 0

    # -- Capacity / keep-mask ----------------------------------------------

    def _keep_mask(self, selected: Tensor, capacity: int) -> tuple[Tensor, Tensor]:
        """Compute which (token, expert) assignments fit within capacity.

        Args:
            selected: Boolean ``[num_tokens, num_experts]`` mask of routed slots.
            capacity: Per-expert buffer size ``C``.

        Returns:
            A pair ``(keep, position)`` where ``keep`` is a boolean mask of the
            same shape (assignments that fit), and ``position`` is the integer
            slot index of each assignment within its expert (valid where
            ``keep`` is ``True``).

        Notes:
            Slot order is the token order (a deterministic FIFO drop): the
            cumulative count of selected tokens *up to and including* row ``t``
            gives token ``t``'s rank within the expert. Ranks ``>= C`` overflow.
        """
        # cumsum along the token axis gives a 1-based running count per expert;
        # subtract 1 for a 0-based slot index.
        position = selected.long().cumsum(dim=0) - 1  # [num_tokens, num_experts]
        keep = selected & (position < capacity)
        return keep, position

    # -- Dispatch strategies -----------------------------------------------

    def _forward_naive(self, x: Tensor, combine: Tensor, keep: Tensor) -> Tensor:
        """Per-expert loop dispatch (readable reference implementation).

        Accumulation happens in float32 for numerical stability (important under
        bfloat16 autocast) and the result is cast back to the expert compute
        dtype so the output flows in the surrounding precision (e.g. bf16).
        """
        output = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        out_dtype = x.dtype
        for e, expert in enumerate(self.experts):
            token_mask = keep[:, e]  # [num_tokens]
            if not bool(token_mask.any()):
                continue  # skip idle experts (also gives expert independence)
            selected_tokens = x[token_mask]  # [n_e, d_model]
            expert_out = expert(selected_tokens)  # [n_e, d_model]
            out_dtype = expert_out.dtype
            weight = combine[token_mask, e].unsqueeze(-1).float()  # [n_e, 1]
            # Accumulate (a token may be selected by several experts for top_k>1).
            output[token_mask] = output[token_mask] + weight * expert_out.float()
        return output.to(out_dtype)

    def _stacked_weights(self) -> _StackedWeights:
        """Stack per-expert linear weights into batched tensors for ``matmul``."""
        w1 = torch.stack([e.w1.weight for e in self.experts])  # [E, d_ff, d_model]
        w2 = torch.stack([e.w2.weight for e in self.experts])  # [E, d_model, d_ff]
        has_bias = self.experts[0].w1.bias is not None
        b1 = torch.stack([e.w1.bias for e in self.experts]) if has_bias else None
        b2 = torch.stack([e.w2.bias for e in self.experts]) if has_bias else None
        if self.experts[0].w3 is not None:
            w3 = torch.stack([e.w3.weight for e in self.experts])  # type: ignore[union-attr]
            b3 = (
                torch.stack([e.w3.bias for e in self.experts])  # type: ignore[union-attr]
                if has_bias
                else None
            )
        else:
            w3, b3 = None, None
        return w1, b1, w2, b2, w3, b3

    def _forward_batch(
        self, x: Tensor, combine: Tensor, keep: Tensor, position: Tensor, capacity: int
    ) -> Tensor:
        """Fixed-capacity batched dispatch via stacked-weight ``matmul``.

        Memory note: the dispatch buffer is ``[num_experts, capacity, d_model]``,
        which is larger than the ``O(num_tokens * d_model)`` input by a factor of
        ``num_experts * capacity / num_tokens``. With ``drop_tokens=False`` and
        ``capacity == num_tokens`` this is ``num_experts``x the input; for large
        ``num_experts`` prefer ``drop_tokens=True`` (a tight capacity) or the
        naive strategy.
        """
        num_tokens, d_model = x.shape
        device, dtype = x.device, x.dtype

        # Flatten the kept (token, expert) assignments.
        tok_idx, exp_idx = keep.nonzero(as_tuple=True)  # each [num_kept]
        slot = position[tok_idx, exp_idx]  # [num_kept]

        # Scatter tokens into the [E, C, d_model] buffer.
        buffer = torch.zeros(
            self.num_experts, capacity, d_model, device=device, dtype=dtype
        )
        buffer[exp_idx, slot] = x[tok_idx]

        # One batched FFN over all experts at once.
        w1, b1, w2, b2, w3, b3 = self._stacked_weights()
        out_buffer = expert_ffn(
            buffer,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            self.config.activation,
            self.config.expert_dropout,
            self.training,
        )  # [E, C, d_model]

        # Gather processed tokens back, weighted by the combine weights.
        # Accumulate in float32 (stable) then cast to the expert compute dtype.
        output = torch.zeros(num_tokens, d_model, device=device, dtype=torch.float32)
        contrib = (
            out_buffer[exp_idx, slot].float()
            * combine[tok_idx, exp_idx].unsqueeze(-1).float()
        )
        output.index_add_(0, tok_idx, contrib)
        return output.to(out_buffer.dtype)

    # -- Public forward -----------------------------------------------------

    def forward(
        self, x: Tensor, dispatch_weights: Tensor, expert_indices: Tensor
    ) -> Tensor:
        """Dispatch tokens to experts and combine the weighted outputs.

        Args:
            x: Flattened token vectors, shape ``[num_tokens, d_model]``.
            dispatch_weights: Combine/dispatch weights, shape
                ``[num_tokens, num_experts, 1]``. Non-zero entries select an
                expert and carry its gating weight.
            expert_indices: Selected expert ids, shape ``[num_tokens, top_k]``.
                Accepted for interface symmetry; routing is driven by the
                non-zero structure of ``dispatch_weights``.

        Returns:
            Combined output ``[num_tokens, d_model]``.

        Raises:
            ValueError: If ``x`` is not 2-D or shapes are inconsistent.

        Notes:
            Capacity is enforced uniformly here so that the naive and batch
            strategies always drop the same assignments and therefore agree.
            ``self.last_overflow_tokens`` records the number dropped.
        """
        if x.dim() != 2:
            raise ValueError(
                f"ExpertBank expects x of shape [num_tokens, d_model], got "
                f"{tuple(x.shape)}."
            )
        del expert_indices  # routing is taken from dispatch_weights' sparsity
        combine = dispatch_weights.squeeze(-1)  # [num_tokens, num_experts]
        selected = combine != 0  # boolean routed mask

        num_tokens = x.size(0)
        capacity = self.config.capacity(num_tokens)
        keep, position = self._keep_mask(selected, capacity)

        # Record overflow statistics for monitoring.
        self.last_dispatched_tokens = int(selected.sum().item())
        self.last_overflow_tokens = int((selected & ~keep).sum().item())

        if self.config.dispatch_strategy == "batch":
            return self._forward_batch(x, combine, keep, position, capacity)
        return self._forward_naive(x, combine, keep)

    def extra_repr(self) -> str:
        """One-line summary of the bank for ``print(module)``."""
        return (
            f"num_experts={self.num_experts}, "
            f"dispatch={self.config.dispatch_strategy}, "
            f"drop_tokens={self.config.drop_tokens}"
        )
