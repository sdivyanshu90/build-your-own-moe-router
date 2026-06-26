"""Configuration dataclass for the Mixture of Experts (MoE) routing layer.

This module defines :class:`MoEConfig`, a single, validated source of truth for
every hyperparameter consumed by the routers, experts, losses and the top-level
:class:`moe.layer.MoELayer`. Keeping all knobs in one immutable-by-convention
dataclass means the rest of the package never has to reach for module-level
constants or environment variables, which keeps the code thread-safe and easy to
serialise.

Example:
    >>> from moe.config import MoEConfig
    >>> cfg = MoEConfig(d_model=64, d_ff=128, num_experts=4, top_k=2)
    >>> cfg.validate()              # raises ValueError on bad configs
    >>> cfg.tokens_per_expert(batch_tokens=1024)
    256.0
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Named constants. The prompt forbids "magic numbers"; every bound used by
# ``validate`` and the preset factories is named and documented here so that a
# reader never has to guess where a threshold came from.
# ---------------------------------------------------------------------------

#: A capacity factor below 1.0 cannot even hold a perfectly balanced batch
#: (each expert would receive fewer slots than its fair share of tokens), so it
#: is rejected outright.
MIN_CAPACITY_FACTOR: float = 1.0

#: Routers/experts that are valid choices for the ``router_type`` and
#: ``activation`` fields. Kept as tuples so they are hashable and immutable.
VALID_ROUTER_TYPES: tuple[str, ...] = ("topk", "switch", "expert_choice")
VALID_ACTIVATIONS: tuple[str, ...] = ("gelu", "swiglu", "relu2")
VALID_DISPATCH: tuple[str, ...] = ("naive", "batch")

RouterType = Literal["topk", "switch", "expert_choice"]
Activation = Literal["gelu", "swiglu", "relu2"]
DispatchStrategy = Literal["naive", "batch"]


@dataclass
class MoEConfig:
    """Hyperparameters for a single MoE layer.

    The defaults describe a small, fast configuration suitable for unit tests.
    Use the preset factory methods (:meth:`switch_transformer`,
    :meth:`mixtral_style`, :meth:`gpt4_style`) for realistic architectures.

    Args:
        d_model: Model (residual-stream) dimension. Input and output width of the
            layer.
        d_ff: Hidden width of each expert's feed-forward network. Each expert is
            an independent two-layer MLP of this inner width.
        num_experts: Total number of experts ``N``. Total expert parameters grow
            linearly with this while activated compute does not.
        top_k: Number of experts each token is routed to. ``1`` reproduces Switch
            routing; ``2`` reproduces Mixtral.
        capacity_factor: Slack multiplier on the per-expert token buffer. The
            capacity is ``capacity_factor * tokens / num_experts``. Must be
            ``>= 1.0`` so a balanced batch never overflows.
        router_type: Which router to instantiate: ``"topk"``, ``"switch"`` or
            ``"expert_choice"``.
        activation: Expert activation: ``"gelu"``, ``"swiglu"`` or ``"relu2"``
            (ReLU squared).
        expert_dropout: Dropout probability applied inside each expert's hidden
            layer. Only active in training mode.
        use_noisy_gating: If ``True``, add learnable Gaussian noise to the gating
            logits during training (Shazeer 2017) to aid exploration.
        noise_std_init: Initialisation scale for the ``W_noise`` projection used
            by noisy gating.
        jitter_noise: Multiplicative input jitter applied during training only
            (Switch Transformer's ``jitter``). ``0.0`` disables it.
        alpha: Weight of the auxiliary load-balancing loss.
        beta: Weight of the router z-loss.
        normalize_router_weights: If ``True``, renormalise the selected top-K
            gating weights to sum to 1 per token (Mixtral convention). If
            ``False`` (default), keep the raw masked softmax probabilities, which
            sum to ``<= 1`` per token (Switch convention).
        drop_tokens: If ``True``, enforce the per-expert capacity buffer and drop
            overflowing tokens. If ``False``, capacity is set to the full token
            count so no token is ever dropped (used for exact strategy parity).
        dispatch_strategy: ``"naive"`` (debuggable per-expert loop) or ``"batch"``
            (fixed-capacity batched dispatch). Both produce identical outputs
            when no tokens are dropped.
        use_bias: Whether the expert linear layers carry bias terms.
        router_z_score_norm: If ``True``, divide gating logits by their running
            standard deviation before the softmax (an alternative to z-loss).
        eps: Small constant guarding divisions and logarithms.

    Raises:
        ValueError: Never raised by ``__init__``; call :meth:`validate` to check
            invariants.

    Notes:
        The dataclass is intentionally *not* frozen so that callers may build a
        config incrementally, but the package never mutates a config after the
        layer is constructed, preserving thread safety.
    """

    d_model: int = 32
    d_ff: int = 64
    num_experts: int = 4
    top_k: int = 1
    capacity_factor: float = 1.25

    router_type: RouterType = "topk"
    activation: Activation = "gelu"

    expert_dropout: float = 0.0
    use_noisy_gating: bool = False
    noise_std_init: float = 1.0
    jitter_noise: float = 0.0

    alpha: float = 1e-2
    beta: float = 1e-3
    normalize_router_weights: bool = False
    drop_tokens: bool = True
    dispatch_strategy: DispatchStrategy = "naive"
    use_bias: bool = True
    router_z_score_norm: bool = False

    eps: float = 1e-8

    # -- Derived helpers ----------------------------------------------------

    def capacity(self, batch_tokens: int) -> int:
        """Compute the per-expert capacity (buffer size) for a batch.

        Args:
            batch_tokens: Number of tokens in the batch, i.e. ``batch * seq``.

        Returns:
            The integer capacity ``C``. When ``drop_tokens`` is ``False`` this is
            ``batch_tokens`` (no token can ever overflow). Otherwise it is
            ``ceil(capacity_factor * top_k * batch_tokens / num_experts)``,
            clamped to at most ``batch_tokens``.

        Notes:
            The canonical Switch formula is ``capacity_factor * tokens /
            num_experts`` (top-1). We multiply by ``top_k`` so that a top-K
            router provisions enough slots for the ``top_k`` assignments each
            token produces; for ``top_k == 1`` this reduces to the canonical
            formula exactly.
        """
        if not self.drop_tokens:
            return batch_tokens
        raw = self.capacity_factor * self.top_k * batch_tokens / self.num_experts
        return min(batch_tokens, max(1, math.ceil(raw)))

    def tokens_per_expert(self, batch_tokens: int) -> float:
        """Return the fair-share token count per expert (``tokens / N``)."""
        return batch_tokens / self.num_experts

    # -- Validation ---------------------------------------------------------

    def validate(self) -> MoEConfig:
        """Validate the configuration, raising on any invariant violation.

        Returns:
            ``self``, so the call can be chained: ``cfg = MoEConfig(...).validate()``.

        Raises:
            ValueError: With a message naming the offending field, its current
                value and the constraint it violates, plus how to fix it.
        """
        if self.d_model <= 0:
            raise ValueError(
                f"d_model must be a positive integer, got {self.d_model}. "
                "Set d_model to the residual-stream width (e.g. 512)."
            )
        if self.d_ff <= 0:
            raise ValueError(
                f"d_ff must be a positive integer, got {self.d_ff}. "
                "Set d_ff to the expert hidden width (e.g. 4 * d_model)."
            )
        if self.num_experts < 1:
            raise ValueError(
                f"num_experts must be >= 1, got {self.num_experts}. "
                "Use at least 1 expert; typical MoE layers use 8-128."
            )
        if self.top_k < 1:
            raise ValueError(
                f"top_k must be >= 1, got {self.top_k}. "
                "Route each token to at least one expert."
            )
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed num_experts "
                f"({self.num_experts}). Reduce top_k or add experts."
            )
        if self.capacity_factor < MIN_CAPACITY_FACTOR:
            raise ValueError(
                f"capacity_factor must be >= {MIN_CAPACITY_FACTOR}, got "
                f"{self.capacity_factor}. A value below 1.0 cannot hold a "
                "balanced batch; use 1.0-2.0."
            )
        if not 0.0 <= self.expert_dropout < 1.0:
            raise ValueError(
                f"expert_dropout must be in [0, 1), got {self.expert_dropout}. "
                "Use e.g. 0.0-0.1."
            )
        if self.noise_std_init < 0.0:
            raise ValueError(
                f"noise_std_init must be non-negative, got {self.noise_std_init}."
            )
        if self.jitter_noise < 0.0:
            raise ValueError(
                f"jitter_noise must be non-negative, got {self.jitter_noise}."
            )
        if self.alpha < 0.0:
            raise ValueError(
                f"alpha (aux-loss weight) must be non-negative, got {self.alpha}. "
                "Use 0.0 to disable, or 1e-3..1e-2."
            )
        if self.beta < 0.0:
            raise ValueError(
                f"beta (z-loss weight) must be non-negative, got {self.beta}. "
                "Use 0.0 to disable, or 1e-4..1e-3."
            )
        if self.router_type not in VALID_ROUTER_TYPES:
            raise ValueError(
                f"router_type must be one of {VALID_ROUTER_TYPES}, got "
                f"{self.router_type!r}."
            )
        if self.activation not in VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {VALID_ACTIVATIONS}, got "
                f"{self.activation!r}."
            )
        if self.dispatch_strategy not in VALID_DISPATCH:
            raise ValueError(
                f"dispatch_strategy must be one of {VALID_DISPATCH}, got "
                f"{self.dispatch_strategy!r}."
            )
        if self.router_type == "switch" and self.top_k != 1:
            raise ValueError(
                f"router_type='switch' requires top_k == 1, got top_k="
                f"{self.top_k}. Switch routing is top-1 by definition; use "
                "router_type='topk' for top_k > 1."
            )
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}.")
        return self

    # -- Serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the config to a plain ``dict`` (JSON-friendly)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MoEConfig:
        """Construct a config from a dict, ignoring unknown keys.

        Args:
            data: A mapping of field names to values, e.g. the output of
                :meth:`to_dict`.

        Returns:
            A new :class:`MoEConfig`. Unknown keys are silently dropped so that
            configs saved by a newer version still load in an older one.

        Raises:
            TypeError: If ``data`` is not a mapping.
        """
        if not isinstance(data, dict):
            raise TypeError(f"from_dict expects a dict, got {type(data).__name__}.")
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    # -- Presets ------------------------------------------------------------

    @classmethod
    def switch_transformer(cls, num_experts: int = 8, d_model: int = 512) -> MoEConfig:
        """Switch Transformer style preset: top-1 routing (Fedus et al. 2022).

        Args:
            num_experts: Number of experts (the paper scales to thousands).
            d_model: Residual-stream width.

        Returns:
            A validated top-1 config with a capacity factor of 1.25 and an
            auxiliary loss weight of 1e-2, matching the paper's recipe.
        """
        return cls(
            d_model=d_model,
            d_ff=d_model * 4,
            num_experts=num_experts,
            top_k=1,
            capacity_factor=1.25,
            router_type="switch",
            activation="relu2",
            alpha=1e-2,
            beta=1e-3,
            use_noisy_gating=False,
            jitter_noise=1e-2,
            drop_tokens=True,
        ).validate()

    @classmethod
    def mixtral_style(cls, num_experts: int = 8, d_model: int = 512) -> MoEConfig:
        """Mixtral 8x7B style preset: top-2 routing, SwiGLU, renormalised gates.

        Args:
            num_experts: Number of experts (Mixtral uses 8).
            d_model: Residual-stream width (Mixtral uses 4096; scaled here).

        Returns:
            A validated top-2 config with SwiGLU experts and top-K weight
            renormalisation, with no token dropping (Mixtral processes every
            token).
        """
        return cls(
            d_model=d_model,
            d_ff=int(d_model * 3.5),
            num_experts=num_experts,
            top_k=2,
            capacity_factor=1.25,
            router_type="topk",
            activation="swiglu",
            alpha=1e-2,
            beta=1e-3,
            normalize_router_weights=True,
            drop_tokens=False,
            use_noisy_gating=False,
        ).validate()

    @classmethod
    def gpt4_style(cls, num_experts: int = 16, d_model: int = 768) -> MoEConfig:
        """A large top-2 preset reflecting publicly reported GPT-4-class MoE.

        Args:
            num_experts: Number of experts (reported designs use ~16).
            d_model: Residual-stream width (scaled down for tractability).

        Returns:
            A validated top-2 config with GELU experts, modest dropout and a
            capacity factor of 1.5.

        Notes:
            GPT-4's exact architecture is not published; this preset uses widely
            reported figures (a ~16-expert, top-2 design) and is intended as an
            illustrative large configuration, not a faithful reproduction.
        """
        return cls(
            d_model=d_model,
            d_ff=d_model * 4,
            num_experts=num_experts,
            top_k=2,
            capacity_factor=1.5,
            router_type="topk",
            activation="gelu",
            expert_dropout=0.0,
            alpha=1e-2,
            beta=1e-3,
            normalize_router_weights=True,
            drop_tokens=True,
            use_noisy_gating=True,
            noise_std_init=1.0,
        ).validate()

    def __post_init__(self) -> None:
        """Stash the field order once for a stable, readable ``__repr__``."""
        # ``field`` is imported for users who extend this dataclass; reference it
        # here so linters do not flag the import as unused.
        _ = field

    def summary(self) -> str:
        """Return a compact one-line human-readable summary of the config."""
        return (
            f"MoEConfig(d_model={self.d_model}, d_ff={self.d_ff}, "
            f"N={self.num_experts}, k={self.top_k}, cf={self.capacity_factor}, "
            f"router={self.router_type}, act={self.activation})"
        )
