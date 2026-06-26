"""moe — a production-grade Mixture of Experts (MoE) routing layer.

This package provides a numerically-stable, fully-typed, drop-in sparse
Mixture-of-Experts feed-forward layer in the style of Switch Transformer, GLaM
and Mixtral, together with the routers, losses and monitoring utilities needed to
train it.

Public API:
    The supported surface is exactly the names re-exported here. Everything else
    (``moe.experts``, ``moe.utils`` internals, helper functions) may change
    without notice.

    * :class:`MoELayer` — the complete MoE FFN replacement.
    * :class:`MoEConfig` — the validated hyperparameter dataclass.
    * :class:`MoELoss` — the combined auxiliary + z-loss module.
    * :class:`TopKRouter`, :class:`SwitchRouter`, :class:`ExpertChoiceRouter` —
      the gating networks.

Versioning:
    This package follows semantic versioning. The public API above is covered by
    the compatibility guarantee: within a major version, names are not removed
    and signatures are only extended with optional, defaulted arguments. The
    internal modules carry no such guarantee.

Example:
    >>> import torch
    >>> from moe import MoELayer, MoEConfig
    >>> layer = MoELayer(MoEConfig.mixtral_style(num_experts=4, d_model=32))
    >>> hidden = torch.randn(2, 8, 32)
    >>> out = layer(hidden)
    >>> residual = hidden + out.output
    >>> loss = out.aux_loss + out.z_loss        # add to your task loss
    >>> residual.shape
    torch.Size([2, 8, 32])
"""

from __future__ import annotations

from .config import MoEConfig
from .layer import MoELayer, MoELayerOutput
from .losses import LossOutput, MoELoss
from .router import (
    ExpertChoiceRouter,
    RouterOutput,
    SwitchRouter,
    TopKRouter,
)

__version__ = "1.0.0"

__all__ = [
    "MoELayer",
    "MoEConfig",
    "MoELoss",
    "TopKRouter",
    "SwitchRouter",
    "ExpertChoiceRouter",
    # Output containers are part of the typed public surface.
    "MoELayerOutput",
    "RouterOutput",
    "LossOutput",
    "__version__",
]
