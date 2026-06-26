"""The complete, composable MoE feed-forward layer.

:class:`MoELayer` is a drop-in replacement for a dense transformer FFN sub-layer.
It wires together a router, an expert bank and the auxiliary losses, and exposes
the routing statistics needed for monitoring. Combine it with a residual
connection exactly as you would a dense FFN::

    layer = MoELayer(MoEConfig.mixtral_style())
    out = layer(hidden_states)
    hidden_states = hidden_states + out.output          # residual
    loss = task_loss + out.aux_loss + out.z_loss        # add MoE regularisers
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
from torch import Tensor, nn

from .config import MoEConfig
from .experts import ExpertBank
from .losses import MoELoss
from .router import RouterOutput, build_router
from .utils import (
    compute_load_imbalance,
    compute_routing_entropy,
    detect_dead_experts,
)


class MoELayerOutput(NamedTuple):
    """Output bundle of :class:`MoELayer`.

    Attributes:
        output: Layer output, same shape as the input ``[batch, seq, d_model]``.
        aux_loss: Scalar *weighted* auxiliary load-balancing loss
            (``alpha * raw``). Add it to your task loss.
        z_loss: Scalar *weighted* router z-loss (``beta * raw``). Add it to your
            task loss.
        router_probs: Full routing distribution ``[num_tokens, num_experts]``
            (detached is the caller's responsibility; kept attached for any
            downstream regularisation).
        expert_utilisation: ``[num_experts]`` fraction of routed slots each expert
            received this forward pass.
    """

    output: Tensor
    aux_loss: Tensor
    z_loss: Tensor
    router_probs: Tensor
    expert_utilisation: Tensor


class MoELayer(nn.Module):
    """A sparse Mixture-of-Experts feed-forward layer.

    Args:
        config: The :class:`~moe.config.MoEConfig` describing the layer.

    Raises:
        ValueError: If ``config`` fails validation.

    Notes:
        The layer returns *weighted* auxiliary losses so callers can simply add
        them to the task loss. Under ``torch.autocast`` the main output flows in
        the autocast dtype (e.g. bfloat16) while the auxiliary losses are
        computed in float32 for stability.
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.router = build_router(config)
        self.expert_bank = ExpertBank(config)
        self.criterion = MoELoss.from_config(config)

        # Detached monitoring state from the most recent forward pass. Instance
        # level (never shared across instances), so no lock is required.
        self._last_router_probs: Tensor | None = None
        self._last_expert_counts: Tensor | None = None

    # -- Helpers ------------------------------------------------------------

    def _expert_counts(self, router_out: RouterOutput) -> Tensor:
        """Count dispatch slots per expert from the selected expert indices."""
        flat = router_out.expert_indices.reshape(-1).detach().to(torch.long)
        return torch.bincount(flat, minlength=self.config.num_experts)

    def _weighted_losses(self, router_out: RouterOutput) -> tuple[Tensor, Tensor]:
        """Compute the weighted aux and z losses via :class:`MoELoss`.

        A zero task loss is supplied because the layer does not know the task
        objective; the caller adds the returned losses to its own task loss.
        """
        zero_task = router_out.router_probs.new_zeros(())
        loss_out = self.criterion(
            zero_task,
            router_out.router_probs,
            router_out.expert_indices,
            router_out.router_logits,
        )
        aux_weighted = self.config.alpha * loss_out.aux_loss
        z_weighted = self.config.beta * loss_out.z_loss
        return aux_weighted, z_weighted

    # -- Forward ------------------------------------------------------------

    def forward(self, x: Tensor) -> MoELayerOutput:
        """Run the full MoE layer.

        Args:
            x: Inputs ``[batch, seq, d_model]`` (or already-flattened
                ``[num_tokens, d_model]``).

        Returns:
            A :class:`MoELayerOutput`.

        Raises:
            ValueError: If the trailing dimension does not match ``config.d_model``.

        Example:
            >>> import torch
            >>> from moe.config import MoEConfig
            >>> layer = MoELayer(MoEConfig(d_model=16, d_ff=32, num_experts=4, top_k=2))
            >>> out = layer(torch.randn(2, 5, 16))
            >>> out.output.shape
            torch.Size([2, 5, 16])
        """
        if x.size(-1) != self.config.d_model:
            raise ValueError(
                f"Input trailing dim ({x.size(-1)}) must equal d_model "
                f"({self.config.d_model})."
            )
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.config.d_model)  # [num_tokens, d_model]

        # 1-2. Route.
        router_out = self.router(x_flat)

        # 3-4. Dispatch to experts and combine (weighted sum done in the bank).
        combined = self.expert_bank(
            x_flat, router_out.dispatch_weights, router_out.expert_indices
        )

        # 5. Restore the original [batch, seq, d_model] shape.
        output = combined.reshape(orig_shape)

        # 6. Compute the weighted auxiliary losses.
        aux_weighted, z_weighted = self._weighted_losses(router_out)

        # Cache detached monitoring state.
        expert_counts = self._expert_counts(router_out)
        self._last_router_probs = router_out.router_probs.detach()
        self._last_expert_counts = expert_counts
        total_slots = expert_counts.sum().clamp_min(1)
        utilisation = expert_counts.float() / total_slots.float()

        # 7. Return everything the caller needs.
        return MoELayerOutput(
            output=output,
            aux_loss=aux_weighted,
            z_loss=z_weighted,
            router_probs=router_out.router_probs,
            expert_utilisation=utilisation,
        )

    # -- Monitoring ---------------------------------------------------------

    def get_routing_stats(self) -> dict[str, Any]:
        """Return routing health metrics from the most recent forward pass.

        Returns:
            A dict with ``entropy_mean``, ``entropy_min``, ``imbalance_ratio``,
            ``coefficient_of_variation``, ``overflow_fraction``, ``dead_experts``
            and ``expert_utilisation``. If :meth:`forward` has not been called
            yet, an empty dict is returned.

        Notes:
            ``overflow_fraction`` is read from the expert bank, which records how
            many dispatch slots were dropped due to capacity on the last call.
        """
        if self._last_router_probs is None or self._last_expert_counts is None:
            return {}

        entropy = compute_routing_entropy(self._last_router_probs)
        imbalance = compute_load_imbalance(self._last_expert_counts.float())
        dead = detect_dead_experts(self._last_expert_counts.float())

        dispatched = max(self.expert_bank.last_dispatched_tokens, 1)
        overflow_fraction = self.expert_bank.last_overflow_tokens / dispatched

        return {
            "entropy_mean": float(entropy.mean),
            "entropy_min": float(entropy.min),
            "imbalance_ratio": float(imbalance.imbalance_ratio),
            "coefficient_of_variation": float(imbalance.coefficient_of_variation),
            "overflow_fraction": float(overflow_fraction),
            "dead_experts": dead,
            "expert_utilisation": (
                self._last_expert_counts.float()
                / self._last_expert_counts.sum().clamp_min(1).float()
            ).tolist(),
        }

    def extra_repr(self) -> str:
        """One-line summary for ``print(module)``."""
        return self.config.summary()
