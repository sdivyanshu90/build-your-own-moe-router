"""Monitoring, diagnostics and visualisation utilities for MoE routing.

These standalone helpers turn raw router outputs into the health metrics every
MoE training run should log: routing entropy, load imbalance, dead-expert
detection, a routing heatmap and a JSONL exporter for offline analysis.

Example:
    >>> import torch
    >>> probs = torch.full((4, 8), 1.0 / 8)         # uniform over 8 experts
    >>> stats = compute_routing_entropy(probs)
    >>> abs(float(stats.mean) - torch.log(torch.tensor(8.0))) < 1e-6
    tensor(True)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure


class EntropyStats(NamedTuple):
    """Routing-entropy summary over a batch of tokens.

    Attributes:
        per_token: ``[num_tokens]`` entropy of each token's routing distribution.
        mean: Scalar mean entropy over the batch.
        min: Scalar minimum entropy over the batch (the most collapsed token).
    """

    per_token: Tensor
    mean: Tensor
    min: Tensor


class ImbalanceStats(NamedTuple):
    """Load-imbalance summary across experts.

    Attributes:
        imbalance_ratio: ``max_i f_i / mean_i f_i``. 1.0 means perfect balance.
        coefficient_of_variation: ``std_i f_i / mean_i f_i``. 0.0 means perfect
            balance; grows without an upper bound as load concentrates.
    """

    imbalance_ratio: Tensor
    coefficient_of_variation: Tensor


def compute_routing_entropy(router_probs: Tensor) -> EntropyStats:
    r"""Compute per-token, mean and minimum routing entropy.

    The Shannon entropy of a token's routing distribution is

    .. math::
        H = -\sum_{i=1}^{N} p_i \log p_i

    in nats. A uniform distribution over ``N`` experts gives ``log(N)`` (maximal
    diversity); a one-hot distribution gives 0 (fully collapsed routing).

    Args:
        router_probs: Full softmax routing distribution ``[num_tokens,
            num_experts]``.

    Returns:
        An :class:`EntropyStats` with per-token entropy and its batch mean/min.

    Raises:
        ValueError: If ``router_probs`` is not 2-D.

    Notes:
        ``torch.special.entr`` computes ``-p log p`` with the correct ``0`` at
        ``p = 0`` (avoiding the ``0 * -inf`` NaN of a naive ``p * p.log()``).

    Example:
        >>> import torch
        >>> onehot = torch.eye(3)[[0, 1, 2]]   # 3 one-hot tokens, 3 experts
        >>> float(compute_routing_entropy(onehot).mean)
        0.0
    """
    if router_probs.dim() != 2:
        raise ValueError(
            f"router_probs must be 2-D [num_tokens, num_experts], got shape "
            f"{tuple(router_probs.shape)}."
        )
    p = router_probs.float()
    per_token = torch.special.entr(p).sum(dim=-1)  # [num_tokens], 0*log0 handled
    return EntropyStats(per_token=per_token, mean=per_token.mean(), min=per_token.min())


def compute_load_imbalance(expert_counts: Tensor) -> ImbalanceStats:
    r"""Compute the load-imbalance ratio and coefficient of variation.

    Given per-expert token counts ``c_i`` with fractions ``f_i = c_i / \sum_j
    c_j``, the imbalance ratio is ``max_i f_i / mean_i f_i`` (equivalently
    ``max_i c_i / mean_i c_i``) and the coefficient of variation is ``std_i c_i /
    mean_i c_i``.

    Args:
        expert_counts: 1-D tensor of per-expert token counts (any numeric dtype).

    Returns:
        An :class:`ImbalanceStats`. Perfect balance yields ``(1.0, 0.0)``.

    Raises:
        ValueError: If ``expert_counts`` is not 1-D or is empty.

    Example:
        >>> import torch
        >>> float(compute_load_imbalance(torch.tensor([4, 4, 4, 4])).imbalance_ratio)
        1.0
    """
    if expert_counts.dim() != 1 or expert_counts.numel() == 0:
        raise ValueError(
            f"expert_counts must be a non-empty 1-D tensor, got shape "
            f"{tuple(expert_counts.shape)}."
        )
    counts = expert_counts.float()
    mean = counts.mean()
    # Guard against an all-zero batch (no tokens routed at all).
    safe_mean = mean.clamp_min(torch.finfo(counts.dtype).tiny)
    imbalance = counts.max() / safe_mean
    # Population standard deviation (unbiased=False) so a single expert gives 0.
    cv = counts.std(unbiased=False) / safe_mean
    return ImbalanceStats(imbalance_ratio=imbalance, coefficient_of_variation=cv)


def detect_dead_experts(expert_counts: Tensor, threshold: float = 0.01) -> list[int]:
    """Identify experts receiving less than ``threshold`` of the tokens.

    Args:
        expert_counts: 1-D tensor of per-expert token counts.
        threshold: Minimum acceptable utilisation fraction. An expert whose share
            of tokens is strictly below this is reported as dead. Defaults to
            ``0.01`` (1%).

    Returns:
        A sorted list of dead-expert indices (possibly empty).

    Raises:
        ValueError: If ``expert_counts`` is not 1-D or ``threshold`` is not in
            ``[0, 1]``.

    Example:
        >>> import torch
        >>> detect_dead_experts(torch.tensor([10, 10, 10, 0]))
        [3]
    """
    if expert_counts.dim() != 1:
        raise ValueError(
            f"expert_counts must be 1-D, got shape {tuple(expert_counts.shape)}."
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}.")
    counts = expert_counts.float()
    total = counts.sum().clamp_min(1.0)  # avoid div-by-zero; counts are >= 0
    fractions = counts / total
    dead = torch.nonzero(fractions < threshold, as_tuple=False).flatten()
    return [int(i) for i in dead.tolist()]


def visualise_routing(
    router_probs: Tensor, tokens: Sequence[str] | None = None
) -> Figure:
    """Render a heatmap of routing probabilities (tokens x experts).

    Args:
        router_probs: Full routing distribution ``[num_tokens, num_experts]``.
        tokens: Optional per-token text labels for the y-axis. If provided its
            length must equal ``num_tokens``.

    Returns:
        A Matplotlib :class:`~matplotlib.figure.Figure` (created without pyplot,
        so it is safe to use off the main thread and in headless environments).

    Raises:
        ImportError: If Matplotlib is not installed.
        ValueError: If ``router_probs`` is not 2-D or ``tokens`` length mismatches.

    Example:
        >>> import torch
        >>> fig = visualise_routing(torch.softmax(torch.randn(5, 4), -1))
        >>> fig.axes[0].get_xlabel()
        'Expert'
    """
    if router_probs.dim() != 2:
        raise ValueError(
            f"router_probs must be 2-D, got shape {tuple(router_probs.shape)}."
        )
    try:
        # Import locally: matplotlib is an optional dependency.
        from matplotlib.figure import Figure as MplFigure
    except ImportError as exc:  # pragma: no cover - exercised only without mpl
        raise ImportError(
            "visualise_routing requires matplotlib. Install it with "
            "`pip install matplotlib`."
        ) from exc

    probs = router_probs.detach().float().cpu().numpy()
    num_tokens, num_experts = probs.shape
    if tokens is not None and len(tokens) != num_tokens:
        raise ValueError(
            f"tokens length ({len(tokens)}) must equal num_tokens ({num_tokens})."
        )

    fig = MplFigure(figsize=(max(4, num_experts), max(3, num_tokens * 0.3)))
    ax = fig.add_subplot(111)
    image = ax.imshow(probs, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Token")
    ax.set_xticks(range(num_experts))
    if tokens is not None:
        ax.set_yticks(range(num_tokens))
        ax.set_yticklabels(list(tokens))
    fig.colorbar(image, ax=ax, label="Routing probability")
    fig.tight_layout()
    return fig


def export_routing_stats(
    stats_list: Sequence[dict[str, Any]], path: str | Path
) -> Path:
    """Serialise a list of per-step stat dicts to a JSONL file.

    Args:
        stats_list: A sequence of JSON-serialisable dicts, one per training step.
        path: Destination file path. Parent directories are created if needed.

    Returns:
        The resolved :class:`~pathlib.Path` that was written.

    Raises:
        TypeError: If any element is not a dict or contains non-serialisable
            values (tensors are converted to Python scalars/lists first).

    Notes:
        Tensor values are converted with ``.tolist()``/``float()`` so callers can
        pass raw metric dicts without manual conversion.

    Example:
        >>> import tempfile, os
        >>> p = export_routing_stats([{"step": 0, "entropy": 1.2}],
        ...     os.path.join(tempfile.mkdtemp(), "stats.jsonl"))
        >>> p.exists()
        True
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_serialisable(value: Any) -> Any:
        """Convert tensors/numpy scalars to plain JSON-friendly Python values."""
        if isinstance(value, Tensor):
            return value.detach().cpu().tolist()
        return value

    with out_path.open("w", encoding="utf-8") as handle:
        for i, record in enumerate(stats_list):
            if not isinstance(record, dict):
                raise TypeError(
                    f"stats_list[{i}] must be a dict, got {type(record).__name__}."
                )
            clean = {k: _to_serialisable(v) for k, v in record.items()}
            handle.write(json.dumps(clean) + "\n")
    return out_path
