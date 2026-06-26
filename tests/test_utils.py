"""Tests for :mod:`moe.utils` monitoring helpers.

The diagnostics here are what an operator watches to catch expert collapse early,
so their boundary behaviour (uniform vs one-hot entropy, perfectly balanced load,
a dead expert) must be exactly right.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from moe.utils import (
    compute_load_imbalance,
    compute_routing_entropy,
    detect_dead_experts,
    export_routing_stats,
    visualise_routing,
)


def test_entropy_uniform_and_onehot() -> None:
    """Uniform routing has entropy ``log(N)``; one-hot routing has entropy 0."""
    num_experts = 8
    uniform = torch.full((4, num_experts), 1.0 / num_experts)
    onehot = torch.zeros(4, num_experts)
    onehot[:, 0] = 1.0

    assert float(compute_routing_entropy(uniform).mean) == pytest.approx(
        math.log(num_experts), abs=1e-6
    )
    assert float(compute_routing_entropy(onehot).mean) == pytest.approx(0.0, abs=1e-6)


def test_entropy_reports_per_token_and_min() -> None:
    """The helper returns per-token entropy plus the batch mean and min."""
    probs = torch.softmax(torch.randn(5, 4), dim=-1)
    stats = compute_routing_entropy(probs)
    assert stats.per_token.shape == (5,)
    assert float(stats.min) <= float(stats.mean)


def test_entropy_rejects_bad_shape() -> None:
    """A non-2-D probability tensor raises an informative error."""
    with pytest.raises(ValueError, match="must be 2-D"):
        compute_routing_entropy(torch.rand(4))


def test_load_imbalance_perfect_balance() -> None:
    """Equal expert counts give imbalance ratio 1.0 and CV 0.0."""
    stats = compute_load_imbalance(torch.tensor([4, 4, 4, 4]))
    assert float(stats.imbalance_ratio) == pytest.approx(1.0)
    assert float(stats.coefficient_of_variation) == pytest.approx(0.0, abs=1e-6)


def test_load_imbalance_collapsed() -> None:
    """A single dominant expert pushes the imbalance ratio toward N."""
    stats = compute_load_imbalance(torch.tensor([10, 0, 0, 0]))
    assert float(stats.imbalance_ratio) == pytest.approx(4.0)


def test_load_imbalance_rejects_bad_shape() -> None:
    """Non-1-D or empty counts raise an informative error."""
    with pytest.raises(ValueError, match="non-empty 1-D"):
        compute_load_imbalance(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="non-empty 1-D"):
        compute_load_imbalance(torch.tensor([]))


def test_detect_dead_experts() -> None:
    """An expert receiving no tokens is reported as dead."""
    counts = torch.tensor([10, 10, 10, 0])
    assert detect_dead_experts(counts) == [3]
    # With a high threshold the lightly-used experts also count as dead.
    assert detect_dead_experts(torch.tensor([97, 1, 1, 1]), threshold=0.02) == [1, 2, 3]


def test_detect_dead_experts_validation() -> None:
    """Bad rank or out-of-range threshold raises an informative error."""
    with pytest.raises(ValueError, match="must be 1-D"):
        detect_dead_experts(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="threshold"):
        detect_dead_experts(torch.tensor([1, 2]), threshold=1.5)


def test_visualise_routing_returns_figure() -> None:
    """The heatmap figure has the expected axis labels, with/without tokens."""
    probs = torch.softmax(torch.randn(3, 4), dim=-1)
    fig = visualise_routing(probs, tokens=["a", "b", "c"])
    assert fig.axes[0].get_xlabel() == "Expert"
    assert fig.axes[0].get_ylabel() == "Token"
    # Also exercise the no-labels path.
    assert visualise_routing(probs).axes[0].get_xlabel() == "Expert"


def test_visualise_routing_validation() -> None:
    """Bad rank or mismatched token labels raise informative errors."""
    with pytest.raises(ValueError, match="must be 2-D"):
        visualise_routing(torch.rand(4))
    with pytest.raises(ValueError, match="must equal num_tokens"):
        visualise_routing(torch.rand(3, 4), tokens=["only", "two"])


def test_export_routing_stats_roundtrip(tmp_path: Path) -> None:
    """Stats (including tensors) serialise to readable JSONL."""
    records = [
        {"step": 0, "entropy": 1.2, "util": torch.tensor([0.5, 0.5])},
        {"step": 1, "entropy": 1.3, "util": torch.tensor([0.6, 0.4])},
    ]
    out_path = export_routing_stats(records, tmp_path / "sub" / "stats.jsonl")
    assert out_path.exists()

    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["step"] == 0
    assert first["util"] == [0.5, 0.5]  # tensor converted to a list


def test_export_routing_stats_rejects_non_dict(tmp_path: Path) -> None:
    """A non-dict record raises ``TypeError`` naming the offending index."""
    with pytest.raises(TypeError, match=r"stats_list\[0\]"):
        export_routing_stats([["not", "a", "dict"]], tmp_path / "bad.jsonl")
