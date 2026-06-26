"""Integration tests: training dynamics, mixed precision, serialisation.

These exercise the whole stack as a user would: a small two-layer network whose
FFNs are :class:`MoELayer`s, trained on a regression task. They confirm the layer
trains, that the auxiliary loss is what prevents expert collapse, that it runs
under bfloat16 autocast, and that it serialises losslessly.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from moe.config import MoEConfig
from moe.layer import MoELayer


class TrainTrace(NamedTuple):
    """Per-step metric traces returned by :func:`_train`."""

    total: list[float]
    imbalance: list[float]
    entropy: list[float]
    layer: MoELayer


# A low-diversity input (shared base direction + small per-token noise) creates
# the rich-get-richer pressure that collapses routing when load balancing is off.
INPUT_DIVERSITY = 0.3


def _train(
    alpha: float,
    steps: int,
    *,
    lr: float = 1e-2,
    num_experts: int = 8,
    d_model: int = 64,
    d_ff: int = 128,
    model_seed: int = 0,
    data_seed: int = 123,
) -> TrainTrace:
    """Train a two-layer MoE network and return its per-step metric traces."""
    torch.manual_seed(model_seed)
    config = MoEConfig(
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=1,
        router_type="switch",
        alpha=alpha,
        beta=1e-3,
        use_noisy_gating=False,
    )
    layer1, layer2 = MoELayer(config), MoELayer(config)
    head = nn.Linear(d_model, d_model)
    params = [*layer1.parameters(), *layer2.parameters(), *head.parameters()]
    optimiser = torch.optim.Adam(params, lr=lr)

    gen = torch.Generator().manual_seed(data_seed)
    base = torch.randn(1, 1, d_model, generator=gen)
    x = base + INPUT_DIVERSITY * torch.randn(8, 16, d_model, generator=gen)
    target = torch.randn(8, 16, d_model, generator=gen)

    total_trace, imbalance_trace, entropy_trace = [], [], []
    for _ in range(steps):
        out1 = layer1(x)
        hidden = x + out1.output
        out2 = layer2(hidden)
        prediction = head(hidden + out2.output)
        task = ((prediction - target) ** 2).mean()
        loss = task + out1.aux_loss + out2.aux_loss + out1.z_loss + out2.z_loss
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        stats = layer1.get_routing_stats()
        total_trace.append(loss.item())
        imbalance_trace.append(stats["imbalance_ratio"])
        entropy_trace.append(stats["entropy_mean"])

    return TrainTrace(
        total=total_trace,
        imbalance=imbalance_trace,
        entropy=entropy_trace,
        layer=layer1,
    )


def _moving_average(values: list[float], window: int = 10) -> list[float]:
    """Simple trailing moving average of ``values``."""
    return [
        sum(values[i : i + window]) / window for i in range(len(values) - window + 1)
    ]


def test_training_converges_and_diversifies() -> None:
    """Loss decreases (smoothed) and routing entropy rises as experts specialise.

    Why it matters: this is the headline promise of the layer — it trains and the
    auxiliary loss spreads tokens across experts rather than collapsing.
    """
    result = _train(alpha=1e-2, steps=100)
    ma = _moving_average(result.total, window=10)
    # Overall the smoothed loss drops substantially...
    assert ma[-1] < 0.5 * ma[0]
    # ...and is non-increasing at almost every step (tiny optimisation noise ok).
    non_increasing = sum(ma[i + 1] <= ma[i] + 1e-6 for i in range(len(ma) - 1))
    assert non_increasing / (len(ma) - 1) >= 0.9

    # Routing entropy increases from step 0 to step 50 (diversification).
    assert result.entropy[50] > result.entropy[0]


def test_routing_collapses_without_aux_loss() -> None:
    """With ``alpha=0`` the load imbalance grows past 2.0 (collapse occurs)."""
    result = _train(alpha=0.0, steps=200)
    final = sum(result.imbalance[-20:]) / 20
    assert final > 2.0


def test_aux_loss_prevents_collapse() -> None:
    """With the default ``alpha`` the load imbalance stays below 1.5."""
    result = _train(alpha=1e-2, steps=200)
    final = sum(result.imbalance[-20:]) / 20
    assert final < 1.5


def test_mixed_precision_bfloat16() -> None:
    """Under bfloat16 autocast the output is bf16 while losses stay float32.

    Why it matters: routing probabilities and loss accumulation need float32 for
    stability even when the expert matmuls run in low precision.
    """
    layer = MoELayer(MoEConfig(d_model=32, d_ff=64, num_experts=4, top_k=2))
    x = torch.randn(2, 8, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = layer(x)
    assert out.output.dtype == torch.bfloat16
    assert out.aux_loss.dtype == torch.float32
    assert out.z_loss.dtype == torch.float32
    assert torch.isfinite(out.output.float()).all()


def test_mixed_precision_dispatch_strategies_consistent() -> None:
    """Both dispatch strategies share the same output dtype under autocast.

    Why it matters: the naive and batched paths must be interchangeable. The
    batched path uses matmul (autocast-eligible) rather than einsum, so under
    bfloat16 autocast it produces bf16 output just like the naive F.linear path —
    not float32 — while the auxiliary losses stay float32 in both.
    """
    x = torch.randn(2, 8, 32)
    output_dtypes = {}
    for strategy in ("naive", "batch"):
        layer = MoELayer(
            MoEConfig(
                d_model=32, d_ff=64, num_experts=4, top_k=2,
                dispatch_strategy=strategy,
            )
        )
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = layer(x)
        output_dtypes[strategy] = out.output.dtype
        assert out.aux_loss.dtype == torch.float32
    assert output_dtypes["naive"] == output_dtypes["batch"] == torch.bfloat16


def test_state_dict_roundtrip() -> None:
    """A reloaded layer reproduces the original output bit-for-bit in eval."""
    config = MoEConfig(
        d_model=32,
        d_ff=64,
        num_experts=4,
        top_k=2,
        use_noisy_gating=True,
        router_z_score_norm=True,
    )
    torch.manual_seed(1)
    source = MoELayer(config).eval()
    x = torch.randn(2, 8, 32)

    target = MoELayer(config)
    target.load_state_dict(source.state_dict())
    target.eval()

    assert torch.equal(source(x).output, target(x).output)
