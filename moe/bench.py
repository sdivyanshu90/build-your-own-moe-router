"""Forward-pass micro-benchmark: sparse MoE vs an equivalent dense FFN.

Run with ``python -m moe.bench`` (or ``make bench``). The dense baseline has the
*same total parameter count* as the MoE bank (``num_experts`` x the expert
width), so the timing gap isolates the benefit of activating only ``top_k`` of
``num_experts`` experts per token.

This module is a developer tool, not part of the public API.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import MoEConfig
from .layer import MoELayer

#: Benchmark configuration mandated by the project checklist.
WARMUP_ITERS: int = 10
TIMED_ITERS: int = 100
MS_PER_SEC: float = 1_000.0


@dataclass
class BenchResult:
    """Timing summary for one module.

    Attributes:
        name: Human-readable label.
        mean_ms: Mean forward-pass latency in milliseconds.
        std_ms: Standard deviation of the latency in milliseconds.
        params: Total parameter count of the benchmarked module.
    """

    name: str
    mean_ms: float
    std_ms: float
    params: int


def _time_forward(module: nn.Module, x: Tensor) -> tuple[float, float]:
    """Return (mean_ms, std_ms) of ``module(x)`` over the timed iterations.

    Args:
        module: The module to benchmark (run under ``torch.no_grad`` + eval).
        x: A representative input tensor.

    Returns:
        The mean and population standard deviation of the latency, in ms.
    """
    module.eval()
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):  # warm up caches / lazy init
            module(x)
        samples: list[float] = []
        for _ in range(TIMED_ITERS):
            start = time.perf_counter()
            module(x)
            samples.append((time.perf_counter() - start) * MS_PER_SEC)
    return statistics.fmean(samples), statistics.pstdev(samples)


def _dense_baseline(config: MoEConfig) -> nn.Sequential:
    """Build a dense FFN with the same total parameters as the expert bank."""
    hidden = config.d_ff * config.num_experts
    return nn.Sequential(
        nn.Linear(config.d_model, hidden),
        nn.GELU(),
        nn.Linear(hidden, config.d_model),
    )


def run_benchmark(
    batch: int = 8, seq: int = 128, config: MoEConfig | None = None
) -> list[BenchResult]:
    """Benchmark the MoE layer against an equivalent dense FFN.

    Args:
        batch: Batch size of the synthetic input.
        seq: Sequence length of the synthetic input.
        config: Optional config; defaults to the checklist configuration
            (top_k=2, num_experts=8, capacity_factor=1.25).

    Returns:
        A list with the MoE and dense :class:`BenchResult`s.
    """
    if config is None:
        config = MoEConfig(
            d_model=512,
            d_ff=1024,
            num_experts=8,
            top_k=2,
            capacity_factor=1.25,
            dispatch_strategy="batch",
        )
    torch.manual_seed(0)
    x = torch.randn(batch, seq, config.d_model)

    moe = MoELayer(config)
    dense = _dense_baseline(config)

    moe_mean, moe_std = _time_forward(_ForwardOnly(moe), x)
    dense_mean, dense_std = _time_forward(dense, x)

    return [
        BenchResult("MoE (top-2 of 8)", moe_mean, moe_std, _count_params(moe)),
        BenchResult(
            "Dense FFN (equal params)",
            dense_mean,
            dense_std,
            _count_params(dense),
        ),
    ]


class _ForwardOnly(nn.Module):
    """Adapter exposing only the tensor output of an :class:`MoELayer`."""

    def __init__(self, layer: MoELayer) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        """Return just the layer's output tensor."""
        output: Tensor = self.layer(x).output
        return output


def _count_params(module: nn.Module) -> int:
    """Total number of parameters in ``module``."""
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    """Run the benchmark and print a formatted timing table."""
    results = run_benchmark()
    header = f"{'Module':<28}{'params':>14}{'mean (ms)':>14}{'std (ms)':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.name:<28}{r.params:>14,}{r.mean_ms:>14.3f}{r.std_ms:>12.3f}")
    moe, dense = results
    speedup = dense.mean_ms / moe.mean_ms if moe.mean_ms > 0 else float("nan")
    print("-" * len(header))
    print(f"MoE speedup vs equal-parameter dense FFN: {speedup:.2f}x")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
