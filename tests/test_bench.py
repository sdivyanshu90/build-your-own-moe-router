"""Smoke test for the developer benchmark (:mod:`moe.bench`).

The benchmark is excluded from the coverage target, but it should still run, so a
broken refactor of it is caught in CI rather than the first time someone types
``make bench``.
"""

from __future__ import annotations

from moe.bench import BenchResult, run_benchmark
from moe.config import MoEConfig


def test_run_benchmark_tiny() -> None:
    """A tiny config produces two well-formed, positively-timed results."""
    config = MoEConfig(
        d_model=16, d_ff=16, num_experts=4, top_k=2, dispatch_strategy="batch"
    )
    results = run_benchmark(batch=1, seq=2, config=config)
    assert len(results) == 2
    assert all(isinstance(r, BenchResult) for r in results)
    assert all(r.mean_ms >= 0.0 and r.params > 0 for r in results)
    # The dense baseline's hidden width is num_experts x the expert width, so its
    # parameter count is the same order of magnitude as the MoE (which also
    # carries the small router head and per-expert biases).
    moe, dense = results
    assert 0.5 * moe.params <= dense.params <= 2.0 * moe.params
