# moe-routing

A production-grade **Mixture of Experts (MoE) routing layer** for PyTorch, in the
style of Switch Transformer, GLaM and Mixtral — fully typed, numerically stable,
device- and dtype-agnostic, and exhaustively tested (100% line coverage).

It gives you a drop-in replacement for a dense transformer FFN that activates
only `top_k` of `num_experts` experts per token, so you scale total parameters
without scaling per-token compute.

## Install

```bash
pip install -e ".[dev]"     # editable install with test/lint tooling
```

Requires Python ≥ 3.10 and `torch>=2.0` (`numpy` required, `matplotlib` optional
for routing visualisations).

## Quickstart

```python
import torch
from moe import MoELayer, MoEConfig

# Build a Mixtral-style top-2 layer (8 experts, SwiGLU, renormalised gates).
config = MoEConfig.mixtral_style(num_experts=8, d_model=512)
layer = MoELayer(config)

hidden = torch.randn(4, 128, 512)          # [batch, seq, d_model]
out = layer(hidden)

# Use it exactly like a dense FFN sub-layer:
hidden = hidden + out.output               # residual connection
loss = task_loss + out.aux_loss + out.z_loss   # add the MoE regularisers

# Monitor routing health every step:
print(layer.get_routing_stats())
# {'entropy_mean': 1.96, 'imbalance_ratio': 1.18, 'overflow_fraction': 0.0, ...}
```

## What you get

- **Three routers** behind one interface: `TopKRouter` (softmax / noisy top-K),
  `SwitchRouter` (top-1 with a capacity buffer), `ExpertChoiceRouter` (experts
  pick their tokens).
- **Load balancing** that actually prevents expert collapse: the Switch
  auxiliary loss `α·N·Σ fᵢ·Pᵢ` plus the ST-MoE router z-loss.
- **Numerical stability** by construction: `logsumexp`-based z-loss, float32
  softmax/loss accumulation under bfloat16 autocast, max-shift softmax.
- **Two dispatch strategies** — a readable naive loop and a fused batched
  `matmul` path — proven to agree to within `1e-4`.
- **Monitoring** utilities: routing entropy, load imbalance, dead-expert
  detection, a routing heatmap, and a JSONL stats exporter.

## Presets

```python
MoEConfig.switch_transformer()   # top-1, capacity 1.25, ReLU², jitter
MoEConfig.mixtral_style()        # top-2, SwiGLU, renormalised gates, no drop
MoEConfig.gpt4_style()           # 16-expert top-2, GELU, noisy gating
```

## Developer commands

```bash
make test     # pytest with coverage (100% on the library)
make lint     # ruff + mypy --strict
make format   # ruff format + autofix
make bench    # forward-pass timing: MoE vs equal-parameter dense FFN
make docs     # pdoc HTML API docs
```

`make bench` on this machine reports a **~2.6× speedup** for a top-2-of-8 MoE
over an equal-parameter dense FFN.

## Documentation

The full technical document — conceptual foundations, gating variants, the
collapse problem and load balancing, numerical stability, distributed training,
a hyperparameter guide and a monitoring playbook — lives in
[`docs/moe_routing.md`](docs/moe_routing.md).

## Public API

`MoELayer`, `MoEConfig`, `MoELoss`, `TopKRouter`, `SwitchRouter`,
`ExpertChoiceRouter` (plus the typed output containers `MoELayerOutput`,
`RouterOutput`, `LossOutput`). Semantic-versioned; within a major version these
names and signatures are stable.

## License

MIT.
