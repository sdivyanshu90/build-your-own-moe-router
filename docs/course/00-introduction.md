# Lesson 00 — Introduction & Setup

> Get oriented: what you're about to build, why MoE matters, and how this repository is laid out.

## Learning objectives

By the end of this lesson you can:

- Explain in one paragraph what a Mixture of Experts layer is and the problem it solves.
- Install the package and run its tests, lint and benchmark.
- Navigate the repository and say what each module is responsible for.
- Run your first forward pass and read the output.

## Prerequisites

Python 3.10+, a working knowledge of PyTorch tensors and autograd, and a rough
idea of what a transformer feed-forward (FFN) sublayer does. No MoE background
needed.

## The one-paragraph pitch

A dense transformer applies the *same* feed-forward network to every token. That
network's cost and its parameter count are welded together: to make the model
"know more" you widen the FFN, and every token pays for every parameter on every
forward pass. A **Mixture of Experts** breaks that weld. It holds *many* FFNs
(the "experts") but, for each token, a small **router** picks only a few of them
to run. You get the knowledge of a huge parameter count while paying the compute
of a small one. The entire difficulty — and the subject of this course — is the
router: how it chooses experts, how we stop it from lazily sending every token to
the same expert, and how we keep all of this numerically stable and fast.

## Setting up

From the repository root:

```bash
pip install -e ".[dev]"      # installs torch, plus pytest/mypy/ruff/hypothesis/matplotlib/pdoc
```

Verify everything works using the provided `make` targets:

```bash
make test     # full test suite with coverage (the library is at 100%)
make lint     # ruff (style) + mypy --strict (types)
make bench    # times the MoE layer vs an equal-parameter dense FFN
```

`make test` should report all tests passing with 100% line coverage on `moe/`.
`make bench` prints a small table; on a typical CPU it shows the sparse MoE layer
running roughly **2.6× faster** than a dense FFN with the same number of
parameters — a concrete, measurable demonstration of the pitch above.

If you ever want HTML API docs generated from the docstrings, run `make docs`
(uses `pdoc`).

## A tour of the repository

```
build-your-own-moe-router/
├── moe/                       ← the library
│   ├── __init__.py            ← the public API (what you import)
│   ├── config.py              ← MoEConfig: one validated source of truth
│   ├── router.py              ← the gating networks (the heart of MoE)
│   ├── losses.py              ← load-balancing aux loss + router z-loss
│   ├── experts.py             ← the expert FFNs + token dispatch
│   ├── layer.py               ← MoELayer: the drop-in FFN replacement
│   ├── utils.py               ← monitoring & diagnostics
│   └── bench.py               ← a forward-pass benchmark (dev tool)
├── tests/                     ← the test suite (unit, integration, property-based)
├── docs/
│   ├── moe_routing.md         ← the dense reference document
│   └── course/                ← you are here
├── pyproject.toml             ← packaging, ruff/mypy/pytest config
└── Makefile                   ← test / lint / format / bench / docs targets
```

The mental model for the data flow through one layer is short, and every lesson
fills in one box of it:

```
        x : [batch, seq, d_model]
              │
              ▼
        ┌───────────┐     router_probs, expert_indices, combine_weights, aux_loss
        │  router   │ ─────────────────────────────────────────────────────────►
        └───────────┘                         (Lesson 05)
              │ combine_weights, expert_indices
              ▼
        ┌───────────┐
        │ExpertBank │  dispatch each token to its top-k experts, weight & sum
        └───────────┘                         (Lesson 04)
              │ y : [batch, seq, d_model]
              ▼
        ┌───────────┐
        │  losses   │  aux load-balancing loss + z-loss   (Lesson 03)
        └───────────┘
              │
              ▼
        MoELayerOutput(output, aux_loss, z_loss, router_probs, expert_utilisation)
                                                (Lesson 06)
```

`MoEConfig` (Lesson 02) parameterises every box, and `utils` (Lesson 07) reads
the router's outputs to tell you whether training is healthy.

## Your first forward pass

Open a REPL and run:

```python
import torch
from moe import MoELayer, MoEConfig

config = MoEConfig(d_model=32, d_ff=64, num_experts=4, top_k=2)
layer = MoELayer(config)

x = torch.randn(2, 8, 32)        # [batch, seq, d_model]
out = layer(x)

print(out.output.shape)          # torch.Size([2, 8, 32]) — same shape as input
print(out.aux_loss, out.z_loss)  # two scalar regularisers to add to your task loss
print(layer.get_routing_stats()) # entropy, imbalance ratio, overflow fraction, ...
```

Three things to notice, each of which a later lesson explains in depth:

1. **The output has the same shape as the input.** That is what makes `MoELayer`
   a drop-in replacement for a dense FFN sublayer — you add it to a residual
   stream exactly as you would a normal FFN: `x = x + out.output`.
2. **It returns extra losses.** `aux_loss` and `z_loss` are not optional
   decoration; without them the router collapses (Lesson 03). You add them to
   your task loss: `loss = task_loss + out.aux_loss + out.z_loss`.
3. **It can tell you how it's doing.** `get_routing_stats()` is your dashboard.
   Learning to read it (Lesson 07) is the difference between a model that trains
   and one that silently wastes 7 of its 8 experts.

## How to get the most from the course

Type the snippets, don't just read them. Do the exercises before reading the
solutions — they are designed to make a specific idea stick. When a lesson
references a function, open the real file in `moe/` alongside it; the whole point
of this course is that the code is the curriculum.

## Common pitfalls

- **Forgetting the auxiliary losses.** If you only optimise the task loss, the
  router collapses and most experts go unused. Always add `out.aux_loss` (and
  usually `out.z_loss`) to your loss.
- **Installing without the dev extras.** `pip install -e .` works for *using* the
  library, but you need `".[dev]"` to run `make test`/`make lint`.
- **Expecting a speedup at tiny sizes.** The MoE's advantage shows up at real
  widths; at `d_model=8` the Python dispatch overhead dominates. Benchmark at
  realistic sizes (Lesson 09).

## Exercises

1. Install the package with dev extras and run `make test`. Confirm it reports
   100% coverage on `moe/`.
2. Run `make bench` and write down the reported speedup on your machine.
3. In a REPL, build a `MoEConfig` with `num_experts=8, top_k=1` and run a forward
   pass on a `[1, 4, d_model]` input. Print `out.expert_utilisation` and explain
   what its 8 numbers mean.

## Solutions

1. `pip install -e ".[dev]"` then `make test`; the coverage table should show
   `TOTAL ... 100%` and "105 passed".
2. The `make bench` table ends with a line like `MoE speedup vs equal-parameter
   dense FFN: 2.60x`. Your exact number depends on your CPU.
3. With `top_k=1`, four tokens are each sent to exactly one of the eight experts,
   so `out.expert_utilisation` is a length-8 vector of fractions summing to 1 —
   the share of routed tokens each expert received. With only 4 tokens it will be
   spiky (several zeros); that is expected at this tiny scale, not collapse.

## Key takeaways

- MoE decouples **parameter count** from **per-token compute** by routing each
  token to only a few of many experts.
- The router is the crux: selection, load balancing, stability, speed.
- `MoELayer` is a shape-preserving, drop-in FFN replacement that also returns
  auxiliary losses and routing statistics.
- The repository is organised one-concept-per-module, mirroring this course.

## Next

Continue to [Lesson 01 — MoE from First Principles](01-moe-first-principles.md),
where we derive the compute/parameter trade-off and the core MoE equation.
