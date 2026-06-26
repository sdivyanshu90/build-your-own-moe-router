# Build Your Own MoE Router — The Course

A hands-on, code-first course that teaches **Mixture of Experts (MoE) routing**
from first principles all the way to a trained mini-transformer — using the real,
production-grade library in this repository as the textbook.

Unlike a paper, every concept here is grounded in code you can read, run and
modify. By the end you will understand *why* every line of `moe/` exists and be
able to build, train, monitor and debug a sparse MoE layer yourself.

## Who this is for

You should be comfortable with Python and have a working mental model of deep
learning (tensors, autograd/backprop, and what a transformer FFN is). You do
**not** need any prior MoE knowledge — that is exactly what we build.

## How to use the course

Work through the lessons in order. Each lesson:

- opens with **learning objectives** and **prerequisites**,
- teaches the **intuition**, then the **math**, then walks the **real code**,
- always explains the **why** behind a design decision,
- ends with **common pitfalls**, **exercises with worked solutions**, and **key
  takeaways**.

Keep a Python REPL open. Every code snippet is runnable against the installed
package:

```bash
pip install -e ".[dev]"     # from the repo root
python                       # then: from moe import MoELayer, MoEConfig
```

## Syllabus

| # | Lesson | You will learn | Source it teaches |
|---|--------|----------------|-------------------|
| 00 | [Introduction & Setup](00-introduction.md) | The big picture, install, repo tour, how to navigate | — |
| 01 | [MoE from First Principles](01-moe-first-principles.md) | Conditional computation, the compute/parameter trade-off, the core MoE equation, the lineage | `moe/__init__.py` |
| 02 | [The Configuration System](02-configuration.md) | One validated source of truth, presets, fail-fast validation | `moe/config.py` |
| 03 | [Load Balancing & the Losses](03-load-balancing-and-losses.md) | Expert collapse, the auxiliary loss, the z-loss, the detach/attach trick | `moe/losses.py` |
| 04 | [The Experts & Token Dispatch](04-experts.md) | Expert FFNs, the shared kernel, naive vs batched dispatch, capacity | `moe/experts.py` |
| 05 | [The Routers (Gating Networks)](05-routing.md) | Top-K, noisy gating, Switch, Expert Choice, combine weights | `moe/router.py` |
| 06 | [Assembling the MoE Layer](06-the-moe-layer.md) | Composing router + experts + losses, the forward pass, stats | `moe/layer.py` |
| 07 | [Monitoring & Debugging](07-monitoring.md) | Entropy, imbalance, dead experts, heatmaps, JSONL logging | `moe/utils.py` |
| 08 | [Testing an MoE Layer](08-testing.md) | Invariant tests, gradient proofs, property-based testing | `tests/` |
| 09 | [Performance, Precision & Scaling](09-performance-and-scaling.md) | Benchmarking, bfloat16, expert/tensor parallelism | `moe/bench.py` |
| 10 | [Capstone: Mini MoE Transformer](10-capstone.md) | Build & train a full block end to end | the whole API |

## Companion reference

When you want the dense, paper-style reference rather than the teaching
narrative, see [`../moe_routing.md`](../moe_routing.md) — it covers the same
material in encyclopedia form with full derivations and a hyperparameter table.

## A note on conventions

This codebase makes deliberate, stated choices that the lessons will keep
reminding you of:

- The auxiliary load-balancing loss is the **canonical Switch Transformer**
  form $L_{aux} = \alpha \cdot N \cdot \sum_i f_i P_i$ (balanced → $\alpha$,
  collapsed → $\alpha N$).
- The router z-loss uses `torch.logsumexp` for overflow-free stability.
- Combine weights default to a **masked full softmax** (per-token sum $\le 1$);
  `normalize_router_weights=True` switches to the Mixtral renormalised form.
- Routing probabilities and losses are computed in **float32** even under
  bfloat16 autocast.

Ready? Start with [Lesson 00 — Introduction & Setup](00-introduction.md).
