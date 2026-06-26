# Lesson 10 — Capstone: Build & Train a Mini MoE Transformer

> Assemble `MoELayer` into a real transformer-style block, train it on a synthetic task, and watch the routing-health metrics from every previous lesson come alive in one runnable script.

## Learning objectives

By the end of this lesson you will be able to:

- Wire `MoELayer` into a pre-norm transformer block as a drop-in FFN sub-layer, with the residual connection and the `out.aux_loss + out.z_loss` regularisers applied correctly.
- Stack those blocks into a small model and train it on CPU with a standard PyTorch loop in seconds.
- Read `get_routing_stats()` every step and interpret entropy, imbalance, overflow and dead-expert counts as a live diagnosis of routing health.
- Demonstrate expert collapse experimentally by setting `alpha = 0`, and demonstrate the cure by turning the load-balancing loss back on.
- Swap router presets (`switch_transformer` vs `mixtral_style`) and toggle bfloat16 autocast, observing the effect on the same metrics.
- Persist a training run with `export_routing_stats` and inspect it with `visualise_routing`.

## Prerequisites

- The whole course so far: you know what a router emits (Lesson 01), how `MoEConfig` and its presets are built (Lesson 02), and why the auxiliary load-balancing loss and the z-loss exist (Lesson 03), plus the later material on routers, dispatch/capacity, monitoring and mixed precision.
- Comfort with a basic PyTorch training loop: `optimiser.zero_grad()`, `loss.backward()`, `optimiser.step()`.
- The public API only: `from moe import MoELayer, MoEConfig, MoELoss, TopKRouter, SwitchRouter, ExpertChoiceRouter`, plus the two helpers `visualise_routing` and `export_routing_stats` from `moe.utils`.

You do **not** need a GPU. Everything below runs on a laptop CPU.

---

## The project (full runnable code + explanation)

We are going to build `mini_moe.py`. It has four parts: a config builder, a transformer block whose FFN is an `MoELayer`, a two-block model, and a training loop that logs routing health. Paste the blocks below into one file in this order and it runs end to end.

### Step 1 — config and the transformer block

The single most important design fact about `MoELayer` is that it is a **drop-in FFN replacement**. A dense transformer block does `x = x + attn(norm(x))` then `x = x + ffn(norm(x))`. We keep the attention exactly as-is and replace only the second sub-layer. `MoELayer.forward` returns an `MoELayerOutput` named tuple — we add `out.output` to the residual stream and stash `out.aux_loss`/`out.z_loss` for the optimiser.

```python
"""mini_moe.py — a tiny MoE transformer you can train on CPU."""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from moe import MoEConfig, MoELayer
from moe.utils import export_routing_stats, visualise_routing


def make_config(preset: str, num_experts: int = 8, d_model: int = 64) -> MoEConfig:
    """Build a config from a named preset. Try switching the preset later."""
    if preset == "switch_transformer":
        return MoEConfig.switch_transformer(num_experts=num_experts, d_model=d_model)
    if preset == "mixtral_style":
        return MoEConfig.mixtral_style(num_experts=num_experts, d_model=d_model)
    raise ValueError(f"unknown preset {preset!r}")


class MoEBlock(nn.Module):
    """A pre-norm transformer block: self-attention, then an MoE FFN."""

    def __init__(self, config: MoEConfig, n_heads: int = 4) -> None:
        super().__init__()
        d_model = config.d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(config)            # <-- the FFN is a sparse MoE layer

    def forward(self, x):
        # 1. Attention sub-layer (a "mixer" across the sequence) with residual.
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        # 2. MoE FFN sub-layer with residual; keep the output bundle for losses.
        moe_out = self.moe(self.norm2(x))
        x = x + moe_out.output
        return x, moe_out
```

Notes:

- The attention is a genuine multi-head self-attention here; if you wanted to stay even simpler you could replace `self.attn` with a `nn.Linear(d_model, d_model)` token mixer — the MoE story is identical. `n_heads` must divide `d_model` (4 divides 64).
- `forward` returns **two** things: the new residual stream `x` and the `moe_out` bundle. The bundle is how losses and stats escape the block.

### Step 2 — the model

```python
class MiniMoETransformer(nn.Module):
    """Input projection -> N MoE blocks -> output head. A regression model."""

    def __init__(self, config: MoEConfig, n_blocks: int = 2) -> None:
        super().__init__()
        d_model = config.d_model
        self.in_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(MoEBlock(config) for _ in range(n_blocks))
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.in_proj(x)
        moe_outputs = []
        for block in self.blocks:
            x, moe_out = block(x)
            moe_outputs.append(moe_out)        # collect from every block
        return self.head(self.norm_out(x)), moe_outputs
```

Two blocks is enough to reproduce the proven two-layer training pattern from `tests/test_integration.py`, where the headline test trains exactly this shape and asserts the loss falls while routing entropy rises. `forward` returns the prediction **and the list of per-block MoE outputs** so the loop can sum every block's auxiliary losses.

### Step 3 — synthetic data and the training loop

The data is deliberately **low-diversity**: every token is a shared base direction plus a little noise (`INPUT_DIVERSITY = 0.3`). This is the exact trick `tests/test_integration.py` uses, and it matters: when all tokens look alike, the router has no reason to spread them out, so the rich-get-richer pressure that causes collapse is strong and *visible*. On a high-diversity dataset collapse is slower and the demo is muddier.

```python
INPUT_DIVERSITY = 0.3


def make_batch(d_model: int, batch: int = 8, seq: int = 16, seed: int = 123):
    """A fixed, low-diversity regression batch (shared base + small noise)."""
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(1, 1, d_model, generator=gen)
    x = base + INPUT_DIVERSITY * torch.randn(batch, seq, d_model, generator=gen)
    target = torch.randn(batch, seq, d_model, generator=gen)
    return x, target


def train(preset="switch_transformer", *, alpha=None, steps=150, lr=1e-2,
          num_experts=8, d_model=64, autocast=False, seed=0, log_every=30):
    """Train the mini MoE transformer and return (model, history, config)."""
    torch.manual_seed(seed)
    config = make_config(preset, num_experts, d_model)
    if alpha is not None:                      # override the load-balancing weight
        config.alpha = alpha
        config.validate()
    model = MiniMoETransformer(config, n_blocks=2)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    x, target = make_batch(d_model)

    history = []
    for step in range(steps):
        ctx = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else nullcontext()
        with ctx:
            prediction, moe_outputs = model(x)
            task = ((prediction.float() - target) ** 2).mean()   # task loss

        # The MoE regularisers from EVERY block, added to the task loss.
        aux = sum(o.aux_loss for o in moe_outputs)
        z = sum(o.z_loss for o in moe_outputs)
        loss = task + aux + z

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Read routing health from the first block's MoE layer.
        stats = model.blocks[0].moe.get_routing_stats()
        history.append({
            "step": step,
            "task_loss": float(task),
            "total_loss": float(loss),
            "entropy_mean": stats["entropy_mean"],
            "imbalance_ratio": stats["imbalance_ratio"],
            "overflow_fraction": stats["overflow_fraction"],
            "dead_experts": stats["dead_experts"],
        })
        if step % log_every == 0 or step == steps - 1:
            r = history[-1]
            print(f"step {step:3d} | task {r['task_loss']:.3f} | "
                  f"H {r['entropy_mean']:.3f} | imbalance {r['imbalance_ratio']:.2f} | "
                  f"overflow {r['overflow_fraction']:.2f} | dead {r['dead_experts']}")
    return model, history, config
```

Three things deserve a close read:

1. **`loss = task + aux + z`.** `MoELayer` returns *already-weighted* losses (`alpha * raw` and `beta * raw`), so you literally add them — no extra coefficient. We sum across both blocks; each block has its own router and its own balancing pressure.
2. **`get_routing_stats()` is a snapshot of the last forward pass.** Call it after `step()` (or any time after a forward) to get `entropy_mean`, `imbalance_ratio`, `overflow_fraction`, `dead_experts` and the per-expert `expert_utilisation`. We log the first block; in a real run you would log all of them.
3. **Autocast is optional and self-documenting.** Under bfloat16 the expert matmuls run in low precision while the router probabilities and the auxiliary losses stay float32 — that is why we call `.float()` on the prediction before the MSE. The `nullcontext()` branch keeps full precision.

### Step 4 — run it

```python
def summarise(history, window=20):
    """Average the first and last `window` steps for a quick before/after."""
    def avg(key, rows):
        return sum(r[key] for r in rows) / len(rows)
    first, last = history[:window], history[-window:]
    print(f"  imbalance : {avg('imbalance_ratio', first):.2f} -> "
          f"{avg('imbalance_ratio', last):.2f}")
    print(f"  entropy   : {avg('entropy_mean', first):.3f} -> "
          f"{avg('entropy_mean', last):.3f}")
    print(f"  task loss : {avg('task_loss', first):.3f} -> "
          f"{avg('task_loss', last):.3f}")


if __name__ == "__main__":
    print("== switch_transformer, default alpha ==")
    model, history, config = train("switch_transformer", steps=150)
    summarise(history)

    # Persist the run and render a routing heatmap of the trained model.
    export_routing_stats(history, "routing_default.jsonl")
    model.eval()
    with torch.no_grad():
        _, outs = model(make_batch(config.d_model)[0])
    fig = visualise_routing(outs[0].router_probs[:16])   # first 16 tokens
    fig.savefig("routing_heatmap.png")
    print("wrote routing_default.jsonl and routing_heatmap.png")
```

Run `python mini_moe.py`. On a CPU the 150-step run finishes in a few seconds and prints something close to:

```
step   0 | task 1.305 | H 1.993 | imbalance 5.69 | overflow 0.00 | dead [1, 2, 4, 6, 7]
step 149 | task 0.115 | H 2.019 | imbalance 1.56 | overflow 0.00 | dead []
  imbalance : 4.21 -> 1.70
  entropy   : 2.01 -> 2.02
  task loss : 1.10 -> 0.20
```

The task loss falls by ~5x, imbalance falls from "five experts idle" to "well balanced", and by the end **no expert is dead**. That is a healthy MoE training run. (Your exact numbers will vary a little with the PyTorch version; the *shape* of the story is what matters. The imbalance is noisy step to step because the batch is tiny — 128 tokens over 8 experts — so read the windowed average, not single steps.)

---

## Guided experiments

Each experiment is a one-line change to the `train(...)` call. Run them and read the metrics; the point is to *feel* the dynamics from Lesson 03, not just to read about them.

**1. Default `alpha` keeps load balanced.** This is the baseline above. The load-balancing loss is on (`alpha = 1e-2`), and imbalance settles well below 2 with no dead experts.

**2. `alpha = 0` collapses routing.** Turn the balancing loss off and train longer:

```python
model, history, config = train("switch_transformer", alpha=0.0, steps=200)
summarise(history)
```

Now watch imbalance *climb and stay pinned at the maximum*. With 8 experts and top-1 routing the worst case is 8.0 (every token to one expert), and that is roughly where it lands:

```
  imbalance : 6.30 -> 7.98
  entropy   : 1.90 -> 1.50
  task loss : 1.10 -> 0.16
```

Seven of the eight experts go dead. Crucially the **task loss still falls** — the model cheats by collapsing into a near-dense network using one or two experts, exactly the failure Lesson 03 warned about. The aux loss is what stands between you and this.

**3. Switch the router type.** Compare `switch_transformer` (top-1) with `mixtral_style` (top-2, SwiGLU, renormalised gates, no token dropping):

```python
train("mixtral_style", steps=150)
```

Top-2 routing sends each token to two experts, so the imbalance and entropy curves look different and the `overflow_fraction` stays at 0.0 because Mixtral-style configs do not drop tokens. This is the cleanest way to feel how a config preset changes routing behaviour without touching the model code.

**4. Toggle bfloat16 autocast.** Add `autocast=True`:

```python
train("switch_transformer", steps=60, autocast=True)
```

The metrics track the full-precision run closely. That is the payoff of the float32-softmax / float32-loss design: the expensive expert matmuls run in bf16, but routing decisions and the auxiliary losses stay numerically stable.

---

## Common pitfalls

- **Forgetting the residual.** `MoELayer` returns the *sub-layer output*, not the updated stream. You must write `x = x + moe_out.output`. Returning `moe_out.output` directly throws away the residual path and training stalls.
- **Dropping the auxiliary losses.** If you train on `task` alone, you have silently set `alpha = beta = 0` and you will reproduce Experiment 2 by accident. Always add `out.aux_loss + out.z_loss` — for *every* block.
- **Double-weighting.** The returned losses are already multiplied by `alpha`/`beta`. Do not multiply again.
- **Calling `get_routing_stats()` before a forward pass.** It reads cached state from the last forward; before the first one it returns `{}`. Call it after the forward (we call it after `step()`).
- **`n_heads` not dividing `d_model`.** `nn.MultiheadAttention` requires `d_model % n_heads == 0`. If you shrink `d_model`, adjust `n_heads`.
- **Mixing dtypes in the loss.** Under autocast, cast the prediction to float32 before comparing to a float32 target (`prediction.float()`), or PyTorch will complain.

---

## Exercises / stretch goals

1. **Log all blocks.** Right now we only record `blocks[0]`. Collect `get_routing_stats()` from every block and confirm the deeper block balances differently.
2. **Plot the traces.** Read `routing_default.jsonl` back (it is one JSON object per line) and plot `imbalance_ratio` and `entropy_mean` over steps for the default vs `alpha=0` runs on the same axes.
3. **Sweep `alpha`.** Train at `alpha in [0, 1e-3, 1e-2, 1e-1]` and plot final imbalance vs `alpha`. Find the knee where balancing kicks in.
4. **Add an `ExpertChoiceRouter` config.** Build a third preset with `router_type="expert_choice"` and compare its overflow/imbalance behaviour. (`from moe import ExpertChoiceRouter` if you want to inspect the router in isolation; inside the model the layer builds it from `config.router_type`.)
5. **Real attention, real data.** Replace the synthetic batch with a token-classification task (embed integer tokens, predict the next token on a toy sequence), keep the MoE FFN, and confirm the metrics still behave.
6. **Capacity stress test.** Lower `capacity_factor` toward 1.0 and raise the batch size until `overflow_fraction` goes non-zero, then watch how dropped tokens interact with imbalance.

## Solutions (or hints)

- **(1)** Append a list of dicts per block: `stats = [b.moe.get_routing_stats() for b in model.blocks]`. Index into it when logging.
- **(2)** `import json`; `rows = [json.loads(line) for line in open("routing_default.jsonl")]`. Then `import matplotlib.pyplot as plt` and plot `[r["step"] for r in rows]` against `[r["imbalance_ratio"] for r in rows]`.
- **(3)** Wrap the `train(...)` call in a loop over alphas and store `summarise`-style final averages; the curve drops sharply between `0` and `1e-2`.
- **(4)** Add a branch to `make_config` returning `MoEConfig(d_model=d_model, d_ff=d_model*4, num_experts=num_experts, top_k=2, router_type="expert_choice")`. Expert-choice routing balances by construction, so imbalance starts low.
- **(5)** Swap `in_proj` for `nn.Embedding(vocab, d_model)`, change the head to `nn.Linear(d_model, vocab)`, and use cross-entropy as the task loss. The MoE plumbing is unchanged.
- **(6)** Use `MoEConfig.switch_transformer(...)` then set `config.capacity_factor = 1.0` and grow `seq`; `overflow_fraction` reports the dropped fraction directly.

## Key takeaways

- `MoELayer` is a true drop-in FFN: `x = x + moe_out.output` for the residual, `loss = task + aux + z` for the objective, one bundle per block.
- The auxiliary load-balancing loss is not optional decoration — turning it off (`alpha = 0`) collapses routing to a near-dense model while the task loss happily keeps falling, hiding the damage.
- `get_routing_stats()` is your dashboard: rising entropy and imbalance near 1 with no dead experts means healthy specialisation; pinned-high imbalance and a growing dead-expert list means collapse.
- Preset and precision are config-level knobs (`switch_transformer` vs `mixtral_style`, bf16 autocast) that you change without touching the model.
- `export_routing_stats` + `visualise_routing` turn a training run into an artefact you can inspect offline.

## Course wrap-up

You now have the full picture: from what a router emits (Lesson 01), through configuration and presets (Lesson 02), the load-balancing and z-losses that keep training honest (Lesson 03), the routers, experts, dispatch, capacity, monitoring and mixed-precision material in between, and finally — here — assembling all of it into a model that trains and reports its own health.

Where to go next:

- **The big picture and quickstart** live in the project [`README.md`](../../README.md): the public API surface, the preset table, and the one-screen usage example.
- **The deep reference**, [`../moe_routing.md`](../moe_routing.md), states the formulas this code implements (the `α·N·Σ fᵢ·Pᵢ` aux loss, the `logsumexp` z-loss, capacity and dispatch) — read it alongside `moe/losses.py` and `moe/router.py` when you want the math behind a metric.
- **Real-world scaling** is the natural sequel: swap the synthetic batch for a real dataset and tokenizer, replace the toy attention with a production attention implementation, and when you outgrow one device, move to expert-parallel / distributed dispatch where each expert lives on a different GPU. The single-device contracts you learned here — residual, weighted losses, routing stats — carry over unchanged.

You have built and trained a mini MoE transformer from the same public API a production model would use. That is the whole course in one file.
