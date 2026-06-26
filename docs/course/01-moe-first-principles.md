# Lesson 01 — MoE from First Principles

> A Mixture of Experts buys you many times the parameters for only a small multiple of the per-token compute — this lesson explains exactly why, and where the library you are about to learn puts each moving part.

## Learning objectives

By the end of this lesson you can:

- Explain *conditional computation* and articulate the compute-versus-parameter trade-off that makes MoE worth the trouble.
- Work the FLOP arithmetic for a dense FFN versus a top-$K$ MoE, with concrete numbers, and state the "$N\times$ parameters for $K\times$ FLOPs" slogan precisely.
- Write the core MoE equation, define every symbol, and explain why unselected experts receive *zero gradient*.
- Place the field's milestones (Jacobs → Shazeer → GShard → Switch → Expert Choice → Mixtral) in a one-line-each lineage.
- Navigate this repository's `moe` package — knowing which module owns the config, the routers, the losses, the experts, the layer, and the monitoring utilities.

## Prerequisites

- Comfortable Python and PyTorch tensors.
- A high-level grasp of transformers: a residual stream of width $d_{model}$, attention sub-layers, and feed-forward (FFN) sub-layers.
- Backprop intuition: a term that is multiplied by zero contributes zero to both the forward value and the gradient.
- No prior MoE knowledge is assumed. That is what we are here to build.

---

## 1. The problem: capacity and compute are rigidly coupled in a dense model

Here is the uncomfortable fact about scaling a transformer. Its *quality* tracks its *parameter count* — more parameters means more room to memorize and specialize. But its *cost* tracks the *activated compute per token* — the FLOPs every token must pay on the forward pass. In a dense model these two are welded together: **every parameter participates in every token's forward pass.** Want more capacity? You pay for it on every single token.

MoE breaks the weld. Instead of one FFN that all tokens traverse, it keeps $N$ parallel FFNs — the *experts* — and a tiny trainable *router* that, per token, picks a small subset to actually run. The parameters all exist (so capacity is large), but only a few execute per token (so compute stays small). The trick that makes this legal is called **conditional computation**: parts of the network switch on or off as a function of the input, rather than always executing.

That is the whole idea in one sentence. The rest of this course is about making it *work* — because, as you will see in Lesson 03, a sparse router left to its own devices reliably destroys itself.

### Why "conditional"? The switch reads the data

The router's choice is **data-dependent** (it reads the token to decide) and **sparse** (it activates only a handful of experts). Conditional computation is precisely what converts extra parameters into extra *capacity* without converting them into proportional extra *cost*.

---

## 2. The compute-versus-parameter trade-off, with real numbers

Let us make the trade-off quantitative, because the numbers are the entire argument for MoE.

### Dense FFN cost

A standard transformer FFN maps a hidden vector up to an inner width and back down. Let $d_{model}$ be the residual-stream (model) dimension and $d_{ff}$ the FFN's inner width. The two matmuls are an up-projection of shape $d_{model}\times d_{ff}$ and a down-projection of shape $d_{ff}\times d_{model}$. Counting one multiply-accumulate as two FLOPs, the per-token cost is

$$
\text{FLOPs}_{\text{dense}} \approx 2\,d_{model}\,d_{ff} \;+\; 2\,d_{ff}\,d_{model} \;=\; 4\,d_{model}\,d_{ff},
$$

where the first term is the up-projection, the second is the down-projection, and each factor of $2$ converts multiply-accumulates into raw FLOPs. With $d_{model}$ fixed, this is the familiar $O(d_{ff})$ per-token cost.

### Top-$K$ MoE cost

Now replace the single FFN with $N$ experts, each an FFN of inner width $d_{ff}$, and route each token to its top $K$ experts. Here $N$ is the number of experts and $K$ is how many a token visits. A token pays only for the $K$ experts it actually touches:

$$
\text{FLOPs}_{\text{MoE/token}} \approx K\cdot 4\,d_{model}\,d_{ff},
$$

while the FFN block's *parameter* count has grown by a factor of $N$ (there are now $N$ expert FFNs instead of one). The slogan:

> **You buy $N\times$ the parameters for $K\times$ the dense per-token FLOPs.** The activated fraction of expert parameters is $K/N$.

### A worked example

Take $d_{model}=1024$, $d_{ff}=4096$, $N=8$, $K=2$.

- Dense FFN: $4\cdot 1024\cdot 4096 = 16{,}777{,}216 \approx 1.68\times10^{7}$ FLOPs/token.
- MoE layer: holds $8\times$ those FFN parameters, yet each token activates only $K=2$ of them, costing $2\cdot 16{,}777{,}216 = 33{,}554{,}432 \approx 3.36\times10^{7}$ FLOPs/token.

So for $2\times$ the per-token compute of a single dense FFN, the model wields $8\times$ the FFN parameters, of which $2/8 = 25\%$ are activated per token. And the router itself? It adds only $d_{model}\cdot N = 1024\cdot 8 = 8192$ multiply-accumulates per token to compute the gate logits — roughly $0.05\%$ of one expert's cost. That is why we can treat routing as essentially "free" relative to the experts.

This same configuration is exactly the kind of thing you will declare in the library's `MoEConfig` — fields like `d_model`, `d_ff`, `num_experts`, and `top_k` are the literal names of these symbols.

---

## 3. The core MoE equation

With the economics settled, here is the layer's forward computation:

$$
y = \sum_{i \in \text{Top-K}} g_i(x)\cdot E_i(x),
$$

where:

- $x \in \mathbb{R}^{d_{model}}$ is the input token representation,
- $E_i(\cdot)$ is the FFN of expert $i$, a map $\mathbb{R}^{d_{model}}\to\mathbb{R}^{d_{model}}$,
- $g_i(x)\in\mathbb{R}$ is the scalar *gating weight* (combine weight) the router assigns to expert $i$ for this token,
- $\text{Top-K}$ is the set of the $K$ expert indices with the highest router logits for $x$,
- $y\in\mathbb{R}^{d_{model}}$ is the layer output.

The sum runs **only** over the selected experts. Experts outside $\text{Top-K}$ contribute nothing.

### The sparsity mask and why unselected experts get zero gradient

The selection induces a *sparsity mask*: the router computes a weight for every expert, but for experts not in $\text{Top-K}$ the effective combine weight is exactly zero — they are masked out before the sum. This has a precise and load-bearing consequence for learning.

Because the term for a non-selected expert is $g_i(x)\cdot E_i(x) = 0\cdot E_i(x)$, both the term *and its derivative* with respect to that expert's parameters $\theta_i$ vanish:

$$
\frac{\partial y}{\partial \theta_i} = g_i(x)\,\frac{\partial E_i(x)}{\partial \theta_i} = 0 \quad\text{when } g_i(x)=0.
$$

So **non-selected experts receive zero gradient on that token** — they are neither rewarded nor penalized, and they do not move. This is what makes the forward pass sparse in *compute* and the backward pass sparse in *gradient*. It is also the seed of the central pathology of MoE: an expert that is never selected is never trained, and an expert that is never trained never becomes worth selecting. We attack that feedback loop head-on in Lesson 03 with the load-balancing loss; for now, simply hold onto the mechanism.

---

## 4. A short lineage: who solved what

Each milestone fixed a problem the previous one exposed.

- **Jacobs, Jordan, Nowlan & Hinton (1991)** — "adaptive mixtures of local experts": a softmax gate plus experts, trained jointly, each specializing on a region of input space. Established the template, but dense (no sparsity at scale).
- **Shazeer et al. (2017), "Outrageously Large Neural Networks"** — made the gate genuinely sparse with *noisy top-$K$* gating and an explicit load-balancing loss, scaling to thousands of experts inside an LSTM. This is where MoE became a tool for billion-parameter models.
- **Lepikhin et al. (2021), GShard** — carried sparse MoE into the transformer *and* into distributed training: expert parallelism, all-to-all dispatch/combine, and a fixed-size per-expert *capacity*.
- **Fedus, Zoph & Shazeer (2022), Switch Transformer** — simplified routing to top-1 (one expert per token), halving communication and proving it scales to a trillion parameters. The library's auxiliary-loss convention (with the factor of $N$) is the Switch convention.
- **Zhou et al. (2022), Expert Choice** — inverted the selection: each *expert* picks its top-$C$ tokens. Perfect load balance by construction, no dropped tokens per expert — at the cost of a token possibly being chosen by zero experts.
- **Jiang et al. (2024), Mixtral 8x7B** — a widely deployed open-weight decoder LLM: 8 experts, top-2, with the selected gate weights renormalized to sum to one. Proof that MoE ships, not just publishes.

You will meet these names again as *router classes*: `TopKRouter` (Shazeer), `SwitchRouter` (Fedus), and `ExpertChoiceRouter` (Zhou).

---

## 5. A map of the library

The package's public surface is small and deliberate. The module docstring states it plainly: "The supported surface is exactly the names re-exported here. Everything else ... may change without notice." Here is what `moe/__init__.py` re-exports:

```python
from moe import (
    MoELayer, MoEConfig, MoELoss,              # the layer, its config, its losses
    TopKRouter, SwitchRouter, ExpertChoiceRouter,  # the three gating networks
    MoELayerOutput, RouterOutput, LossOutput,  # typed output containers
)
```

And here is where each concept from this lesson lives:

| Module | Public names | Role |
|---|---|---|
| `moe.config` | `MoEConfig` | One validated dataclass holding every knob ($d_{model}$, $d_{ff}$, $N$, $K$, `capacity_factor`, `alpha`, `beta`, …). The single source of truth. |
| `moe.router` | `TopKRouter`, `SwitchRouter`, `ExpertChoiceRouter`, `RouterOutput` | The gating networks — score, select, normalize — all returning a common `RouterOutput` so they are interchangeable. |
| `moe.losses` | `MoELoss`, `LossOutput` | The auxiliary load-balancing loss and the router z-loss that keep training stable (Lesson 03). |
| `moe.experts` | *(internal)* | `Expert` and `ExpertBank`: the FFNs themselves plus the dispatch/combine that sends tokens to their experts. |
| `moe.layer` | `MoELayer`, `MoELayerOutput` | The drop-in FFN replacement that wires router + experts + losses together. |
| `moe.utils` | *(internal)* | Monitoring: routing entropy, load imbalance, dead-expert detection (Lesson 07). |

The smallest end-to-end use, straight from the package docstring, ties it together:

```python
import torch
from moe import MoELayer, MoEConfig

layer = MoELayer(MoEConfig.mixtral_style(num_experts=4, d_model=32))
hidden = torch.randn(2, 8, 32)
out = layer(hidden)                  # out is a MoELayerOutput
residual = hidden + out.output       # add the layer output to the residual stream
loss = out.aux_loss + out.z_loss     # add these to your task loss
```

Notice three things, because they preview the whole course. First, the config is built by a *preset factory* — `MoEConfig.mixtral_style(...)` — that fills in Mixtral's top-2, renormalized-weights recipe for you. Second, the layer returns a `MoELayerOutput`, a typed bundle whose `.output` you add to the residual just like a dense FFN. Third, the layer hands you `aux_loss` and `z_loss` *already weighted*, so balancing the experts is as simple as adding two scalars to your task loss. Everything else in this course is detail beneath these three lines.

---

## Common pitfalls

- **Confusing parameter count with compute.** MoE multiplies parameters by $N$ but per-token FLOPs only by $K$. People who quote "8x7B" as a 56B-FLOP model are double-counting; only $K$ experts run per token.
- **Forgetting the router is nearly free.** The gate is a single $d_{model}\times N$ projection. If your mental model spends real compute on routing, you will mis-budget the layer.
- **Expecting all experts to train every step.** They do not. Zero gating weight means zero gradient. This is correct and intended — but it is exactly why balance is a problem, not a footnote.
- **Reaching past the public API.** `moe.experts` and `moe.utils` internals carry no compatibility guarantee. Build against the re-exported names only.

## Exercises

1. **FLOP recompute.** For $d_{model}=2048$, $d_{ff}=8192$, $N=16$, $K=2$, compute the dense FFN FLOPs/token, the MoE FLOPs/token, the parameter multiplier, and the activated fraction $K/N$.
2. **Router overhead.** For the same config, compute the router's per-token multiply-accumulates and express them as a percentage of *one* expert's FFN cost.
3. **Read the equation in code terms.** Without running anything, predict the shape of `out.output` for `MoEConfig.mixtral_style(num_experts=4, d_model=32)` fed `torch.randn(2, 8, 32)`. Which dimension corresponds to $d_{model}$ in the equation $y=\sum g_i(x)E_i(x)$?
4. **Lineage matching.** Match each router class — `TopKRouter`, `SwitchRouter`, `ExpertChoiceRouter` — to the paper and the one-line idea that introduced it.
5. **Run it.** Execute the smallest example from Section 5 and print `out.output.shape`, `out.aux_loss`, and `out.z_loss`. Confirm the shape matches your prediction in Exercise 3.

## Solutions

1. Dense $=4\cdot2048\cdot8192 = 67{,}108{,}864\approx6.71\times10^{7}$ FLOPs/token. MoE $=2\times$ that $=134{,}217{,}728\approx1.34\times10^{8}$ FLOPs/token. Parameter multiplier $=N=16\times$. Activated fraction $=K/N=2/16=12.5\%$.
2. Router MACs $=d_{model}\cdot N = 2048\cdot16 = 32{,}768$. One expert's FFN cost $\approx 4\,d_{model}\,d_{ff}/2 = 2\,d_{model}\,d_{ff} = 2\cdot2048\cdot8192 = 33{,}554{,}432$ MACs. Ratio $\approx 32{,}768 / 33{,}554{,}432 \approx 0.098\%$ — about a tenth of a percent. Routing is effectively free.
3. `torch.Size([2, 8, 32])` — the layer is shape-preserving. The last dimension, $32$, is $d_{model}$: each token's output $y$ is a vector in $\mathbb{R}^{d_{model}}$.
4. `TopKRouter` → Shazeer 2017, noisy top-$K$ sparse gating. `SwitchRouter` → Fedus 2022, top-1 routing with a capacity buffer. `ExpertChoiceRouter` → Zhou 2022, experts pick their top-$C$ tokens.
5. Running the Section 5 snippet prints `torch.Size([2, 8, 32])` for the shape, and two finite scalar tensors for `aux_loss` and `z_loss` (already weighted by `alpha` and `beta`). Add them to your task loss and you are training a balanced MoE.

## Key takeaways

- A dense model couples capacity to compute; MoE decouples them via **conditional computation** — a data-dependent, sparse switch over experts.
- The arithmetic is the argument: **$N\times$ parameters for $K\times$ FLOPs**, with $K/N$ of expert parameters active per token, and a router that costs almost nothing.
- The forward pass is $y=\sum_{i\in\text{Top-K}} g_i(x)E_i(x)$; a zero combine weight means a zero gradient, which is both the source of MoE's efficiency and the seed of expert collapse.
- The lineage — Jacobs → Shazeer → GShard → Switch → Expert Choice → Mixtral — maps directly onto the library's `TopKRouter`, `SwitchRouter`, and `ExpertChoiceRouter`.
- The package's public API is exactly the names in `moe/__init__.py`: `MoELayer`, `MoEConfig`, `MoELoss`, the three routers, and the typed output containers.

## Next → `02-configuration.md`

You have seen `MoEConfig.mixtral_style(...)` conjure a working config in one call. Next we open that dataclass up: every field, its valid range, the `validate()` contract that rejects bad configs, the `tokens_per_expert(...)` helper, and the preset factories (`switch_style`, `mixtral_style`, and friends) that encode each paper's recipe so you do not have to memorize it.
