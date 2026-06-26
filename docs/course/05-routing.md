# Lesson 05 — The Routers (Gating Networks)

> The router is the small neural network that decides *which experts each token visits* — and getting that decision right (stable, balanced, differentiable) is the whole game in a Mixture of Experts layer.

## Learning objectives

By the end of this lesson you will be able to:

- Explain the `RouterOutput` contract — every field, every shape — and why all three routers return the same thing.
- Walk through `_routing_logits` line by line: input jitter, the gating projection, `_z_score_normalise`, and noisy top-K gating.
- Justify *why* `w_gate` has **no bias**, *why* the noise scale is wrapped in `softplus`, and *why* noise helps load balancing.
- Distinguish the two combine-weight conventions in `_combine_weights` (masked full softmax vs. Mixtral renormalisation) and predict the per-token sum each produces.
- Trace how `SwitchRouter` enforces a capacity buffer with a `cumsum` FIFO and how dropped tokens become *observable*.
- Describe how `ExpertChoiceRouter` inverts the routing question and why that removes dropped-token starvation.
- Use `build_router` to swap routers without touching the layer.

## Prerequisites

- Comfortable Python and PyTorch tensor ops (`softmax`, `topk`, `argmax`, `scatter_`, `gather`, `cumsum`, broadcasting).
- The earlier lessons: you know what `MoEConfig` holds (Lesson on config) and what the auxiliary load-balancing loss computes (Lesson on losses). We will lean on both.
- A mental picture of an MoE layer: a router scores `num_experts` experts per token, a few experts run, their outputs are combined by weight. This lesson is *only* the scoring/decision half. The expert FFNs and the dispatch come next.

Open `moe/router.py` beside this lesson. Everything below quotes that exact file.

---

## 1. One contract for three routers: `RouterOutput`

The single most important design decision in `router.py` is at the very top: all three routers — `TopKRouter`, `SwitchRouter`, `ExpertChoiceRouter` — return the **same** `NamedTuple`. That is what lets `MoELayer` treat the router as a black box.

```python
class RouterOutput(NamedTuple):
    dispatch_weights: Tensor   # [num_tokens, num_experts, 1]
    combine_weights: Tensor    # [num_tokens, num_experts, 1]
    expert_indices: Tensor     # [num_tokens, top_k]
    router_logits: Tensor      # [num_tokens, num_experts]
    aux_loss: Tensor           # scalar
    router_probs: Tensor       # [num_tokens, num_experts]
```

Let `N = num_tokens` (the flattened `batch * seq`) and `E = num_experts`. Read each field as a job:

- **`dispatch_weights` `[N, E, 1]`** — how strongly each token is *sent* to each expert. Non-zero only for selected experts.
- **`combine_weights` `[N, E, 1]`** — how strongly each expert's output is *mixed back* into the token. In these routers it is **identical** to `dispatch_weights`; they are kept as separate fields to mirror GShard's dispatch/combine split, where they can differ. The trailing `1` is a singleton broadcast axis so the layer can multiply it against `[N, E, d_model]` expert outputs.
- **`expert_indices` `[N, top_k]`** — the integer ids of the selected experts per token. The dispatcher uses these to gather tokens; the aux loss uses them to *count* how many tokens each expert got.
- **`router_logits` `[N, E]`** — the pre-softmax gating scores **actually used for routing** (after jitter/normalisation/noise). The z-loss squares these, so the router must hand back the exact values it routed on.
- **`aux_loss`** — the scalar, already `alpha`-weighted, load-balancing loss for this batch.
- **`router_probs` `[N, E]`** — the *full* softmax distribution over all experts (rows sum to 1). Used for monitoring and as the differentiable `P_i` in the aux loss.

Why two separate logit/prob fields? Because routing decisions are made on `router_logits`, but the load-balancing signal needs a proper probability distribution (`router_probs`). Keeping both means downstream code never has to recompute a softmax or guess which tensor was used.

---

## 2. `TopKRouter`: the reference router

### 2.1 The gating head has no bias — on purpose

```python
self.w_gate = nn.Linear(config.d_model, config.num_experts, bias=False)
```

The class docstring spells out the reasoning, and it is worth internalising because it is a real piece of MoE folklore:

1. **A bias is a token-independent preference.** A per-expert bias $b_i$ adds the same amount to expert $i$'s logit for *every* token. That is exactly a thumb on the scale of the token partition — it fights the load-balancing loss instead of helping it.
2. **It is redundant anyway.** Softmax is invariant to a constant added to *all* logits of a token, but a *per-expert* bias is constant across tokens, not across experts, so it does shift the partition — in the wrong direction. The clean choice is to make routing a pure function of the input: $\text{logits} = W_g\,x$, nothing added.

So the gating projection is simply

$$\ell = W_g\, x, \qquad W_g \in \mathbb{R}^{E \times d}, \quad x \in \mathbb{R}^{d}$$

where $\ell \in \mathbb{R}^{E}$ is the per-token logit vector, $d = $ `d_model`, $E = $ `num_experts`.

### 2.2 `_routing_logits` step by step

This method builds the logits we route on. It applies four things *in order*, and the ordering matters.

```python
def _routing_logits(self, x_flat: Tensor) -> Tensor:
    # 1) Switch-style multiplicative input jitter — TRAINING ONLY
    if self.training and self.config.jitter_noise > 0.0:
        jitter = self.config.jitter_noise
        scale = torch.empty_like(x_flat).uniform_(1.0 - jitter, 1.0 + jitter)
        x_flat = x_flat * scale

    # 2) gating projection (no bias)
    logits: Tensor = self.w_gate(x_flat)

    # 3) optional z-score normalisation
    if self.config.router_z_score_norm:
        logits = self._z_score_normalise(logits)

    # 4) noisy top-K gating — TRAINING ONLY
    if self.training and self.w_noise is not None:
        noise_std = F.softplus(self.w_noise(x_flat))
        noise = noise_std * torch.randn_like(logits)
        logits = logits + noise

    return logits
```

**Step 1 — input jitter (training only).** Each input element is multiplied by an independent uniform factor in $[1-j,\,1+j]$ where $j =$ `jitter_noise`. This is the Switch Transformer trick: a tiny multiplicative perturbation that breaks ties between near-equal experts and discourages the router from becoming brittle. The guard `self.training` means evaluation is deterministic — you never want test-time predictions to wobble because of jitter. The scale tensor is built with `torch.empty_like(x_flat)`, so it inherits `x_flat`'s device and dtype automatically.

**Step 2 — the projection.** `self.w_gate(x_flat)` maps `[N, d_model] → [N, E]`. (The type annotation `logits: Tensor` is only there because `nn.Module.__call__` is typed to return `Any`.)

**Step 3 — z-score normalisation (optional).** Covered in 2.3.

**Step 4 — noisy top-K gating (training only).** This is Shazeer 2017's idea:

$$\ell' = \ell + \operatorname{softplus}(W_n\, x)\odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

where $W_n =$ `w_noise`, $\varepsilon =$ `torch.randn_like(logits)`, and $\odot$ is elementwise. Two design choices deserve attention:

- **Why `softplus`?** The noise *standard deviation* must be strictly positive — a negative std is meaningless and would flip the sign of the perturbation in a way the network could not control. `softplus(z) = \log(1+e^{z})` maps any real number to a positive value, smoothly (it is differentiable everywhere, unlike `relu`), so the network can *learn* per-expert, per-token noise magnitudes via gradient descent. At init, `w_noise.weight` is set with `std = noise_std_init * 1e-2`, so noise starts modest and only grows where it helps.
- **Why add noise at all?** Top-K routing is a hard, discrete decision. Early in training the router is near-random; if it happened to slightly prefer expert 3, *every* token might go to expert 3, that expert improves, the preference compounds, and the rest of the experts starve — **expert collapse**. Injecting input-dependent Gaussian noise lets a token that was the runner-up for an expert occasionally win, so under-used experts still receive gradient and the router *explores*. Noise is a load-balancing aid; that is why it is training-only.

> Note the asymmetry: jitter and noise are gated by `self.training`, but the z-score branch is **not** — it runs in both modes (with different statistics). Keep that distinction; it is a common quiz trap.

### 2.3 `_z_score_normalise`: taming logit scale without a z-loss

```python
def _z_score_normalise(self, logits: Tensor) -> Tensor:
    running_std = cast(Tensor, self.running_logit_std)
    if self.training:
        batch_std = logits.detach().float().std().clamp_min(self.config.eps)
        running_std.mul_(_ZSCORE_MOMENTUM).add_(batch_std * (1.0 - _ZSCORE_MOMENTUM))
        denom = batch_std
    else:
        denom = running_std.clamp_min(self.config.eps)
    return logits / denom.to(logits.dtype)
```

If logits drift to large magnitudes, the softmax saturates (one expert gets ~all the mass) and gradients vanish. One fix is the **z-loss** (penalise large logits; see the losses lesson). This is an *alternative*: divide the logits by their standard deviation before the softmax, so their spread stays $\mathcal{O}(1)$.

The subtlety is the BatchNorm-style train/eval split:

- **Training:** use the *current batch* std (`batch_std`), and update a persistent running estimate via an exponential moving average with momentum `_ZSCORE_MOMENTUM = 0.99`. The std is computed under `.detach().float()` so no gradient flows through the statistic and the reduction is numerically stable under mixed precision.
- **Eval:** use the stored `running_logit_std` buffer — a fixed, batch-independent statistic, so evaluation is deterministic and does not depend on how the eval batch happens to be composed.

The `clamp_min(self.config.eps)` guards against division by zero on a degenerate (constant-logit) batch.

### 2.4 Selecting experts and building combine weights

Back in `forward`:

```python
router_probs = F.softmax(routing_logits, dim=-1)          # [N, E], rows sum to 1
k = min(self.top_k, self.num_experts)
expert_indices = routing_logits.topk(k, dim=-1).indices    # [N, k]
combine = self._combine_weights(routing_logits, router_probs, expert_indices)
weights = combine.unsqueeze(-1)                            # [N, E, 1]
```

Selection is `topk` on the **logits** (not the probs — they give the same ranking, but logits avoid a needless softmax dependency). `k` is clamped to `num_experts` defensively.

Now the heart of the convention question, `_combine_weights`:

```python
topk_mask = torch.zeros_like(router_probs, dtype=torch.bool)
topk_mask.scatter_(-1, expert_indices, True)              # True at selected experts

if self.config.normalize_router_weights:
    neg_inf = torch.finfo(routing_logits.dtype).min
    masked = routing_logits.masked_fill(~topk_mask, neg_inf)
    return F.softmax(masked, dim=-1)                       # renormalised: sums to 1
return router_probs * topk_mask.to(router_probs.dtype)    # masked full softmax: sum <= 1
```

There are **two conventions**, and they are genuinely different models:

- **Default — masked full softmax (Switch convention).** Take the full softmax over *all* experts, then zero out the non-selected ones. The surviving weights are the *original* probabilities, so per token they sum to **$\le 1$** (the dropped experts took some mass with them). Formally, for selected set $S$ of token $t$:

  $$g_{t,i} = p_{t,i}\cdot \mathbf{1}[i \in S], \qquad \textstyle\sum_i g_{t,i} = \sum_{i\in S} p_{t,i} \le 1.$$

  This preserves the router's *confidence*: a token routed to two experts it was unsure about contributes less to the output, leaning more on the residual.

- **`normalize_router_weights = True` — Mixtral convention.** Set the non-top-K logits to $-\infty$ (numerically, `torch.finfo(dtype).min`) and softmax *again*, so the mass renormalises over exactly the chosen experts and sums to **1**:

  $$g_{t,i} = \frac{e^{\ell_{t,i}}}{\sum_{j\in S} e^{\ell_{t,j}}}\cdot \mathbf{1}[i \in S].$$

  Every token gets a full unit of expert output regardless of confidence — what Mixtral does, since it never drops tokens.

Using `finfo(dtype).min` rather than literal `-inf` is a deliberate **large-logit safety** choice: a true `-inf` can produce `NaN` if an entire row were masked, whereas the finite minimum keeps the softmax well-defined.

### 2.5 The auxiliary loss, computed in the router

```python
aux_loss = auxiliary_load_balancing_loss(
    router_probs, expert_indices, self.num_experts, alpha=self.config.alpha
)
```

The router computes its own load-balancing loss and ships it in `RouterOutput.aux_loss`. From the losses lesson, recall

$$L_{\text{aux}} = \alpha\, N \sum_{i=1}^{N} f_i\, P_i,$$

where $f_i$ is the *detached* fraction of dispatch slots landing on expert $i$ (counted from `expert_indices`), and $P_i$ is the *differentiable* mean softmax probability of expert $i$ (from `router_probs`). The product pushes down the probability mass of over-used experts without ever differentiating through the discrete `topk`. The router supplies *both* ingredients — counts and probabilities — which is exactly why `RouterOutput` carries `expert_indices` and `router_probs` side by side.

---

## 3. `SwitchRouter`: top-1 with a capacity buffer

`SwitchRouter` **subclasses** `TopKRouter`, so it inherits `_routing_logits`, `_z_score_normalise`, and the noise machinery for free. It only overrides `forward`, and it *requires* `top_k == 1`:

```python
def __init__(self, config: MoEConfig) -> None:
    if config.top_k != 1:
        raise ValueError(f"SwitchRouter requires top_k == 1, got {config.top_k}. ...")
    super().__init__(config)
```

**Why is top-1 stable?** With one expert per token there is no per-token weight renormalisation, no interaction between co-selected experts, and the dispatch is trivially simple — each token has exactly one destination. The price is that one expert can be swamped, so Switch adds an explicit **capacity buffer**.

```python
expert_idx = routing_logits.argmax(dim=-1)                       # [N], top-1 via argmax
gate = router_probs.gather(-1, expert_idx.unsqueeze(-1)).squeeze(-1)  # [N], chosen prob

one_hot = F.one_hot(expert_idx, self.num_experts).to(torch.long)  # [N, E]
position = one_hot.cumsum(dim=0) - 1                               # [N, E], FIFO rank
token_pos = position.gather(-1, expert_idx.unsqueeze(-1)).squeeze(-1)  # [N]
capacity = self.config.capacity(num_tokens)
keep = token_pos < capacity                                       # [N]
gate = gate * keep.to(gate.dtype)                                 # drop = zero the weight
```

Selection is `argmax` (cheaper than `topk` for $k=1$). The clever part is the **FIFO capacity accounting**:

- `one_hot` marks each token's chosen expert.
- `cumsum(dim=0) - 1` walks *down the token axis* accumulating a running count per expert. So the value at row $t$, column $e$ is "how many tokens up to and including $t$ chose expert $e$, minus one." `token_pos` reads off that count for the token's *own* expert — i.e. **this token is the `token_pos`-th token assigned to its expert** (0-indexed, in token order).
- `capacity = config.capacity(num_tokens)` is $\lceil \text{capacity\_factor} \cdot \text{top\_k} \cdot N / E \rceil$ (clamped), the buffer size each expert gets. With `drop_tokens = False`, capacity is the full token count and nothing is ever dropped.
- `keep = token_pos < capacity` keeps the *first* `capacity` tokens per expert (FIFO) and marks the overflow for dropping. Dropping is done by **zeroing the combine weight** (`gate * keep`), not by erasing the token. A dropped token contributes nothing from the experts and survives only through the residual connection added later in the layer.

Finally the per-token gate is scattered into a dense `[N, E]` matrix:

```python
combine = torch.zeros_like(router_probs)
combine.scatter_(-1, expert_idx.unsqueeze(-1), gate.unsqueeze(-1))
weights = combine.unsqueeze(-1)
expert_indices = expert_idx.unsqueeze(-1)                         # [N, 1]
```

**Drops are observable.** Because a dropped token has an all-zero row, you can count drops directly, exactly as the docstring states:

```python
num_dropped = (combine_weights.squeeze(-1).sum(-1) == 0).sum()
```

That is a metric you should log during training: a high drop rate means your `capacity_factor` is too small or your router is too imbalanced.

---

## 4. `ExpertChoiceRouter`: invert the question

`TopKRouter` and `SwitchRouter` ask *"which experts does this token want?"* Expert Choice (Zhou 2022) flips it to *"which tokens does this expert want?"* It also subclasses `TopKRouter` and overrides only `forward`.

```python
router_probs = F.softmax(routing_logits, dim=-1)        # [N, E] affinity matrix
capacity = min(self.config.capacity(num_tokens), num_tokens)

# Each expert keeps its top-C tokens — topk along the TOKEN axis (dim=0):
topk_scores, topk_tokens = router_probs.topk(capacity, dim=0)   # [C, E]

combine = torch.zeros_like(router_probs)
expert_axis = torch.arange(self.num_experts, device=x_flat.device)
combine[topk_tokens, expert_axis.unsqueeze(0).expand_as(topk_tokens)] = topk_scores
weights = combine.unsqueeze(-1)
```

The decisive line is `router_probs.topk(capacity, dim=0)`: `dim=0` is the **token** axis, so for each of the $E$ columns (experts) we select the $C$ highest-affinity tokens. The scatter writes those affinities into a dense `[N, E]` combine matrix at the chosen `(token, expert)` coordinates.

**Why this matters.** Every expert is now *exactly full* ($C$ tokens each), so there is **no dropped-token starvation** — no expert is ever idle, no expert is ever swamped. Load balance is structural, not learned. The trade-off is on the token side: a popular token may be picked by *many* experts (great — it gets lots of compute), while an unpopular token may be picked by **none** (it then relies entirely on the residual). Gradients flow only through the `(token, expert)` pairs that were actually selected.

One interface detail to internalise:

```python
# Interface-only per-token indices:
k = min(self.top_k, self.num_experts)
expert_indices = router_probs.topk(k, dim=-1).indices   # [N, k]
```

Here `expert_indices` is computed token-side (`dim=-1`) **only to satisfy the common `RouterOutput` contract** and to give the aux loss a per-token counter. The *actual* dispatch is governed entirely by `combine_weights` (the expert-side top-C selection). Do not confuse the two: in Expert Choice, `expert_indices` is informational, `combine_weights` is authoritative.

---

## 5. `build_router`: the layer stays router-agnostic

```python
def build_router(config: MoEConfig) -> TopKRouter:
    if config.router_type == "topk":
        return TopKRouter(config)
    if config.router_type == "switch":
        return SwitchRouter(config)
    if config.router_type == "expert_choice":
        return ExpertChoiceRouter(config)
    raise ValueError(f"Unknown router_type {config.router_type!r}; expected one of ...")
```

A single factory maps the string `config.router_type` to a class. Because all three return `RouterOutput` and the return type is annotated `TopKRouter` (the common base), `MoELayer` does this and nothing more:

```python
self.router = build_router(config)        # in __init__
...
router_out = self.router(x_flat)          # in forward
# then it reads router_out.dispatch_weights and router_out.expert_indices
```

The layer never branches on router type. Swapping Switch for Expert Choice is a one-field config change — the engine of the whole library's modularity.

---

## 6. Numerical stability notes

Routers are where mixed-precision training quietly goes wrong, so two safeguards are baked in:

- **Softmax stays float32 under autocast.** `_routing_logits`'s docstring is explicit: logits are *not* force-cast to float32 there, because under `torch.autocast` the downstream `F.softmax` is on the fp32 autocast policy and runs in float32 *automatically*. So routing probabilities are computed in full precision (stable, no saturation surprises) while the heavy expert matmuls stay in the low-precision compute dtype. You get stability where it matters and speed where it matters.
- **Large-logit safety.** In the Mixtral branch of `_combine_weights`, non-selected logits are filled with `torch.finfo(dtype).min` rather than literal `-inf`. The finite minimum cannot produce `NaN`, and softmax's internal max-subtraction handles the large magnitudes. (The related z-loss in `losses.py` uses `logsumexp` for the same overflow-free reason.)

---

## Common pitfalls

- **Expecting eval to be noisy.** Jitter and noisy gating are guarded by `self.training`. If you forget `model.eval()`, your "deterministic" inference still perturbs logits. Conversely, z-score normalisation runs in *both* modes (with different stats) — it is not training-only.
- **Assuming combine weights sum to 1.** By default they sum to `<= 1` (masked full softmax). Only `normalize_router_weights = True` makes them sum to 1. Code that assumes normalised gates will silently mis-scale outputs.
- **Reading `expert_indices` as dispatch truth for Expert Choice.** There it is interface-only; the real routing lives in `combine_weights`.
- **Forgetting `SwitchRouter` needs `top_k == 1`.** It raises in `__init__`; the config also rejects `router_type='switch'` with `top_k != 1` in `validate()`.
- **Ignoring drops.** A Switch token that overflows capacity has an all-zero combine row and contributes nothing but the residual. If you never log the drop count, you cannot see your capacity is too tight.
- **Adding a bias to `w_gate`.** Tempting, but it biases the partition and fights the aux loss. The library forbids it deliberately.

---

## Exercises

1. **Verify train vs. eval noise.** Build a `TopKRouter` with `use_noisy_gating=True` and `jitter_noise=0.1`. Feed the *same* input twice in `.train()` mode and twice in `.eval()` mode. Show that `router_logits` differ across the two training calls but are identical across the two eval calls.

2. **Count combine-weight sums under both conventions.** With `d_model=8, num_experts=4, top_k=2`, run a `TopKRouter` once with `normalize_router_weights=False` and once with `True`. For each, compute `combine_weights.squeeze(-1).sum(-1)` per token and confirm the first is `<= 1` while the second is `== 1`.

3. **Overflow a Switch expert and count drops.** Construct an input where (almost) every token's argmax is the *same* expert, with a small `capacity_factor`, then count dropped tokens via `(combine_weights.squeeze(-1).sum(-1) == 0).sum()`. Confirm the survivors equal the capacity.

4. **Swap routers via `build_router`.** Using one fixed input, instantiate all three routers through `build_router` (changing only `router_type`, keeping `top_k=1` so Switch is legal) and verify each returns a `RouterOutput` whose `combine_weights` has shape `[N, E, 1]`. For Expert Choice, additionally check that every *expert column* of `combine_weights` has exactly `capacity` non-zeros.

5. **(Stretch) See z-score normalisation cap the logit spread.** With `router_z_score_norm=True`, push large-magnitude inputs through the router in `.train()` for several steps, then inspect `router.running_logit_std` and confirm the post-normalisation logits have std near 1.

---

## Solutions

**1 — train vs. eval noise.**

```python
import torch
from moe.config import MoEConfig
from moe.router import TopKRouter

cfg = MoEConfig(d_model=8, num_experts=4, top_k=2,
                use_noisy_gating=True, jitter_noise=0.1)
r = TopKRouter(cfg)
x = torch.randn(5, 8)

r.train()
a, b = r(x).router_logits, r(x).router_logits
print("train differ:", not torch.allclose(a, b))   # True (jitter + noise)

r.eval()
c, d = r(x).router_logits, r(x).router_logits
print("eval identical:", torch.allclose(c, d))      # True
```

**2 — combine-weight sums.**

```python
for norm in (False, True):
    cfg = MoEConfig(d_model=8, num_experts=4, top_k=2,
                    normalize_router_weights=norm, drop_tokens=False)
    r = TopKRouter(cfg).eval()
    cw = r(torch.randn(6, 8)).combine_weights.squeeze(-1)
    print(norm, cw.sum(-1))   # False: each <= 1 ; True: each ~1.0
```

The default keeps the raw masked probabilities (mass lost to the non-selected experts); the Mixtral branch renormalises over the top-2 so every row sums to one.

**3 — overflow a Switch expert.**

```python
cfg = MoEConfig(d_model=4, num_experts=4, top_k=1,
                router_type="switch", capacity_factor=1.0, drop_tokens=True)
r = SwitchRouter(cfg).eval()

# Force every token toward expert 0 by hand-setting the gate weights.
with torch.no_grad():
    r.w_gate.weight.zero_()
    r.w_gate.weight[0] = 10.0          # expert 0 always wins
x = torch.ones(20, 4)
out = r(x)
sums = out.combine_weights.squeeze(-1).sum(-1)
dropped = int((sums == 0).sum())
cap = cfg.capacity(20)                 # ceil(1.0 * 1 * 20 / 4) = 5
print("capacity:", cap, "kept:", 20 - dropped, "dropped:", dropped)
# kept == cap (5), dropped == 15
```

The first `cap` tokens (FIFO by position) keep a non-zero gate; the rest overflow and are zeroed.

**4 — swap routers.**

```python
from moe.router import build_router
x = torch.randn(12, 8)
for rt in ("topk", "switch", "expert_choice"):
    cfg = MoEConfig(d_model=8, num_experts=4, top_k=1, router_type=rt)
    out = build_router(cfg).eval()(x)
    assert out.combine_weights.shape == (12, 4, 1)
    if rt == "expert_choice":
        nz = (out.combine_weights.squeeze(-1) != 0).sum(0)   # per-expert count
        print("EC per-expert non-zeros:", nz, "capacity:", cfg.capacity(12))
```

Each expert column carries exactly `capacity` non-zeros — the structural guarantee of Expert Choice.

**5 — z-score spread (sketch).** After several `.train()` forward passes on large inputs, `router.running_logit_std` settles near the batches' logit std (EMA, momentum 0.99). Dividing logits by it yields a post-normalisation std near 1, so the softmax never saturates regardless of input scale.

---

## Key takeaways

- All three routers return one `RouterOutput`; that uniform contract is what makes the layer router-agnostic and `build_router` a one-line swap.
- `_routing_logits` is the pipeline: training-only jitter → biasless `W_g` projection → optional z-score normalisation (both modes) → training-only `softplus`-scaled Gaussian noise. Noise and jitter aid *exploration and balancing*; `softplus` keeps the noise scale positive and learnable.
- `_combine_weights` has two conventions: default masked full softmax (per-token sum `<= 1`, Switch) vs. `normalize_router_weights` renormalisation to 1 (Mixtral).
- `SwitchRouter` is top-1 (`argmax`) with a `cumsum` FIFO capacity buffer; overflow tokens get a zero combine weight and are *observable* as all-zero rows.
- `ExpertChoiceRouter` inverts routing (`topk` along the token axis): no dropped-token starvation, but some tokens get many experts and some none; its `expert_indices` is interface-only.
- Stability comes for free: softmax runs in float32 under autocast, and masking uses `finfo.min` rather than `-inf`.

## Next → `06-the-moe-layer.md`

You now know how a token is *scored* and *selected*. Next we assemble the full `MoELayer`: how it flattens inputs, calls `build_router`, **dispatches** tokens to the expert FFNs using `dispatch_weights` and `expert_indices`, **combines** their outputs, adds the residual, and folds `aux_loss` into the training objective.
