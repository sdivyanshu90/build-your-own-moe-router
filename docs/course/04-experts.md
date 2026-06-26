# Lesson 04 — The Experts & Token Dispatch

> An expert is just a small two-layer MLP; the real engineering is in *dispatch* — deciding which tokens each expert sees, capping how many it will take, and running it two different ways that must give bit-for-bit agreeing answers.

## Learning objectives

By the end of this lesson you will be able to:

- Describe what an `Expert` is — the `Linear → activation → Linear` FFN — and how the three activations (`gelu`, `swiglu`, `relu2`) differ.
- Explain the weight-init scheme (`kaiming_uniform_` in `fan_in` mode, zeroed down-projection bias) and *why* `fan_in` is the right mode here.
- Read `expert_ffn` and explain how one kernel serves both a single 2-D expert and a 3-D stack of all experts via a batched `matmul`.
- Trace the capacity logic in `_keep_mask`: cumsum-based FIFO slotting and overflow dropping, plus the `last_overflow_tokens` / `last_dispatched_tokens` bookkeeping.
- Contrast the two dispatch strategies — `_forward_naive` and `_forward_batch` — and explain the float32-accumulation trick and the memory trade-off.
- Explain *why* the two strategies are guaranteed to agree, and how the test suite proves it.

## Prerequisites

- Lessons 01–03 (config and the routing data you'll consume here).
- Comfortable Python and basic deep learning: what a linear layer, an activation, and dropout do.
- A little PyTorch: `nn.Linear`, `F.linear`, batched `torch.matmul`, and tensor indexing.
- The file we're reading: [`moe/experts.py`](../../moe/experts.py).

---

## What an expert actually is

In a Mixture-of-Experts layer, the "experts" are not exotic. Each one is an ordinary feed-forward network — the same `Linear → activation → Linear` block you'd find in any Transformer MLP. What makes the layer a *mixture* is that we keep many of these blocks around and send each token to only a few of them. So let's start with a single expert and build up.

Here is the `Expert.__init__` in full:

```python
def __init__(self, config: MoEConfig) -> None:
    super().__init__()
    self.config = config
    self.activation = config.activation
    self.dropout_p = config.expert_dropout

    bias = config.use_bias
    self.w1 = nn.Linear(config.d_model, config.d_ff, bias=bias)
    self.w2 = nn.Linear(config.d_ff, config.d_model, bias=bias)
    # SwiGLU needs an extra gate projection of the same shape as ``w1``.
    self.w3: nn.Linear | None
    if config.activation == "swiglu":
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=bias)
    else:
        self.w3 = None

    self.reset_parameters()
```

Three things to notice. First, `w1` is the **up-projection** ($d_{\text{model}} \to d_{\text{ff}}$, where `d_ff` is typically a few times larger) and `w2` is the **down-projection** back to $d_{\text{model}}$. Second, there is an optional **gate** projection `w3`, created *only* when the activation is `"swiglu"`. Third, biases are controlled globally by `config.use_bias`, so an expert is either all-bias or bias-free.

### The three activations

The non-linearity sits between the two projections. The library supports exactly three, and you'll meet all of them inside `expert_ffn`:

- **`gelu`** — the plain Gaussian Error Linear Unit, `F.gelu(hidden)`. One input, one output, no extra weights.
- **`relu2`** — "squared ReLU": take `F.relu`, then square it. The code spells the exponent as a named constant `_SQUARE = 2` so there's no bare `2` floating around the kernel. Squared ReLU is the activation the Switch-Transformer preset uses.
- **`swiglu`** — the gated variant. Instead of a fixed non-linearity, the up-projection `hidden` is **multiplied element-wise by `SiLU(gate)`**, where `gate` is a *second* projection of the same input through `w3`:

$$\text{SwiGLU}(x) = \big(\text{SiLU}(W_3 x)\big) \odot (W_1 x)$$

That $\odot$ is why SwiGLU needs the extra `w3`: it learns a data-dependent gate that decides, per hidden unit, how much signal to let through. The price is one more weight matrix per expert.

### Why this init scheme?

`reset_parameters` is short but every line is deliberate:

```python
def reset_parameters(self) -> None:
    nn.init.kaiming_uniform_(self.w1.weight, mode="fan_in", nonlinearity="relu")
    if self.w1.bias is not None:
        nn.init.zeros_(self.w1.bias)
    if self.w3 is not None:
        nn.init.kaiming_uniform_(self.w3.weight, mode="fan_in", nonlinearity="relu")
        if self.w3.bias is not None:
            nn.init.zeros_(self.w3.bias)
    nn.init.kaiming_uniform_(self.w2.weight, mode="fan_in", nonlinearity="relu")
    if self.w2.bias is not None:
        nn.init.zeros_(self.w2.bias)
```

Every weight gets `kaiming_uniform_` in `mode="fan_in"`. Kaiming (He) init scales the weights so that the *variance of the activations* stays roughly constant as signal flows forward. `fan_in` mode bases that scaling on the number of **input** units to the layer. The intuition: each output unit is a sum over `fan_in` inputs, so the more inputs you sum, the larger the output variance — unless you shrink the weights to compensate. Choosing `fan_in` keeps forward-pass activation variance stable as `d_model` grows, which is exactly what you want for a block whose output is added back onto a residual stream. (The alternative, `fan_out`, stabilises the *backward* variance instead; for a residual FFN the forward choice is the conventional one.)

The down-projection **bias is zeroed**. Combined with the symmetric weight init, the expert starts as a small, near-zero perturbation on the residual stream rather than yanking it in some arbitrary direction at step zero — a gentle, stable starting point for training.

---

## One kernel, two shapes: `expert_ffn`

Here's the elegant part of the file. Both a single expert and the entire bank-at-once run through **the same function**, `expert_ffn`. That sharing is not a tidiness nicety — it is the mechanism that *guarantees* the two dispatch strategies later produce the same numbers. Keep that in mind as we read it.

The function takes `x`, the six weight/bias tensors (`w1, b1, w2, b2, w3, b3`), plus `activation`, `dropout_p`, and `training`. Its first line decides which world we're in:

```python
batched = w1.dim() == _SQUARE + 1  # 3-D weights => batched per-expert path
```

`_SQUARE + 1` is just `3`. If `w1` is 2-D (`[d_ff, d_model]`) we have **one expert** and `x` is `[num_tokens, d_model]`. If `w1` is 3-D (`[num_experts, d_ff, d_model]`) we have a **stack of every expert** and `x` is `[num_experts, capacity, d_model]`. Same math, two shapes.

The up-projection branches on that flag:

```python
if batched:
    hidden = _batched_linear(x, w1, b1)
else:
    hidden = F.linear(x, w1, b1)
```

For a single expert, `F.linear` does the job. For the batched case we **cannot** use `F.linear`, because it can't apply a *different* weight matrix to each slice of `x`. So we call the small helper `_batched_linear`, which is just a per-expert matrix multiply:

```python
def _batched_linear(x, weight, bias):
    # x: [E, C, in], weight: [E, out, in]  ->  [E, C, out]
    out = torch.matmul(x, weight.transpose(-1, -2))
    if bias is not None:
        out = out + bias.unsqueeze(1).to(out.dtype)
    return out
```

`torch.matmul` on 3-D tensors is a *batched* matmul: it multiplies token slice `x[e]` (`[C, in]`) by expert `e`'s transposed weight (`[in, out]`), independently for every expert `e`, giving `[E, C, out]`. The bias, shaped `[E, out]`, is broadcast across the capacity axis with `bias.unsqueeze(1)` and cast to the result dtype so it can't accidentally re-promote a bfloat16 result back to float32.

Why `matmul` and not `einsum` (which would express the same contraction)? Because `matmul` is **autocast-eligible**: under `torch.autocast` it runs in bf16 exactly like the single-expert `F.linear`, so the naive and batched dispatch paths produce the *same output dtype* in mixed precision. `einsum` is not on the autocast cast list, so it would silently keep the batched path in float32 — a subtle inconsistency we deliberately avoid. (You'll measure this in Lesson 09.)

The activation block handles all three cases, with SwiGLU computing its gate the same dual-path way (`_batched_linear` when batched, `F.linear` otherwise) and raising a clear `ValueError` if `swiglu` was requested but `w3 is None`. Then dropout — applied **only** when `dropout_p > 0.0 and training` — and finally the down-projection mirrors the up-projection: `_batched_linear(hidden, w2, b2)` when batched, `F.linear` otherwise.

The payoff: there is exactly one place in the codebase where "what an expert computes" is defined. Anything that calls `expert_ffn` — a lone `Expert.forward`, or the batched bank — is, by construction, computing the identical function. We'll cash that in shortly.

---

## The dispatch problem and capacity

Now to `ExpertBank`, which owns a list of experts and the logic that feeds them. The bank's `forward` receives, per token, a set of `dispatch_weights` (which expert(s) to use and with what combine weight) produced by the router in the next lesson. The dispatch problem is: *take `[num_tokens, d_model]` tokens, send each to its chosen expert(s), run the experts, and reassemble a `[num_tokens, d_model]` output.*

There's a catch that real systems must handle: **load is uneven**. If the router sends 900 of 1000 tokens to expert 3, you can't let that one expert balloon. So MoE layers impose a **capacity** `C` — a fixed maximum number of tokens any single expert will accept this batch. Tokens beyond `C` are *dropped* (they skip the expert and contribute nothing). The config computes `C` for you via `config.capacity(num_tokens)`; with `drop_tokens=False` it equals `num_tokens` (nothing can overflow), otherwise it's roughly `ceil(capacity_factor * top_k * num_tokens / num_experts)`.

### `_keep_mask`: cumsum FIFO slotting

The bank decides *which* assignments fit in one clean stroke. Given a boolean `selected` mask of shape `[num_tokens, num_experts]` (true where a token picked an expert):

```python
def _keep_mask(self, selected: Tensor, capacity: int) -> tuple[Tensor, Tensor]:
    position = selected.long().cumsum(dim=0) - 1  # [num_tokens, num_experts]
    keep = selected & (position < capacity)
    return keep, position
```

Read it column by column (one column per expert). `cumsum(dim=0)` runs *down the token axis*, giving a 1-based running count of how many tokens so far have chosen this expert. Subtract 1 and you get a **0-based slot index**: the first token to pick expert `e` lands in slot 0, the second in slot 1, and so on. Because cumsum follows token order, this is a deterministic **FIFO**: earlier tokens get the slots, latecomers spill over.

`keep` is then simply "was this assignment selected **and** does its slot fit?" — `position < capacity`. Any assignment with rank $\ge C$ has `keep = False` and is dropped. `position` is meaningful only where `keep` is true, but we return both because the batched path needs the slot numbers.

### Overflow bookkeeping

Back in `forward`, right after computing the mask, the bank records two diagnostics:

```python
self.last_dispatched_tokens = int(selected.sum().item())
self.last_overflow_tokens = int((selected & ~keep).sum().item())
```

`last_dispatched_tokens` is how many (token, expert) assignments were *requested*; `last_overflow_tokens` is how many were **dropped** (`selected` but not `keep`). These are plain instance attributes — monitoring only, never part of autograd. They're invaluable for spotting a collapsing router: a healthy run drops few tokens; a high overflow count means load is badly imbalanced.

Crucially, capacity is enforced **here, once**, before either strategy runs. Both strategies receive the *same* `keep` mask — which is the first half of why they agree.

---

## Strategy 1 — `_forward_naive`: the readable loop

The reference implementation is a plain Python loop, one iteration per expert:

```python
def _forward_naive(self, x: Tensor, combine: Tensor, keep: Tensor) -> Tensor:
    output = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    out_dtype = x.dtype
    for e, expert in enumerate(self.experts):
        token_mask = keep[:, e]  # [num_tokens]
        if not bool(token_mask.any()):
            continue  # skip idle experts (also gives expert independence)
        selected_tokens = x[token_mask]            # [n_e, d_model]
        expert_out = expert(selected_tokens)       # [n_e, d_model]
        out_dtype = expert_out.dtype
        weight = combine[token_mask, e].unsqueeze(-1).float()  # [n_e, 1]
        output[token_mask] = output[token_mask] + weight * expert_out.float()
    return output.to(out_dtype)
```

For each expert we slice out exactly the tokens that kept a slot, run them through that expert (which calls `expert_ffn` with 2-D weights), scale by the per-token combine weight, and **add** into the output. The add — rather than assign — matters because with `top_k > 1` a single token may be processed by several experts, and its outputs sum.

Two design points worth dwelling on:

- **`continue` on idle experts.** If an expert received no kept tokens, we skip it entirely. This isn't just a speed-up: it means an unused expert's weights are *never touched*, so perturbing them cannot change the output. That's **expert independence**, and `test_expert_independence` proves it by adding `100.0` to an idle expert's weights and asserting the output is bit-for-bit unchanged.
- **float32 accumulation.** The `output` buffer is `float32`, and each contribution is `.float()`-ed before adding, even though the experts may compute in bf16 (under autocast). Summing many small bf16 values loses precision fast; accumulating in float32 keeps the running total clean. The final `.to(out_dtype)` casts the result back to the **compute dtype** so the output flows on in the surrounding precision (e.g. bf16) rather than leaking float32 downstream.

This strategy is easy to read and debug, but the Python `for` loop launches separate kernels per expert — fine for study, less ideal for throughput at scale.

## Strategy 2 — `_forward_batch`: one big batched `matmul`

The production path does all experts in a single batched call. First it stacks the per-expert weights into 3-D tensors via `_stacked_weights`, which simply `torch.stack`s each expert's `w1/w2` (and `w3/biases` when present) along a new leading expert axis — producing exactly the 3-D shapes `expert_ffn`'s batched branch expects.

Then:

```python
def _forward_batch(self, x, combine, keep, position, capacity):
    num_tokens, d_model = x.shape
    device, dtype = x.device, x.dtype

    tok_idx, exp_idx = keep.nonzero(as_tuple=True)  # each [num_kept]
    slot = position[tok_idx, exp_idx]               # [num_kept]

    buffer = torch.zeros(self.num_experts, capacity, d_model, device=device, dtype=dtype)
    buffer[exp_idx, slot] = x[tok_idx]

    w1, b1, w2, b2, w3, b3 = self._stacked_weights()
    out_buffer = expert_ffn(buffer, w1, b1, w2, b2, w3, b3,
                            self.config.activation, self.config.expert_dropout,
                            self.training)  # [E, C, d_model]

    output = torch.zeros(num_tokens, d_model, device=device, dtype=torch.float32)
    contrib = (out_buffer[exp_idx, slot].float()
               * combine[tok_idx, exp_idx].unsqueeze(-1).float())
    output.index_add_(0, tok_idx, contrib)
    return output.to(out_buffer.dtype)
```

The shape of the idea:

1. **Flatten** the kept assignments. `keep.nonzero(as_tuple=True)` gives parallel arrays `tok_idx` (which token) and `exp_idx` (which expert), and `position` tells us each one's `slot`.
2. **Scatter** into a dense `[E, C, d_model]` buffer: `buffer[exp_idx, slot] = x[tok_idx]`. Now every expert's tokens sit in a contiguous block of `C` slots (unused slots stay zero).
3. **One batched FFN** — a single `expert_ffn` over stacked weights computes all experts at once. This is the whole point: one fused kernel instead of `num_experts` separate ones.
4. **Gather back** with `index_add_`. Each processed token, weighted by its combine weight, is scattered-added to its original row. `index_add_` handles the `top_k > 1` case automatically: multiple contributions to the same `tok_idx` accumulate.

The same float32 trick reappears — accumulate in float32, then `.to(out_buffer.dtype)` to return in the compute precision.

**The memory cost.** The dispatch buffer is `[num_experts, capacity, d_model]`. The docstring spells out the trade-off: that's larger than the `O(num_tokens * d_model)` input by a factor of `num_experts * capacity / num_tokens`. With `drop_tokens=False` and `capacity == num_tokens` it's `num_experts`× the input. So for many experts, either use a tight capacity (`drop_tokens=True`) or fall back to the naive strategy. Speed and memory pull in opposite directions; this is the dial.

---

## Why the two strategies must agree

This is the design centrepiece, and it rests on two facts you've already seen:

1. **Same capacity decision.** `forward` computes `keep` and `position` *once* and hands the identical mask to whichever strategy runs. They drop exactly the same assignments.
2. **Same kernel.** Both paths ultimately call `expert_ffn` with the same weights and activation — naive with 2-D slices, batch with the 3-D stack — and we proved above those compute the identical function.

So the only difference is *the order of additions* (a loop vs. `index_add_`), which floating-point makes non-associative at the 1e-6 level — hence "identical up to floating-point reordering," not bitwise. `test_dispatch_strategies_agree` nails this down: it routes real top-2 router output through the bank twice — once with `dispatch_strategy="naive"`, once with `"batch"` — across `gelu`, `swiglu`, and bias/bias-free configs, and asserts `torch.allclose(out_naive, out_batch, atol=1e-4)`. Because production runs the batched path, this test is what lets you *trust* the fast path is a faithful reimplementation of the readable one.

---

## Common pitfalls

- **Assuming `expert_indices` drives routing.** In `forward`, `expert_indices` is accepted for interface symmetry but immediately `del`-eted; routing is taken entirely from the **non-zero structure of `dispatch_weights`** (`selected = combine != 0`). If you zero out a combine weight, that assignment vanishes — even if the index still names the expert.
- **Confusing "dispatched" with "kept".** `last_dispatched_tokens` counts *requested* assignments; `last_overflow_tokens` counts *dropped* ones. Kept = dispatched − overflow.
- **Expecting bitwise equality between strategies.** It's `atol=1e-4`, not `torch.equal`. Float addition order differs.
- **Forgetting `drop_tokens=False` disables capacity.** With it off, `capacity == num_tokens` and `last_overflow_tokens` is always 0 — handy for tests, but not how you'd run at scale.
- **Reaching for `F.linear` in the batched path.** It can't broadcast a per-expert weight stack; that's precisely why the batched `torch.matmul` in `_batched_linear` is used instead.
- **Mutating an expert's weights and expecting a stale output.** Idle experts are skipped, so changes to them are invisible *until a token is actually routed there*.

## Exercises

1. **Force an overflow.** Build a 2-expert bank with `capacity_factor=1.0`, `drop_tokens=True`, 10 tokens, and route *all* of them to expert 0. Predict `last_overflow_tokens` and `last_dispatched_tokens`, then verify.
2. **Swap the strategy and diff.** Take a small bank and a top-2 router, run the same input with `dispatch_strategy="naive"` then `"batch"`, and measure `(a - b).abs().max()`. Is it below `1e-4`? Below `1e-6`?
3. **See expert independence.** Route everything to expert 0, snapshot the output, add `100.0` to expert 1's `w1.weight`, and confirm the output is unchanged. Then route one token to expert 1 and watch it change.
4. **Confirm the SwiGLU gate exists.** Construct experts with each activation and check which one has a non-`None` `w3`. Count parameters and confirm SwiGLU's expert is larger.
5. **(Stretch) Eyeball the memory factor.** For `num_experts=8`, `drop_tokens=False`, `num_tokens=64`, print `buffer.shape` inside `_forward_batch` and compare its element count to `x`'s.

## Solutions

```python
import torch
from moe.config import MoEConfig
from moe.experts import Expert, ExpertBank

def all_to_expert0(n, e):
    d = torch.zeros(n, e, 1); d[:, 0, 0] = 1.0
    return d, torch.zeros(n, 1, dtype=torch.long)

# 1) Overflow. capacity = ceil(1.0 * 1 * 10 / 2) = 5, so 10 - 5 = 5 dropped.
cfg = MoEConfig(d_model=4, d_ff=8, num_experts=2, top_k=1,
                capacity_factor=1.0, drop_tokens=True, dispatch_strategy="batch")
bank = ExpertBank(cfg).eval()
d, i = all_to_expert0(10, 2)
bank(torch.randn(10, 4), d, i)
assert bank.last_dispatched_tokens == 10 and bank.last_overflow_tokens == 5

# 2) Strategies agree. Reuse one bank; flip the config field between calls.
cfg2 = MoEConfig(d_model=16, d_ff=32, num_experts=4, top_k=2,
                 drop_tokens=False, expert_dropout=0.0)
bank2 = ExpertBank(cfg2).eval()
x = torch.randn(20, 16)
d2 = torch.zeros(20, 4, 1)                      # hand-built top-2 dispatch
for t in range(20):
    a, b = torch.randperm(4)[:2]
    d2[t, a, 0], d2[t, b, 0] = 0.6, 0.4
i2 = torch.zeros(20, 2, dtype=torch.long)
cfg2.dispatch_strategy = "naive"; out_n = bank2(x, d2, i2)
cfg2.dispatch_strategy = "batch"; out_b = bank2(x, d2, i2)
print((out_n - out_b).abs().max())             # ~1e-7, well under 1e-4

# 3) Independence.
cfg3 = MoEConfig(d_model=8, d_ff=16, num_experts=4, top_k=1, drop_tokens=False)
bank3 = ExpertBank(cfg3).eval()
xx = torch.randn(6, 8); d3, i3 = all_to_expert0(6, 4)
before = bank3(xx, d3, i3)
with torch.no_grad():
    bank3.experts[1].w1.weight.add_(100.0)     # idle expert
assert torch.equal(before, bank3(xx, d3, i3))  # unchanged

# 4) Only SwiGLU has a gate.
for act in ("gelu", "swiglu", "relu2"):
    e = Expert(MoEConfig(d_model=8, d_ff=16, activation=act))
    print(act, "w3 is", "present" if e.w3 is not None else "None")
```

- **(1)** Per-expert capacity is `ceil(capacity_factor * top_k * tokens / num_experts) = ceil(1.0*1*10/2) = 5`; the first 5 tokens (FIFO by `_keep_mask`) keep slots, the rest overflow — exactly the values asserted.
- **(2)** The max abs diff is dominated by float reordering, comfortably under `1e-4` (the suite's tolerance) and usually near `1e-7`.
- **(3)** Expert 1 is idle, so `_forward_naive`'s `continue` never touches it; the output is bit-for-bit identical.
- **(4)** Only `swiglu` builds `w3`; the others print `None`. The SwiGLU expert therefore has one extra `d_ff × d_model` matrix.
- **(5)** `buffer` is `[8, 64, 8]` = 4096 elements vs. `x`'s `64 × 8 = 512` — an 8× (`num_experts`×) blow-up, matching the docstring's memory note.

## Key takeaways

- An expert is a two-layer FFN; `swiglu` alone adds a gate projection `w3`. Init is `kaiming_uniform_` `fan_in` (stable forward variance for a residual block) with a **zeroed down-bias** so experts start near-identity.
- `expert_ffn` is one kernel that handles a single expert (2-D weights, `F.linear`) and all experts at once (3-D weights, a batched `matmul` via `_batched_linear`). Sharing it is what *makes the two dispatch strategies equal* — and using `matmul` (autocast-eligible) keeps their output dtype equal under mixed precision too.
- `_keep_mask` uses a `cumsum` along the token axis to assign FIFO slots and drop everything at rank `≥ capacity`; `last_overflow_tokens` / `last_dispatched_tokens` record the damage for monitoring.
- `_forward_naive` loops per expert (readable, gives expert independence by skipping idle experts); `_forward_batch` scatters into an `[E, C, d_model]` buffer, runs one batched `matmul`, and gathers back with `index_add_` — faster but `num_experts`× the memory.
- Both accumulate in **float32** then cast back to the compute dtype for bf16 safety, and both are proven to agree to `1e-4` because they share the kernel *and* the keep-mask.

## Next → [`05-routing.md`](05-routing.md)

You now know what happens *after* a routing decision. Next we open the box that *makes* those decisions: the router that produces the `dispatch_weights` this bank consumes.
