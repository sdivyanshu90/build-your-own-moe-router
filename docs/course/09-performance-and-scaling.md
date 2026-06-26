# Lesson 09 — Performance, Precision & Scaling Out

> A sparse MoE is only worth building if it is *faster* than the dense model it replaces and *stable* in low precision; this lesson measures the first with `moe/bench.py`, pins down the second by reading where the code forces float32, and then teaches the all-to-all machinery you would need to scale the (single-device) library across many GPUs.

## Learning objectives

By the end of this lesson you will be able to:

- Read `moe/bench.py` end to end — `run_benchmark`, `_time_forward`, `_dense_baseline`, `BenchResult`, `_ForwardOnly` — and explain why the dense baseline is built with `hidden = d_ff * num_experts` (an **equal-parameter** comparison).
- Explain *why* top-2-of-8 routing is several times faster than that dense FFN, and why the speedup measures **activated compute**, not memory.
- State which operations must stay in `float32` (gating logits, softmax, aux/z-loss reductions, the dispatch accumulator) and which are safe in `bfloat16` (the expert matmuls), and point to the exact lines in `moe/` that enforce this.
- Describe **expert parallelism** and the **all-to-all** dispatch/combine pattern, estimate its per-layer communication volume, and contrast it with tensor parallelism.
- Tune the **capacity factor** against the `overflow_fraction` metric as a throughput/quality dial.

## Prerequisites

- Lessons 01–08: you know what a router emits (`RouterOutput`), how `ExpertBank` dispatches tokens, what capacity is, and what the auxiliary and z-losses do.
- Comfort with PyTorch tensors, `torch.no_grad()`, and the idea of `torch.autocast`.
- Optional but recommended: skim `docs/moe_routing.md` §1.4 (numerical stability) and §1.5 (distributed training). This lesson is the hands-on companion to those two sections.

---

## 1. The benchmark: what `make bench` actually measures

The whole point of a sparse MoE is to *buy more parameters without paying for all of them on every token*. `moe/bench.py` is the developer tool that proves we got that bargain. Run it:

```bash
make bench        # == python -m moe.bench
```

It prints a small table and one headline number:

```
Module                              params     mean (ms)    std (ms)
--------------------------------------------------------------------
MoE (top-2 of 8)                 8,404,992        ...          ...
Dense FFN (equal params)         8,397,312        ...          ...
--------------------------------------------------------------------
MoE speedup vs equal-parameter dense FFN: ~2.6x
```

Let's read the file. The timing harness is `_time_forward`:

```python
def _time_forward(module: nn.Module, x: Tensor) -> tuple[float, float]:
    module.eval()
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):   # 10 — warm up caches / lazy init
            module(x)
        samples: list[float] = []
        for _ in range(TIMED_ITERS):    # 100 — the measured loop
            start = time.perf_counter()
            module(x)
            samples.append((time.perf_counter() - start) * MS_PER_SEC)
    return statistics.fmean(samples), statistics.pstdev(samples)
```

Two details matter. First, the **warmup loop** runs `WARMUP_ITERS = 10` forward passes whose timings are *thrown away* — the first calls pay one-time costs (kernel selection, allocator warm-up, cache population) that have nothing to do with steady-state speed. Second, we report a **mean** and a population **standard deviation** (`pstdev`) over `TIMED_ITERS = 100` samples, because a single timing is noise. Everything runs under `module.eval()` and `torch.no_grad()`: a pure forward pass, dropout off, no autograd graph.

The honest comparison lives in `_dense_baseline`:

```python
def _dense_baseline(config: MoEConfig) -> nn.Sequential:
    hidden = config.d_ff * config.num_experts          # <-- the key line
    return nn.Sequential(
        nn.Linear(config.d_model, hidden),
        nn.GELU(),
        nn.Linear(hidden, config.d_model),
    )
```

Why `hidden = d_ff * num_experts`? Because we want the dense model to have the **same total parameter count** as the expert bank. The bank holds `num_experts` separate FFNs, each of width `d_ff`; a single dense FFN with `num_experts × d_ff` hidden units has (up to biases and the tiny router head) the *same* number of weights. That is exactly what the printed `params` columns confirm — `8,404,992` vs `8,397,312`, equal to within the router's `w_gate` and the bias terms.

This equality is the whole trick. Against a parameter-matched dense FFN, the only thing that differs is **how many of those parameters each token actually touches**. The dense FFN pushes every token through all `num_experts × d_ff` hidden units; the MoE pushes each token through only `top_k` of the `num_experts` experts. With the default config (`top_k=2`, `num_experts=8`), each token activates `2/8 = 25%` of the FFN parameters. The FLOP argument is direct: a dense FFN costs $4 \cdot d_{model} \cdot d_{ff}\_\text{total}$ multiply-accumulates per token, while the MoE costs

$$
\text{FLOPs/token}_{\text{MoE}} \approx \text{top\_k} \times \big(4 \cdot d_{model} \cdot d_{ff}\big),
$$

independent of `num_experts`. For top-2-of-8 that is roughly `2/8` of the dense work, and you observe a multi-fold latency win — typically **around 2.6× on CPU** (the exact factor swings with machine, thread count, and the batch/seq shape; on a quiet box it can be noticeably higher).

`run_benchmark` wires it together, with one adapter, `_ForwardOnly`. An `MoELayer` returns a rich `MoELayerOutput` (output, aux loss, z loss, router probs, utilisation); the dense `nn.Sequential` returns a bare tensor. To time *like for like*, `_ForwardOnly.forward` returns just `self.layer(x).output`, so both modules do the same observable work. `main` then computes `speedup = dense.mean_ms / moe.mean_ms` and prints it.

**The one caveat to remember:** this benchmark measures **activated compute** (forward latency), not **memory**. The MoE still *stores* all `num_experts` experts' weights, and the `"batch"` dispatch path even allocates a `[num_experts, capacity, d_model]` buffer larger than the input. Sparse routing saves FLOPs per token; it does **not** save parameter memory. That trade — lots of cheap-to-store-but-rarely-touched parameters — is the entire value proposition of MoE.

---

## 2. Mixed precision: what stays float32, what goes bfloat16

Modern training runs in **bfloat16**: an **8-bit exponent** (the same dynamic range as float32) and a **7-bit mantissa** (about 2–3 decimal digits). The wide exponent means bf16 rarely overflows; the narrow mantissa means it is *imprecise*, and that imprecision compounds in **sums**. The rule follows: keep the precision-sensitive, accumulation-heavy parts in float32, let the throughput-bound matmuls run in bf16.

Three kinds of computation must stay float32, and this library enforces each explicitly rather than hoping autocast does it:

**1. The auxiliary load-balancing loss** averages probabilities over the whole batch — a long sum where bf16 rounding would corrupt the balance statistics. `moe/losses.py` casts up front:

```python
mean_prob = router_probs.float().mean(dim=0)   # P_i, in float32
```

**2. The router z-loss** takes a log-sum-exp over experts. It is computed in float32 *and* through `torch.logsumexp`, which subtracts the row max before exponentiating:

```python
log_z = torch.logsumexp(router_logits.float(), dim=-1)   # stable, fp32
```

**3. The dispatch/combine accumulator.** When the bank gathers each expert's weighted output back into the token grid, it accumulates in a float32 buffer and only casts back to the compute dtype at the very end:

```python
# ExpertBank._forward_batch
output = torch.zeros(num_tokens, d_model, device=device, dtype=torch.float32)
contrib = (out_buffer[exp_idx, slot].float()
           * combine[tok_idx, exp_idx].unsqueeze(-1).float())
output.index_add_(0, tok_idx, contrib)
return output.to(out_buffer.dtype)   # cast back to the expert compute dtype
```

The naive path (`_forward_naive`) does the same thing — `output = torch.zeros(..., dtype=torch.float32)`, accumulate, then `output.to(out_dtype)`. The comment in the source says it plainly: *"Accumulation happens in float32 for numerical stability (important under bfloat16 autocast) and the result is cast back to the expert compute dtype so the output flows in the surrounding precision."* A token routed to several experts (top_k > 1) sums several contributions; doing that sum in bf16 would visibly drift.

What is *safe* in bf16 is the heavy lifting: the **expert FFN matmuls** (`expert_ffn`'s `F.linear` on the single-expert path, and the batched `torch.matmul` in `_batched_linear` on the stacked path). These are large GEMMs whose rounding error averages out and whose dynamic range bf16 handles comfortably — and they are exactly where the time goes. Both are autocast-eligible, so under `torch.autocast` they cast to bf16 automatically and the naive and batched dispatch paths return the **same** dtype (Exercise 3). This is exactly why the kernel uses `matmul` rather than `einsum` for the batched case: `einsum` is *not* on the autocast cast list, so it would have left the batched path stuck in float32.

One subtle choice in `moe/router.py`: the gating logits are **not** force-cast to float32. The `_routing_logits` docstring explains why — the softmax sits on the autocast float32 policy and the loss functions already cast their inputs, so probabilities stay stable while `w_gate(x)` runs in the compute dtype. The safety net is the explicit `.float()` calls in `losses.py`, not a blanket router upcast.

---

## 3. Numerical stability as a recurring engineering practice

The same three moves keep reappearing across this codebase — they are the *house style* for sparse routing in low precision:

- **Max-shift softmax.** Subtracting the per-token max before `exp` is mathematically identical to the naive softmax but never overflows (the largest exponent becomes $e^0 = 1$). Every `F.softmax` in `router.py` does this internally.
- **`logsumexp` instead of `log(sum(exp))`.** The z-loss's log-partition function goes through `torch.logsumexp`, which folds the max-shift in and stays finite even for logits in the thousands, where `exp` overflows to `inf` (above ~88 in float32).
- **float32 accumulation, then cast back.** Any reduction over the batch or over experts — `P_i`, the dispatch combine, the loss means — runs in float32, returning the compute dtype only at the end.

None of these change the math; they change the *representable range* so the math survives bf16 — exact algorithm, robust implementation.

---

## 4. Scaling out: expert parallelism and the all-to-all

This library is deliberately **single-device** — every expert lives in one process. But the architecture is built to scale, so it's worth understanding *how*. At large `num_experts` the experts no longer fit on one GPU, so MoE is trained with **expert parallelism**: the experts are **sharded** across devices, each device owning a disjoint subset. Because any token may be routed to any expert, the layer must shuffle tokens to the device that holds their chosen expert, run the experts locally, and shuffle the results back. Those two shuffles are **all-to-all** collectives — *dispatch* before the experts, *combine* after.

```
   Device 0           Device 1           Device 2           Device 3
 [tokens ...]       [tokens ...]       [tokens ...]       [tokens ...]
      |                  |                  |                  |
      v                  v                  v                  v
  +----------------------------------------------------------------+
  |   route locally: assign each token its top-k expert id(s)       |
  +----------------------------------------------------------------+
      |   \   \           /   |   \           /   |                |
      v    v   v         v    v    v         v    v   (each token sent
  ==============   ALL-TO-ALL  (DISPATCH)  ===================
      v    v   v         v    v    v         v    v    to the device that
  [expert 0,1]       [expert 2,3]       [expert 4,5]       [expert 6,7]
   run FFN            run FFN            run FFN            run FFN
      |                  |                  |                  |
  ==============   ALL-TO-ALL  (COMBINE)  ====================
      |                  |                  |                  |
      v                  v                  v                  v
   weight by g_i, scatter back to original positions, add residual
```

**Communication volume.** Each all-to-all moves roughly one token-vector per token, and there are two of them, so per MoE layer per forward pass:

$$
\text{bytes} \approx 2 \times T \times d_{model} \times \text{sizeof(dtype)},
$$

where $T$ is the number of routed tokens and the factor of 2 counts dispatch + combine. For $T = 4096$, $d_{model} = 1024$, bf16 (2 bytes): $2 \times 4096 \times 1024 \times 2 \approx 16$ MiB per layer per forward — and the same again on the backward pass. This is why **interconnect bandwidth**, not FLOPs, is the binding constraint for MoE at scale.

**The layout transform** is exactly the one `ExpertBank` already builds locally: `[batch, seq, d_model]` → a capacity-padded `[num_experts, capacity, d_model]` buffer (the `buffer = torch.zeros(self.num_experts, capacity, d_model, ...)` in `_forward_batch`) so the all-to-all moves **fixed-size, contiguous blocks** → inverse scatter (`output.index_add_(0, tok_idx, contrib)`) back to `[batch, seq, d_model]`. Fixed-size buffers are *why capacity exists* — variable-size all-to-all is far harder to schedule. The single-device code is the same algorithm with the collectives elided.

**Replicated vs sharded.** The **router `w_gate`** is tiny and **replicated** across data-parallel devices; its gradients are all-reduced like any data-parallel parameter. The **expert weights are sharded**; an expert's gradient comes only from the tokens dispatched to it and is *not* all-reduced across the expert-parallel group. Consequence for the aux loss: its global statistics ($f_i$, $P_i$) span all devices, so they must be reduced across that group before the loss is formed.

**Expert vs tensor parallelism.** Tensor parallelism splits each matmul across devices and all-reduces *every layer* — fine-grained, latency-sensitive, best *within* one high-bandwidth node. Expert parallelism communicates only twice per MoE layer but moves whole tokens — it tolerates more latency and scales as you add experts, best *across* nodes. Large models combine both.

---

## 5. Capacity factor: the throughput/quality dial

The one knob that ties performance to quality is the **capacity factor**. Recall `MoEConfig.capacity`:

$$
C = \Big\lceil \text{capacity\_factor} \times \text{top\_k} \times \frac{T}{\text{num\_experts}} \Big\rceil .
$$

`capacity_factor = 1.0` provisions exactly the fair share; the default `1.25` adds 25% slack so a mildly imbalanced batch still fits. Larger capacity → fewer dropped tokens (better quality) but bigger buffers and more padding (worse throughput). Smaller capacity → cheaper and faster, but tokens **overflow** and are dropped (surviving only via the residual).

You don't have to guess — the layer reports it. `MoELayer.get_routing_stats()` exposes `overflow_fraction`, computed in the bank as `last_overflow_tokens / last_dispatched_tokens`. Pick the **smallest** `capacity_factor` whose `overflow_fraction` stays acceptably low (a few percent at most). That is the dial; `overflow_fraction` is the gauge next to it.

---

## Common pitfalls

- **Comparing against the wrong dense model.** If your baseline FFN doesn't use `hidden = d_ff * num_experts`, the speedup is meaningless — different capacity, not different sparsity.
- **No warmup.** The first forward pass measures kernel selection and allocator warm-up, not steady-state speed. Discard a few iterations (the harness uses 10).
- **Believing MoE saves memory.** It saves *activated FLOPs per token*, not storage. All experts sit in memory; the `"batch"` path even adds a `[num_experts, capacity, d_model]` buffer.
- **Accumulating in bf16.** Summing top_k contributions or batch-wide probabilities in bf16 drifts. The code accumulates in float32 and casts back — don't "optimize" that away.
- **Setting `capacity_factor` blind.** Too low silently drops tokens; too high wastes compute. Drive it with `overflow_fraction`.

## Exercises

1. **Run and read the benchmark.** Run `make bench`. Confirm the two `params` columns are nearly equal and explain the small gap. Report the speedup and relate it to `top_k / num_experts`.
2. **`overflow_fraction` vs `capacity_factor`.** Sweep `capacity_factor` over `{1.0, 1.25, 2.0}` for a deliberately imbalanced input and read `overflow_fraction` from `get_routing_stats()`. Which value is the smallest that keeps overflow near zero?
3. **Confirm the dtypes.** Under `torch.autocast`, verify that an expert matmul runs in bf16 while the aux and z losses come back float32.
4. **Estimate communication.** For `num_experts=64`, `T=8192`, `d_model=2048`, bf16, compute the per-layer per-forward all-to-all volume from the formula in §4.

## Solutions

**1.** The MoE has slightly *more* parameters (`8,404,992` vs `8,397,312`): the extra ~7.7k are the per-expert bias terms plus the router's `w_gate` head, which the bare dense `nn.Sequential` doesn't fully mirror. The speedup is governed by `top_k / num_experts = 2/8 = 0.25` activated fraction, so you expect a several-fold win (≈2.6× typical on CPU; higher on a quiet machine).

**2.**

```python
import torch
from moe.config import MoEConfig
from moe.layer import MoELayer

torch.manual_seed(0)
x = torch.randn(4, 256, 128)
for cf in (1.0, 1.25, 2.0):
    cfg = MoEConfig(d_model=128, d_ff=256, num_experts=8, top_k=2,
                    capacity_factor=cf, drop_tokens=True)
    layer = MoELayer(cfg).eval()
    with torch.no_grad():
        layer(x)
    print(cf, round(layer.get_routing_stats()["overflow_fraction"], 4))
```

`overflow_fraction` falls as `capacity_factor` rises; pick the smallest `cf` that drives it near zero for your data. (With `drop_tokens=False`, capacity becomes the full token count and overflow is always 0 — at the cost of the largest buffers.)

**3.**

```python
import torch, torch.nn.functional as F
from moe.config import MoEConfig
from moe.router import build_router

cfg = MoEConfig(d_model=64, num_experts=8, top_k=2)
r = build_router(cfg).eval()
x = torch.randn(32, 64)
with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
    print("expert matmul :", F.linear(x, torch.randn(128, 64)).dtype)  # bfloat16
    out = r(x)
    print("aux_loss      :", out.aux_loss.dtype)                       # float32
```

The matmul casts to bf16 under autocast; `aux_loss` (and the z-loss) come back float32 because `losses.py` calls `.float()` on their inputs explicitly.

**4.** $2 \times 8192 \times 2048 \times 2 = 67{,}108{,}864$ bytes $\approx 64$ MiB per MoE layer per forward pass (and the same again on the backward). Scale that by your layer count to see why the network fabric dominates.

## Key takeaways

- `make bench` (`moe/bench.py`) times a 10-iter warmup + 100-iter loop against an **equal-parameter** dense FFN (`hidden = d_ff * num_experts`), isolating the benefit of activating `top_k` of `num_experts`. Expect ≈2.6× for top-2-of-8 on CPU.
- The speedup is **activated-compute** savings (FLOPs/token ≈ `top_k × dense_FFN`), **not** memory savings — all experts still live in memory.
- bf16 is safe for the **expert matmuls** but not for sums. The code forces float32 for the **aux and z losses** (explicit `.float()` in `losses.py`) and the **dispatch accumulator** (`torch.zeros(..., dtype=torch.float32)` → `.to(out_dtype)`).
- Stability is a *practice*: max-shift softmax, `torch.logsumexp`, and float32-accumulate-then-cast recur throughout — exact math, robust implementation.
- Scaling out means **expert parallelism** + **all-to-all** dispatch/combine, router replicated and experts sharded; communication ≈ $2 \cdot T \cdot d_{model} \cdot \text{sizeof(dtype)}$ per layer per forward dominates.
- The **capacity factor** is the throughput/quality dial; tune it by `overflow_fraction` from `get_routing_stats()`.

## Next → `10-capstone.md`

You can now build, balance, route, dispatch, *and* reason about the speed, precision, and scaling of an MoE layer. In the capstone you'll put the whole stack together end to end — config, router, experts, losses, and monitoring — and train a small model that demonstrates everything these nine lessons built.
