# Lesson 06 — Assembling the MoE Layer

> In this lesson we wire the router, the expert bank, and the auxiliary losses into `MoELayer` — a single `nn.Module` that drops into a transformer exactly where a dense FFN used to sit.

## Learning objectives

By the end of this lesson you will be able to:

- Explain how `MoELayer` composes a router, an `ExpertBank`, and a `MoELoss` into one drop-in feed-forward replacement.
- Trace the seven steps of `MoELayer.forward` from a `[batch, seq, d_model]` input to a `MoELayerOutput`.
- Wire the layer into a residual block: `residual = x + out.output` and `loss = task_loss + out.aux_loss + out.z_loss`.
- Read `get_routing_stats()` and interpret every metric it returns.
- Explain why, under bfloat16 autocast, `out.output` is bf16 while `out.aux_loss` and `out.z_loss` come back as float32.

## Prerequisites

- Lessons 02–05: you should already know what a router produces (`RouterOutput`), what the `ExpertBank` does with dispatch weights, and what the auxiliary and z-losses measure.
- Comfort with PyTorch `nn.Module`, tensor reshaping, and `NamedTuple`.
- A passing familiarity with `torch.autocast` / mixed precision is helpful for the final section but not required.

The file we are studying is `moe/layer.py`. Open it alongside this lesson.

## The big picture: one layer, three collaborators

A sparse MoE layer is not a monolithic block — it is a small orchestra. Three independent components each do one job:

- **the router** decides *where each token goes* and with *what weight*;
- **the `ExpertBank`** runs the chosen experts and combines their outputs;
- **the `MoELoss`** turns routing decisions into a regularisation signal that keeps the experts balanced.

`MoELayer` is the conductor. It owns one of each, hands data between them in the right order, and packages the result so a caller can use it like any other FFN. Look at the constructor:

```python
def __init__(self, config: MoEConfig) -> None:
    super().__init__()
    config.validate()
    self.config = config
    self.router = build_router(config)
    self.expert_bank = ExpertBank(config)
    self.criterion = MoELoss.from_config(config)

    self._last_router_probs: Tensor | None = None
    self._last_expert_counts: Tensor | None = None
```

Three things to notice. First, `config.validate()` runs immediately, so an inconsistent config fails at construction rather than deep inside a training loop. Second, `build_router(config)` is a *factory*: it reads `config.router_type` and returns a `TopKRouter`, `SwitchRouter`, or `ExpertChoiceRouter`. The layer never names a concrete router class — it asks the factory — which is exactly why you can swap routing strategies by changing one config field. Third, the two `_last_*` attributes start as `None`. They are *monitoring state*, cached on each forward pass, and we will return to them when we discuss `get_routing_stats()`.

`MoELoss.from_config(config)` builds the criterion with the same `alpha`, `beta`, and `num_experts`, so the layer and its loss can never drift out of sync.

## The forward pass, step by step

Here is the heart of the layer. Read it once, then we will walk through it.

```python
def forward(self, x: Tensor) -> MoELayerOutput:
    if x.size(-1) != self.config.d_model:
        raise ValueError(
            f"Input trailing dim ({x.size(-1)}) must equal d_model "
            f"({self.config.d_model})."
        )
    orig_shape = x.shape
    x_flat = x.reshape(-1, self.config.d_model)  # [num_tokens, d_model]

    # 1-2. Route.
    router_out = self.router(x_flat)

    # 3-4. Dispatch to experts and combine (weighted sum done in the bank).
    combined = self.expert_bank(
        x_flat, router_out.dispatch_weights, router_out.expert_indices
    )

    # 5. Restore the original [batch, seq, d_model] shape.
    output = combined.reshape(orig_shape)

    # 6. Compute the weighted auxiliary losses.
    aux_weighted, z_weighted = self._weighted_losses(router_out)

    # Cache detached monitoring state.
    expert_counts = self._expert_counts(router_out)
    self._last_router_probs = router_out.router_probs.detach()
    self._last_expert_counts = expert_counts
    total_slots = expert_counts.sum().clamp_min(1)
    utilisation = expert_counts.float() / total_slots.float()

    # 7. Return everything the caller needs.
    return MoELayerOutput(
        output=output,
        aux_loss=aux_weighted,
        z_loss=z_weighted,
        router_probs=router_out.router_probs,
        expert_utilisation=utilisation,
    )
```

**Shape check.** A transformer hidden state is `[batch, seq, d_model]`. If the trailing dimension does not equal `config.d_model`, the gating matmul would silently mismatch, so the layer raises a clear `ValueError` up front. This is a contract, not paranoia — failing loudly here saves you a confusing stack trace three modules deep.

**Flatten `[B, T, d] → [B·T, d]`.** Routing is a *per-token* decision; the batch and sequence axes carry no routing meaning. We remember `orig_shape`, then collapse everything but the feature axis into one long list of token vectors with `x.reshape(-1, d_model)`. The router and the expert bank both think in terms of `num_tokens`, never `batch × seq`.

**Run the router (steps 1–2).** `self.router(x_flat)` returns a `RouterOutput` — the bundle from Lesson 03. The fields the layer uses are `dispatch_weights` (where to send each token, with weights), `expert_indices` (which experts were chosen), `router_logits` (pre-softmax, for the z-loss), and `router_probs` (the full softmax distribution, for the aux loss and for monitoring).

**Dispatch and combine (steps 3–4).** `self.expert_bank(x_flat, dispatch_weights, expert_indices)` does the heavy lifting: it groups tokens by expert, runs each expert's MLP, and — crucially — performs the *weighted sum* of expert outputs itself, using the combine weights baked into `dispatch_weights`. The layer never multiplies anything by a gate value; that arithmetic lives inside the bank. The result `combined` is `[num_tokens, d_model]`.

**Reshape back (step 5).** `combined.reshape(orig_shape)` restores `[batch, seq, d_model]`, so the output lines up element-for-element with the input — a precondition for the residual add the caller will perform.

**Compute the weighted losses (step 6).** This is delegated to a small helper:

```python
def _weighted_losses(self, router_out: RouterOutput) -> tuple[Tensor, Tensor]:
    zero_task = router_out.router_probs.new_zeros(())
    loss_out = self.criterion(
        zero_task,
        router_out.router_probs,
        router_out.expert_indices,
        router_out.router_logits,
    )
    aux_weighted = self.config.alpha * loss_out.aux_loss
    z_weighted = self.config.beta * loss_out.z_loss
    return aux_weighted, z_weighted
```

Here is the key design decision. `MoELoss.forward` expects a `task_loss` as its first argument, but **the layer does not know the task** — it has no idea whether you are doing language modelling, classification, or regression. So it passes `zero_task`, a scalar zero created with `new_zeros(())` so it matches the dtype and device of `router_probs`. The criterion returns *raw* (unweighted) `aux_loss` and `z_loss`; the layer then multiplies them by `config.alpha` and `config.beta`. That is why `MoELayerOutput.aux_loss` is documented as `alpha * raw` and `z_loss` as `beta * raw` — they are pre-weighted and ready to add straight to your loss. The caller supplies the real task loss separately and sums everything up.

**Cache monitoring state.** Two helpers feed the cache. `_expert_counts` counts how many dispatch slots each expert received:

```python
def _expert_counts(self, router_out: RouterOutput) -> Tensor:
    flat = router_out.expert_indices.reshape(-1).detach().to(torch.long)
    return torch.bincount(flat, minlength=self.config.num_experts)
```

Note the `.detach()`: these counts are diagnostics, never gradients. The layer then stashes `self._last_router_probs = router_out.router_probs.detach()` and `self._last_expert_counts = expert_counts`, and computes `utilisation` — each expert's share of the routed slots — guarding the division with `.clamp_min(1)` so an all-empty batch cannot divide by zero.

**Return `MoELayerOutput` (step 7).** Everything the caller could want, in one typed bundle.

## `MoELayerOutput` and how you wire it up

`MoELayerOutput` is a `NamedTuple` with five fields:

| Field | Shape | Meaning |
| --- | --- | --- |
| `output` | `[batch, seq, d_model]` | the layer's contribution to the residual stream |
| `aux_loss` | scalar | weighted load-balancing loss (`alpha * raw`) |
| `z_loss` | scalar | weighted router z-loss (`beta * raw`) |
| `router_probs` | `[num_tokens, num_experts]` | full routing distribution (kept attached) |
| `expert_utilisation` | `[num_experts]` | fraction of slots each expert got this pass |

The intended usage — straight from the package docstring in `moe/__init__.py` — is two lines:

```python
import torch
from moe import MoELayer, MoEConfig

layer = MoELayer(MoEConfig.mixtral_style(num_experts=4, d_model=32))
hidden = torch.randn(2, 8, 32)

out = layer(hidden)
residual = hidden + out.output                 # 1) residual connection
loss = out.aux_loss + out.z_loss               # 2) add the MoE regularisers
# loss = task_loss + out.aux_loss + out.z_loss # ... plus your real task loss
```

`out.output` is *the FFN sub-layer's output*, not the post-residual value — adding `hidden + out.output` is your responsibility, exactly as with a dense FFN. And because `aux_loss` / `z_loss` arrive pre-weighted, you simply add them; the layer already applied `alpha` and `beta`. The package docstring keeps `out.router_probs` attached on purpose, so you can layer on extra regularisation if you want — detaching it is the caller's call.

## Reading `get_routing_stats()`

After a forward pass, ask the layer how healthy its routing was:

```python
def get_routing_stats(self) -> dict[str, Any]:
    if self._last_router_probs is None or self._last_expert_counts is None:
        return {}

    entropy = compute_routing_entropy(self._last_router_probs)
    imbalance = compute_load_imbalance(self._last_expert_counts.float())
    dead = detect_dead_experts(self._last_expert_counts.float())

    dispatched = max(self.expert_bank.last_dispatched_tokens, 1)
    overflow_fraction = self.expert_bank.last_overflow_tokens / dispatched

    return {
        "entropy_mean": float(entropy.mean),
        "entropy_min": float(entropy.min),
        "imbalance_ratio": float(imbalance.imbalance_ratio),
        "coefficient_of_variation": float(imbalance.coefficient_of_variation),
        "overflow_fraction": float(overflow_fraction),
        "dead_experts": dead,
        "expert_utilisation": (
            self._last_expert_counts.float()
            / self._last_expert_counts.sum().clamp_min(1).float()
        ).tolist(),
    }
```

If `forward` has never run, the cache is `None` and you get an empty dict — a safe sentinel. Otherwise the method reads the cached `_last_router_probs` and `_last_expert_counts`, plus two counters that live on the expert bank, and reports:

- **`entropy_mean`** — average Shannon entropy (in nats) of the per-token routing distributions. High means tokens spread their probability across experts; it falls toward 0 as routing sharpens.
- **`entropy_min`** — the entropy of the *most collapsed* token. A near-zero minimum is your early warning that at least one token has committed almost entirely to a single expert.
- **`imbalance_ratio`** — `max_i f_i / mean_i f_i` over expert load fractions. `1.0` is perfect balance; larger means one expert is hogging tokens.
- **`coefficient_of_variation`** — `std_i f_i / mean_i f_i` of the load. `0.0` is perfectly even; it grows without bound as load concentrates.
- **`overflow_fraction`** — the share of dispatched (token, expert) slots that were *dropped* for exceeding capacity. This is the only metric not derived from the cached tensors: it is `expert_bank.last_overflow_tokens / last_dispatched_tokens`, two counters the bank refreshes on every call. A persistently high value means your `capacity_factor` is too tight (or your routing is too lopsided).
- **`dead_experts`** — a list of expert indices that received under 1% of the tokens. An expert that stays dead is wasted capacity and a sign the load-balancing loss is too weak.
- **`expert_utilisation`** — the per-expert load fraction as a plain Python list, recomputed here from the same counts (again `clamp_min(1)`-guarded). This is the list form of the `expert_utilisation` tensor returned in `MoELayerOutput`.

A healthy layer shows high `entropy_mean`, `imbalance_ratio` near 1, `coefficient_of_variation` near 0, `overflow_fraction` near 0, and an empty `dead_experts`.

## Dtype behaviour under autocast

Mixed-precision training wants the *big* matmuls (the expert FFNs) in bfloat16 for speed and memory, but the *small, sensitive* reductions (the losses) in float32 for stability. `MoELayer` gives you exactly that split, and it does so almost for free.

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = layer(hidden)

out.output.dtype    # torch.bfloat16
out.aux_loss.dtype  # torch.float32
out.z_loss.dtype    # torch.float32
```

Why does this happen? The `output` flows through the expert bank's matmuls, which autocast runs in bf16 — so `output` inherits bf16. (Internally the bank *accumulates* its weighted sum in float32 for stability and then casts back to the bf16 compute dtype, so you get stable accumulation but a bf16 result.) The losses, by contrast, are built from operations that autocast keeps in float32 by policy and that the code explicitly promotes: the aux loss takes `router_probs.float().mean(...)`, and the z-loss runs `torch.logsumexp(router_logits.float(), ...)`. A `.float()` inside an autocast region produces float32, and multiplying by the Python scalars `alpha`/`beta` does not demote it. So the regularisers stay in float32 even while the main signal is bf16 — no special-casing in `MoELayer` itself, just careful dtype hygiene in its collaborators.

## Common pitfalls

- **Forgetting the residual.** `out.output` is the FFN output, not `x + FFN(x)`. If you assign `hidden = out.output` you have thrown away the residual stream and training will struggle. Always `hidden = hidden + out.output`.
- **Double-weighting the losses.** `out.aux_loss` and `out.z_loss` are *already* multiplied by `alpha` and `beta`. Do not multiply again; just add them to your task loss.
- **Calling `get_routing_stats()` too early.** Before the first `forward`, it returns `{}`. Guard your logging code against an empty dict.
- **Wrong trailing dimension.** Pass `[*, d_model]`. Any other trailing size raises a `ValueError`. A `[batch, d_model, seq]` tensor (channels-first) will be rejected — transpose first.
- **Expecting fp32 output under autocast.** The *output* is bf16 by design; only the losses are fp32. If a downstream op needs fp32 activations, cast explicitly.

## Exercises

1. **Plug `MoELayer` into a residual block.** Write a tiny `nn.Module` `MoEBlock` that holds a `nn.LayerNorm(d_model)` and a `MoELayer`, and whose `forward(x)` returns `(x + moe(norm(x)).output, aux, z)`. Confirm the output shape equals the input shape for `x` of shape `[4, 16, 32]`.
2. **Read and interpret `get_routing_stats`.** Build a `MoEConfig(d_model=16, d_ff=32, num_experts=8, top_k=2)`, run one forward pass on `torch.randn(4, 32, 16)`, then print `get_routing_stats()`. Which experts (if any) are dead at random initialisation? Is `imbalance_ratio` close to 1?
3. **Confirm bf16 output but fp32 losses.** On CPU you can autocast with `dtype=torch.bfloat16`. Run the layer inside `torch.autocast(device_type="cpu", dtype=torch.bfloat16)` and assert `out.output.dtype == torch.bfloat16` while `out.aux_loss.dtype == torch.float32`.
4. **Verify the weighting.** Manually call `layer.criterion(...)` with a zero task loss and the cached router tensors, then check that `layer.config.alpha * raw_aux` equals the `aux_loss` you got from `forward`.
5. **Empty-cache behaviour.** Construct a fresh `MoELayer` and call `get_routing_stats()` *before* any forward pass. Confirm it returns `{}`.

## Solutions

```python
import torch
from torch import nn
from moe import MoELayer, MoEConfig

cfg = MoEConfig(d_model=32, d_ff=64, num_experts=8, top_k=2)

# 1. MoEBlock with a pre-norm residual.
class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.moe = MoELayer(config)

    def forward(self, x):
        out = self.moe(self.norm(x))
        return x + out.output, out.aux_loss, out.z_loss

block = MoEBlock(cfg)
y, aux, z = block(torch.randn(4, 16, 32))
assert y.shape == (4, 16, 32)

# 2. Read routing stats.
layer = MoELayer(MoEConfig(d_model=16, d_ff=32, num_experts=8, top_k=2))
_ = layer(torch.randn(4, 32, 16))
stats = layer.get_routing_stats()
print(stats["dead_experts"], round(stats["imbalance_ratio"], 3))

# 3. bf16 output, fp32 losses.
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    out = layer(torch.randn(4, 32, 16))
assert out.output.dtype == torch.bfloat16
assert out.aux_loss.dtype == torch.float32
assert out.z_loss.dtype == torch.float32

# 4. Verify the alpha-weighting.
out = layer(torch.randn(4, 32, 16))
loss_out = layer.criterion(
    layer._last_router_probs.new_zeros(()),
    layer._last_router_probs,                  # cached, detached probs
    # NOTE: indices/logits below are illustrative; in a real check, capture the
    # RouterOutput from a single forward and reuse its fields directly.
    layer._last_expert_counts.new_zeros((layer._last_router_probs.size(0), 1)),
    layer._last_router_probs.log(),
)
# raw_aux * alpha matches the weighted aux returned by forward:
assert torch.allclose(layer.config.alpha * loss_out.aux_loss, out.aux_loss, atol=1e-4)

# 5. Empty cache before any forward.
fresh = MoELayer(cfg)
assert fresh.get_routing_stats() == {}
```

Exercise 4's cleanest form is to capture the `RouterOutput` once (e.g. by calling `layer.router(x_flat)` yourself) and feed its real `expert_indices` and `router_logits` into `layer.criterion`; the snippet above only sketches the idea, since `forward` does not expose the per-call `RouterOutput`.

## Key takeaways

- `MoELayer` is a conductor: it owns a router (via `build_router`), an `ExpertBank`, and a `MoELoss`, and sequences them into a drop-in FFN.
- `forward` flattens `[B, T, d] → [B·T, d]`, routes, dispatches (the bank does the weighted sum), reshapes back, and computes weighted aux/z losses with a *zero* task loss — because the layer does not know your task.
- You wire it up in two lines: `residual = x + out.output` and `loss = task_loss + out.aux_loss + out.z_loss`. The aux/z losses arrive pre-weighted by `alpha`/`beta`.
- `get_routing_stats()` reads cached `_last_router_probs` / `_last_expert_counts` plus the bank's overflow counters to report entropy, imbalance, overflow, dead experts, and utilisation — empty until the first forward.
- Under bf16 autocast the output is bf16 while the losses are float32, thanks to explicit `.float()` promotion and autocast's fp32 policy on the reductions.

## Next → `07-monitoring.md`

We have seen `get_routing_stats()` produce a dictionary of health metrics. In the next lesson we open up `moe/utils.py` and study those diagnostics in depth — entropy, load imbalance, dead-expert detection, the routing heatmap, and exporting stats to JSONL — so you can build a real monitoring dashboard for an MoE training run.
