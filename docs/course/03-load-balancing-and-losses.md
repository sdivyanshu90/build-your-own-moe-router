# Lesson 03 — Load Balancing & the Auxiliary Losses

> A sparse router will quietly cannibalize itself unless you add two small auxiliary losses; this lesson builds both of them, line by line, from `moe/losses.py`.

## Learning objectives

By the end of this lesson you will be able to:

- Explain **expert collapse** as a rich-get-richer feedback loop and recognize its empirical signature (near-zero routing entropy, exploding load-imbalance ratio).
- Read and reproduce `auxiliary_load_balancing_loss`, including the **detach/attach trick** that lets a non-differentiable counter shape a differentiable loss.
- Derive the two reference values of the auxiliary loss — `alpha` at perfect balance, `alpha * N` at full collapse — and explain why the Switch `N`-factor makes the balanced value clean.
- Read and reproduce `router_z_loss`, and say precisely why `torch.logsumexp` is used instead of `log(sum(exp(...)))`.
- Wire both terms together with a task loss through the `MoELoss` module, read its `LossOutput`, build it with `from_config`, and emit `log_metrics` (including the `aux_fraction` early-warning signal).

## Prerequisites

- Lessons 01–02 (you know what a router produces: per-token logits, a softmax distribution, and discrete top-K expert indices).
- Comfort with PyTorch tensors, broadcasting, and `loss.backward()`.
- The idea of a softmax and that `argmax`/`topk` are **not** differentiable.
- Optional but useful: skim `docs/moe_routing.md` §1.3, which states the same formulas this code implements.

---

## 1. The problem: experts collapse on their own

Recall the one fact that makes MoE both efficient and fragile: **only the experts a token is routed to receive a gradient on that token.** An expert that is never selected is never trained, and an expert that is never trained never becomes worth selecting.

That asymmetry creates a feedback loop. Suppose, purely by initialization luck, a couple of experts are marginally preferred. They get slightly more tokens, hence more gradient, hence they improve faster, hence the router prefers them *even more* — and the neglected experts starve. Left alone, the router funnels nearly all traffic into a small clique and the rest of your expensive parameters become dead weight. You paid for `N` experts and got a dense model wearing a costume. This is **expert collapse**, and its dynamics are *rich-get-richer*.

How do you *see* it? The cleanest signal is the **routing entropy**

$$ H = -\sum_{i=1}^{N} p_i \log p_i $$

where $N$ is the number of experts and $p_i$ is the batch-averaged routing probability for expert $i$. A healthy router spreads mass and keeps $H$ somewhere near $\log N$; a collapsing router concentrates mass on one or two experts and drives $H$ toward $0$. A second, blunter signal is the load-imbalance ratio $\max_i f_i / \mathrm{mean}_i f_i$, where $f_i$ is the fraction of tokens dispatched to expert $i$; it sits near $1$ when healthy and climbs well above it under collapse. (The integration test `test_routing_collapses_without_aux_loss` literally trains with `alpha=0.0` and asserts this ratio grows past `2.0`.)

The fix is to add a differentiable penalty that nudges the router toward uniform usage. That penalty is the **auxiliary load-balancing loss**, and a companion **router z-loss** keeps the logits numerically sane. Both live in `moe/losses.py`.

---

## 2. `auxiliary_load_balancing_loss`: the anti-collapse penalty

### 2.1 Intuition

We want a single scalar that is *small when load is balanced* and *large when it is lopsided*, and whose gradient pushes probability mass off crowded experts and onto starving ones. The Switch Transformer answer is to multiply two per-expert quantities and sum them:

- $f_i$ — the **fraction of dispatch slots** that actually landed on expert $i$. This comes from the hard top-K assignment, so it is a *count*. Counts come from `argmax`/`topk`, which are non-differentiable, so we treat $f_i$ as a constant.
- $P_i$ — the **mean softmax probability** the router assigned to expert $i$, averaged over the batch. This is smooth and differentiable in the router weights.

### 2.2 The math

$$ L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i $$

where $N$ is the number of experts, $\alpha$ is the loss coefficient, $f_i \in [0,1]$ is the detached dispatch fraction for expert $i$ (with $\sum_i f_i = 1$), and $P_i \in [0,1]$ is the differentiable mean routing probability for expert $i$ (with $\sum_i P_i = 1$). The dot product $\sum_i f_i P_i$ is the load-imbalance signal: it is *minimized* when both vectors are uniform and *maximized* when both pile onto the same expert.

**Derive the two reference values.** They are worth memorizing because the unit tests assert them exactly.

- *Perfect balance:* every $f_i = 1/N$ and every $P_i = 1/N$, so $\sum_i f_i P_i = N \cdot \tfrac{1}{N}\cdot\tfrac{1}{N} = \tfrac{1}{N}$, and $L_{\text{aux}} = \alpha \cdot N \cdot \tfrac{1}{N} = \alpha$.
- *Full collapse:* one expert $j$ takes everything, so $f_j = P_j = 1$ and all other terms vanish, giving $\sum_i f_i P_i = 1$ and $L_{\text{aux}} = \alpha \cdot N \cdot 1 = \alpha N$.

So the loss ranges from $\alpha$ (balanced) up to $\alpha N$ (collapsed). The explicit factor of $N$ is exactly what makes the *balanced* value a clean, $N$-independent $\alpha$ — convenient when you retune across different expert counts.

> **Honest note — this is a derivation, not a typo.** This codebase uses the canonical Switch Transformer form *with* the $N$ factor. That is why `tests/test_losses.py::test_aux_loss_perfect_balance` asserts the loss equals `ALPHA`, and `test_aux_loss_collapsed` asserts it equals `ALPHA * num_experts`. If you have seen the "mean balance" form elsewhere whose balanced value is $1/N$, you are not confused — that form simply omits the $N$. Both are in the literature; this library commits to the Switch convention, top to bottom.

### 2.3 The detach/attach trick (the heart of the lesson)

Here is the part that looks like cheating but is the whole point. We deliberately backpropagate through $P_i$ **only**, while $f_i$ is frozen as a constant weight:

$$ \frac{\partial L_{\text{aux}}}{\partial W_g} = \alpha N \sum_{i=1}^{N} f_i \, \frac{\partial P_i}{\partial W_g} $$

where $W_g$ is the router (gating) projection. Read $f_i$ as *"how crowded expert $i$ currently is"* — a fixed per-expert weight on how hard we push its probability $P_i$ down. Over-used experts have large $f_i$, so the optimizer leans hard on shrinking their $P_i$; starving experts have small $f_i$ and are barely touched. Minimizing $\sum_i f_i P_i$ therefore *moves mass off the crowded experts and onto the empty ones* — precisely the balancing we want — and it never has to differentiate the discrete `argmax` that produced the assignments. The non-differentiable information (where the crowding is) enters through $f_i$; the gradient flows entirely through the differentiable $P_i$.

### 2.4 The code

```python
def auxiliary_load_balancing_loss(
    router_probs: Tensor,
    expert_indices: Tensor,
    num_experts: int,
    alpha: float = DEFAULT_WEIGHT,
) -> Tensor:
    if router_probs.dim() != 2:
        raise ValueError(...)               # must be [num_tokens, num_experts]
    if router_probs.size(-1) != num_experts:
        raise ValueError(...)               # expert dim must match N

    # P_i: mean gating probability per expert, differentiable. float32 for a
    # stable mean under bf16/fp16 autocast.
    mean_prob = router_probs.float().mean(dim=0)          # [num_experts]

    # f_i: fraction of dispatch slots per expert. bincount counts integer expert
    # ids; we DETACH (non-differentiable counter) and normalise so sum_i f_i == 1.
    flat_idx = expert_indices.reshape(-1).detach().to(torch.long)
    counts = torch.bincount(flat_idx, minlength=num_experts).to(mean_prob.dtype)
    dispatch_fraction = counts / counts.sum().clamp_min(1.0)   # [num_experts]

    # L_aux = alpha * N * <f, P>. The dot product is the load-imbalance signal.
    loss = alpha * num_experts * torch.sum(dispatch_fraction * mean_prob)
    return loss
```

Walk the four ideas:

- `mean_prob = router_probs.float().mean(dim=0)` is $P_i$ — averaged over the token axis, kept differentiable, cast to float32 so the batch mean does not lose precision under mixed-precision autocast.
- `.reshape(-1).detach()` flattens the `[num_tokens, top_k]` index tensor and **cuts it off the graph**. This is the "detach" half of the trick — no gradient will ever flow back through the indices.
- `torch.bincount(..., minlength=num_experts)` turns the integer expert ids into per-expert counts; dividing by `counts.sum()` makes them fractions. `clamp_min(1.0)` guards the degenerate empty-batch case so you never divide by zero.
- The final line multiplies the constant `dispatch_fraction` by the differentiable `mean_prob` — the "attach" half. Only `mean_prob` carries grad, so `loss.backward()` updates the router exactly as the derivative above predicts.

The test `test_moeloss_gradient_only_through_probs` proves this: it derives `probs` from one leaf and `indices` from a *separate* leaf, calls backward, and asserts the probs' leaf has a gradient while the index leaf's `.grad is None`.

---

## 3. `router_z_loss`: keeping the logits sane

### 3.1 Intuition

Nothing stops a router from learning enormous logits — they make the softmax razor-sharp and overconfident, and they overflow `exp` in low precision. The z-loss is a gentle leash: penalize the **log-partition function** of each token's logits so they stay small, the softmax stays in range, and the router stays a little humble.

### 3.2 The math

$$ L_z = \beta \cdot \frac{1}{B} \sum_{b=1}^{B} \Bigl( \log \sum_{i=1}^{N} e^{z_{b,i}} \Bigr)^2 $$

where $B$ is the number of tokens, $N$ the number of experts, $z_{b,i}$ the pre-softmax gating logit of token $b$ for expert $i$, and $\beta$ the z-loss coefficient. The inner quantity $\log \sum_i e^{z_{b,i}}$ is the log-sum-exp (the softmax's normalizer); squaring and averaging pushes the logits toward smaller magnitude.

**Reference value.** With all logits zero, $\sum_i e^0 = N$, so the inner term is $\log N$, and $L_z = \beta (\log N)^2$. That is exactly what `test_z_loss_at_origin` asserts.

### 3.3 Why `logsumexp` and not `log(sum(exp))`

Compute $\log \sum_i e^{z_i}$ naively and any logit above ~88 overflows `exp` to `inf` in float32 — and then `log(inf) = inf` poisons the whole step. `torch.logsumexp` uses the shift identity

$$ \log \sum_i e^{z_i} = m + \log \sum_i e^{\,z_i - m}, \qquad m = \max_i z_i $$

so the largest exponent becomes $e^0 = 1$ and nothing overflows; underflows in the small terms vanish harmlessly. The result is *mathematically identical* — the $m$ cancels — just overflow-free. The test `test_z_loss_numerical_stability` slams the function with logits of magnitude `1e4` and asserts the result is still finite.

### 3.4 The code

```python
def router_z_loss(router_logits: Tensor, beta: float = DEFAULT_WEIGHT) -> Tensor:
    if router_logits.dim() != 2:
        raise ValueError(...)               # must be [num_tokens, num_experts]
    # logsumexp over experts -> per-token log-partition, float32 for stability.
    log_z = torch.logsumexp(router_logits.float(), dim=-1)   # [num_tokens]
    return beta * torch.mean(log_z**2)
```

Because it is a mean of squares, the result is always $\ge 0$ (`test_z_loss_non_negative`). Note both losses default `alpha`/`beta` to `DEFAULT_WEIGHT = 1.0` so that, called bare, they return the **raw** unweighted term — the `MoELoss` module applies the real weights itself.

---

## 4. `MoELoss`: combining everything

The two functions are easy to unit-test in isolation, but training wants a single criterion. `MoELoss` is an `nn.Module` that holds `alpha`, `beta`, `num_experts` and combines the task loss with the two raw auxiliaries:

$$ L_{\text{total}} = L_{\text{task}} + \alpha \cdot L_{\text{aux}}^{\text{raw}} + \beta \cdot L_z^{\text{raw}} $$

```python
def forward(self, task_loss, router_probs, expert_indices, router_logits):
    raw_aux = auxiliary_load_balancing_loss(
        router_probs, expert_indices, self.num_experts, alpha=DEFAULT_WEIGHT)
    raw_z = router_z_loss(router_logits, beta=DEFAULT_WEIGHT)
    total = task_loss + self.alpha * raw_aux + self.beta * raw_z
    # cache detached floats for logging (never touches autograd) ...
    return LossOutput(total, raw_aux, raw_z, task_loss)
```

Key design choices:

- It calls the pure functions with the **default weight 1.0**, gets back the *raw* terms, and applies `self.alpha` / `self.beta` itself. That way it can report the raw terms (great for monitoring) independently of the weights.
- The return type is `LossOutput`, a `NamedTuple` of `(total_loss, aux_loss, z_loss, task_loss)` where `aux_loss` and `z_loss` are the **raw** terms. `test_moeloss_total_is_sum_of_parts` re-derives `task + ALPHA*aux + BETA*z` from these fields and asserts it matches `total_loss`.
- `from_config(config)` is a classmethod that pulls `alpha`, `beta`, and `num_experts` straight off an `MoEConfig`, so your one config dataclass stays the single source of truth.
- The constructor validates: negative `alpha`/`beta` or `num_experts < 1` raise `ValueError`.

### 4.1 `log_metrics`: the early-warning dashboard

After each `forward`, `MoELoss` caches detached Python floats; `log_metrics(step)` returns them as a flat dict for W&B / TensorBoard:

```python
def log_metrics(self, step: int) -> dict[str, float | int]:
    metrics = {"step": step}
    metrics.update(self._last)
    if "loss/total" in self._last and self._last["loss/total"] != 0.0:
        metrics["loss/aux_fraction"] = (
            self._last["loss/aux_weighted"] / self._last["loss/total"])
    return metrics
```

The standout is **`loss/aux_fraction`** = $\alpha L_{\text{aux}} / L_{\text{total}}$, the share of your total loss coming from the balancer. The auxiliary loss is a *means*, not an *end* — it should nudge balance without fighting the task. If `aux_fraction` climbs above roughly 5%, `alpha` is too high and is degrading quality; lower it. If it sits near zero while the imbalance ratio rises, `alpha` is too low; raise it. Called before any `forward`, `log_metrics` returns just `{"step": step}` (asserted in `test_moeloss_init_validation_and_helpers`).

---

## Common pitfalls

- **Forgetting to detach the indices.** If you build $f_i$ from a tensor still on the graph, you change the gradient and break the balancing logic. The code's `.detach()` is load-bearing, not decorative.
- **Passing probabilities where logits go (or vice versa).** `auxiliary_load_balancing_loss` wants the **softmax probs**; `router_z_loss` wants the **pre-softmax logits**. Swapping them silently computes nonsense.
- **Expecting the balanced loss to be `1/N`.** With the Switch $N$-factor the balanced value is `alpha`, not `1/N`. Re-derive §2.2 if it surprises you.
- **Reaching for `log(sum(exp(...)))`.** It overflows above ~88 in float32. Always `torch.logsumexp`.
- **Setting `alpha` huge to "force" balance.** Watch `aux_fraction`: past ~5% you are training a balancer instead of a model.
- **Reading `LossOutput.aux_loss` as already-weighted.** It is the *raw* term; multiply by `alpha` yourself if you want the weighted contribution (or read `loss/aux_weighted`).

## Exercises

1. **Predict a collapsed loss.** With `num_experts=4` and `alpha=1e-2`, construct `probs` that put all mass on expert 0 and matching all-zero `indices`. Predict `L_aux` by hand, then verify.
2. **Half-and-half balance.** With `num_experts=4`, route every token to experts 0 and 1 only (each gets 50%), with `probs` uniform `0.25` everywhere. Predict $\sum_i f_i P_i$ and `L_aux` at `alpha=1.0`.
3. **z-loss stays finite at 1e4.** Build `logits = torch.full((4, 8), 1e4)`, compute `router_z_loss`, and confirm it is finite. Then compute the naive `torch.log(torch.exp(logits).sum(-1))` and observe what happens.
4. **Prove the detach.** Make `prob_logits` and `index_source` two separate leaves with `requires_grad=True`, derive `probs` from one and `indices = index_source.topk(2, -1).indices` from the other, run `auxiliary_load_balancing_loss(...).backward()`, and check which leaf got a gradient.
5. **Read `aux_fraction`.** Build `MoELoss(alpha=0.5, beta=1e-3, num_experts=4)`, push a deliberately collapsed batch through `forward`, then inspect `log_metrics(0)["loss/aux_fraction"]`. Is `alpha` too high?

## Solutions

1. Collapse ⇒ $\sum_i f_i P_i = 1$, so $L_{\text{aux}} = 0.01 \cdot 4 \cdot 1 = 0.04$.
   ```python
   import torch
   from moe.losses import auxiliary_load_balancing_loss
   probs = torch.zeros(8, 4); probs[:, 0] = 1.0
   idx = torch.zeros(8, 1, dtype=torch.long)
   print(float(auxiliary_load_balancing_loss(probs, idx, 4, alpha=1e-2)))  # 0.04
   ```
2. $f = (0.5, 0.5, 0, 0)$, $P = (0.25, 0.25, 0.25, 0.25)$, so $\sum_i f_i P_i = 0.5\cdot0.25 + 0.5\cdot0.25 = 0.25$, and $L_{\text{aux}} = 1.0 \cdot 4 \cdot 0.25 = 1.0$. (More balanced *dispatch* than expert-0-only, but `probs` are uniform, so it is not minimal — minimal needs both $f$ and $P$ uniform.)
3. `router_z_loss` returns a finite number; the naive version overflows `exp(1e4)` to `inf` and yields `inf`. `torch.logsumexp` shifts by the row max first.
   ```python
   from moe.losses import router_z_loss
   logits = torch.full((4, 8), 1e4)
   print(torch.isfinite(router_z_loss(logits, beta=1e-3)))            # tensor(True)
   print(torch.log(torch.exp(logits).sum(-1)))                        # inf
   ```
4. Only the `probs` leaf gets a gradient; `index_source.grad is None`, because the indices are detached inside the loss (and `topk` is non-differentiable anyway). This is exactly `test_moeloss_gradient_only_through_probs`.
5. With a collapsed batch the weighted aux term is large relative to a small task loss, so `aux_fraction` will be well above 0.05 — `alpha=0.5` is far too high for production; it would dominate the objective. The realistic range is `alpha` in `[0.001, 0.01]`.

## Key takeaways

- Sparse routers collapse on their own (rich-get-richer); the symptom is routing entropy crashing toward 0 and the imbalance ratio climbing past ~2.
- `auxiliary_load_balancing_loss` implements $L_{\text{aux}} = \alpha N \sum_i f_i P_i$ with a **detached** count $f_i$ (via `bincount`) and a **differentiable** mean prob $P_i$; the gradient flows only through $P_i$, weighted by how crowded each expert is.
- The Switch $N$-factor makes balanced ⇒ `alpha` and collapsed ⇒ `alpha * N` — a derivation the unit tests pin exactly.
- `router_z_loss` implements $L_z = \beta\,\mathrm{mean}((\mathrm{logsumexp}\,z)^2)$ with `torch.logsumexp` for overflow-free stability; zero logits give $\beta(\log N)^2$.
- `MoELoss` combines `task + alpha*aux + beta*z`, returns the raw terms in a `LossOutput` NamedTuple, builds via `from_config`, and exposes `log_metrics` whose `aux_fraction` warns you when `alpha` is too high.

## Next → `04-experts.md`

You now have a router that *stays balanced*. Next we build the things it routes to: the expert feed-forward networks themselves — how they are constructed, batched, and dispatched.
