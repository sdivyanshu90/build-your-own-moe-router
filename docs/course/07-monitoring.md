# Lesson 07 — Monitoring, Diagnostics & Debugging

> A sparse MoE can be quietly destroying itself while its loss curve looks perfectly healthy — `moe/utils.py` gives you five small instruments that make routing collapse visible *before* it ruins a run.

## Learning objectives

By the end of this lesson you will be able to:

- Explain *why* MoE training needs routing-health metrics that the loss curve cannot show you.
- Compute routing entropy with `compute_routing_entropy` and read its healthy band (near `log K` … `log N`) versus collapse (near `0`).
- Quantify load imbalance with `compute_load_imbalance` and interpret the imbalance ratio and coefficient of variation.
- Flag starving experts with `detect_dead_experts` and apply the two standard remedies.
- Build a tokens×experts heatmap with `visualise_routing` and stream per-step metrics to disk with `export_routing_stats`.
- Assemble all of these into a "monitoring playbook" — which numbers to log every step and what each one warns you about.

## Prerequisites

- Comfort with Python, basic tensors, and `NamedTuple`.
- Lessons 01–03 (the MoE picture: `N` experts, top-`K` routing, the router producing a softmax distribution per token).
- A passing acquaintance with **expert collapse** — the rich-get-richer failure mode described in `docs/moe_routing.md` §1.3. We will *measure* it here.

All code below lives in `moe/utils.py`. Open it alongside this lesson. These are deliberately *standalone* helper functions — they take raw router outputs (probability tensors, per-expert counts) and return health metrics. Nothing here depends on the rest of the library, so you can drop them into any training loop.

## Why you must watch routing health

Here is the uncomfortable truth that organizes this whole lesson: **the training loss will not tell you that your router has collapsed.** If the router funnels almost every token into two of your eight experts, the model still trains — it just behaves like a smaller dense model that paid for capacity it cannot use. The cross-entropy keeps falling, the curve looks fine, and six experts quietly rot.

Collapse is a positive feedback loop. Only *selected* experts receive gradients, so an expert that wins a few extra tokens early improves faster, gets preferred more strongly, wins even more tokens, and starves its neighbours. Nothing in the loss objects to this. The only way to catch it is to instrument the router directly and watch a handful of numbers, each with a *known healthy band*. The module docstring names exactly these:

```python
"""Monitoring, diagnostics and visualisation utilities for MoE routing.

These standalone helpers turn raw router outputs into the health metrics every
MoE training run should log: routing entropy, load imbalance, dead-expert
detection, a routing heatmap and a JSONL exporter for offline analysis.
"""
```

Let us take them one at a time, intuition first.

## Routing entropy: is the router spreading its bets?

**Intuition.** Each token produces a routing distribution `p` over the `N` experts. If that distribution is *uniform*, the router is undecided and spreads its mass everywhere; if it is *one-hot*, the router has committed all its mass to a single expert. Entropy is the standard one-number measure of "how spread out" a distribution is. High entropy = healthy diversity; entropy crashing toward zero = collapse in progress.

**Math.** The Shannon entropy of a token's routing distribution, in *nats* (natural log), is

$$
H = -\sum_{i=1}^{N} p_i \log p_i,
$$

where `N` is the number of experts and $p_i$ is the routing probability the token assigns to expert `i`. Two reference points anchor the scale. A uniform distribution over `N` experts gives the maximum, $H = \log N$. A one-hot distribution gives the minimum, $H = 0$ (fully collapsed). Because a sparse top-`K` router concentrates mass on its `K` chosen experts, a *healthy* router lives somewhere between $\log K$ and $\log N$: low enough to specialize, high enough not to ignore most of the experts. For `N = 8`, `K = 2`, that is roughly $[\,0.69,\ 2.08\,]$ — and an `H` sliding toward `0` is your collapse alarm.

**Code.** The implementation in `compute_routing_entropy` is short, but one line deserves a close look:

```python
p = router_probs.float()
per_token = torch.special.entr(p).sum(dim=-1)  # [num_tokens], 0*log0 handled
return EntropyStats(per_token=per_token, mean=per_token.mean(), min=per_token.min())
```

Why `torch.special.entr` instead of the obvious `-(p * p.log()).sum(-1)`? Because at $p_i = 0$ the naive form computes `0 * log(0) = 0 * (-inf) = NaN`, and a one-hot row (the *most important* case to measure) is full of exact zeros. `special.entr` computes $-p\log p$ with the mathematically correct value `0` at `p = 0`, so it never poisons your metric with a `NaN`. This is a small numerical-safety choice that you would otherwise discover the hard way.

The function returns an `EntropyStats` named tuple with three fields, and the choice of three is deliberate:

```python
class EntropyStats(NamedTuple):
    per_token: Tensor  # [num_tokens] entropy of each token's routing distribution
    mean: Tensor       # scalar mean entropy over the batch
    min: Tensor        # scalar minimum entropy (the most collapsed token)
```

Log `mean` as your headline trend line. But also watch `min`: it surfaces *the single most collapsed token* in the batch. A healthy mean can hide a subpopulation of tokens that are all being hard-routed to one expert, and `min` is what catches that before it spreads. The function raises `ValueError` if `router_probs` is not 2-D `[num_tokens, num_experts]`, so feed it a flattened batch.

## Load imbalance: is the work shared fairly?

**Intuition.** Entropy looks at the router's *probabilities*. Imbalance looks at the *actual dispatch* — after the hard top-`K` decision, how many tokens did each expert really receive? Even a reasonable-looking probability distribution can dispatch lopsidedly. We want one expert's share to be close to its fair share of `1/N`.

**Math.** Given per-expert token counts $c_i$ with fractions $f_i = c_i / \sum_j c_j$, `compute_load_imbalance` returns two numbers. The **imbalance ratio** is

$$
R = \frac{\max_i f_i}{\operatorname{mean}_i f_i} = \frac{\max_i c_i}{\operatorname{mean}_i c_i},
$$

the busiest expert's load divided by the average load. Perfect balance gives $R = 1.0$; the practical acceptable ceiling is about `1.5` (the busiest expert handles at most 50% more than its fair share). The **coefficient of variation** is

$$
\mathrm{CV} = \frac{\operatorname{std}_i c_i}{\operatorname{mean}_i c_i},
$$

the standard deviation of the counts divided by their mean. It is `0.0` at perfect balance and grows without an upper bound as load concentrates. The ratio answers "how bad is the *worst* expert?"; the CV answers "how uneven is the *whole* distribution?".

**Code.** Two implementation details are worth calling out:

```python
counts = expert_counts.float()
mean = counts.mean()
# Guard against an all-zero batch (no tokens routed at all).
safe_mean = mean.clamp_min(torch.finfo(counts.dtype).tiny)
imbalance = counts.max() / safe_mean
# Population standard deviation (unbiased=False) so a single expert gives 0.
cv = counts.std(unbiased=False) / safe_mean
return ImbalanceStats(imbalance_ratio=imbalance, coefficient_of_variation=cv)
```

First, `clamp_min(... .tiny)` protects against a division by zero when *no* tokens were routed at all (an all-zero count vector) — the metric degrades gracefully instead of returning `inf`/`NaN`. Second, `std(unbiased=False)` uses the *population* standard deviation; with the unbiased (`N-1`) version a single-expert input would divide by zero and break, whereas the population form correctly reports `CV = 0` for a degenerate one-expert case. The result is an `ImbalanceStats` named tuple, and `compute_load_imbalance(torch.tensor([4, 4, 4, 4]))` returns exactly `(1.0, 0.0)` — perfect balance. The function raises `ValueError` on anything that is not a non-empty 1-D tensor.

## Dead-expert detection: who has stopped learning?

**Intuition.** Imbalance tells you the distribution is skewed; dead-expert detection names the *specific experts* that have effectively stopped receiving tokens. An expert below ~1% utilization, sustained over many steps, is no longer being trained — its parameters are frozen dead weight.

**Code.** `detect_dead_experts` is a thresholded fraction check returning the offending indices:

```python
counts = expert_counts.float()
total = counts.sum().clamp_min(1.0)  # avoid div-by-zero; counts are >= 0
fractions = counts / total
dead = torch.nonzero(fractions < threshold, as_tuple=False).flatten()
return [int(i) for i in dead.tolist()]
```

The default `threshold=0.01` (1%) matches the convention in §1.7. An expert whose share is *strictly below* the threshold is reported, so `detect_dead_experts(torch.tensor([10, 10, 10, 0]))` returns `[3]`. The threshold is validated to lie in `[0, 1]`, and `total` is clamped so an empty batch cannot divide by zero. The return is a plain sorted Python `list[int]`, which is exactly what you want for logging ("dead experts: [3, 6]") or for triggering remediation.

**Remediation.** Once you have the indices, §1.7 prescribes two standard cures. (1) **Reinitialize the dead expert's weights with small random noise**, so it becomes a fresh, slightly-different function the router can rediscover. (2) **Temporarily reduce `K`** (or briefly raise the exploration noise) so routing decisions reshuffle and the starving expert gets a chance to win tokens and climb back into use. Combined with a sufficiently large `alpha`, these keep the whole expert population alive.

## Visualising routing: the tokens×experts heatmap

Numbers tell you *that* something is wrong; a picture tells you *where*. `visualise_routing` renders the full `[num_tokens, num_experts]` probability matrix as a heatmap — rows are tokens, columns are experts, colour is probability:

```python
fig = MplFigure(figsize=(max(4, num_experts), max(3, num_tokens * 0.3)))
ax = fig.add_subplot(111)
image = ax.imshow(probs, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
ax.set_xlabel("Expert")
ax.set_ylabel("Token")
```

**How to read it.** A *healthy* router shows colour scattered across many columns — different tokens lighting up different experts. *Collapse* shows up as one or two bright vertical stripes (every token pouring mass into the same expert columns) with the rest of the grid dark. Because `vmin=0.0, vmax=1.0` fix the colour scale, the brightness is comparable across snapshots, so you can flip through heatmaps over training and literally watch stripes form.

Two engineering choices matter. First, the figure is built with the **object-oriented `matplotlib.figure.Figure`, not `pyplot`** — `pyplot` keeps global state and is not safe to call off the main thread, so building the `Figure` directly makes `visualise_routing` safe in a background logging thread and in headless environments (no display required). Second, matplotlib is imported *locally inside the function* and is an optional dependency: if it is missing you get a clear `ImportError` telling you to `pip install matplotlib`, rather than the whole `utils` module failing to import. You may pass optional `tokens` text labels for the y-axis; their length must equal `num_tokens` or you get a `ValueError`.

## Exporting stats: a JSONL trace for offline analysis

You will not eyeball metrics every step. The right pattern is to *append* a record per step to a file and analyze it later. `export_routing_stats` writes [JSON Lines](https://jsonlines.org/) — one JSON object per line — which is trivially streamable and re-loadable:

```python
def _to_serialisable(value: Any) -> Any:
    """Convert tensors/numpy scalars to plain JSON-friendly Python values."""
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    return value

with out_path.open("w", encoding="utf-8") as handle:
    for i, record in enumerate(stats_list):
        if not isinstance(record, dict):
            raise TypeError(...)
        clean = {k: _to_serialisable(v) for k, v in record.items()}
        handle.write(json.dumps(clean) + "\n")
```

The convenience here is the `_to_serialisable` helper: raw metric dicts usually contain *tensors* (your entropy mean, your counts), and `json.dumps` cannot serialize a tensor. The helper transparently `.detach().cpu().tolist()`s any tensor — moving it off the GPU and detaching it from the graph — so you can pass `{"step": s, "entropy": stats.mean, ...}` straight from the training loop with no manual conversion. Parent directories are created automatically, each non-dict element raises a `TypeError` (so a malformed record fails loudly), and the resolved `Path` is returned. Re-loading is the standard JSONL idiom: read the file line by line and `json.loads` each line.

## The monitoring playbook

Pulling it together, here is the tight diagnostic loop §1.7 recommends — log these every few steps, and know what each one warns you about:

| Metric | How to get it | Healthy | Warns you about |
|---|---|---|---|
| Routing entropy (`mean`, `min`) | `compute_routing_entropy` | `~log K … log N` | Mean → 0 = collapse; low `min` = a collapsing subpopulation |
| Imbalance ratio + CV | `compute_load_imbalance` | ratio `< ~1.5`, CV small | Ratio climbing past ~2 = balance failing, `alpha` too low |
| Dead experts | `detect_dead_experts` | empty list | Named experts that have stopped training |
| Overflow fraction | (tokens dropped) / (total tokens) | a few % | High = `capacity_factor` too low for current imbalance |
| Aux-loss fraction | `L_aux / L_total` | `< ~5%` | Too high = `alpha` hurting quality; ~0 while imbalance rises = `alpha` too low |

The logic of the loop: entropy and the imbalance ratio tell you *whether* the router is healthy; the heatmap and dead-expert detector tell you *where* it is failing; the overflow fraction tells you whether capacity is the bottleneck; and the aux-loss fraction tells you whether your cure (a big `alpha`) has become worse than the disease. Watching all of these is what turns MoE training from a fragile art into a controllable engineering process.

## Common pitfalls

- **Trusting the loss curve.** It can fall smoothly while six of eight experts are dead. The loss is *not* a routing-health monitor; these utilities are.
- **Computing entropy with `p * p.log()`.** A one-hot row gives `NaN`. Use `compute_routing_entropy`, which relies on `torch.special.entr` for the correct `0·log0 = 0`.
- **Watching only the entropy mean.** A healthy mean hides collapsed subpopulations — always log `min` too.
- **Confusing probabilities with dispatch.** Entropy measures the router's *probabilities*; imbalance measures *actual token counts* after the hard top-`K`. They can disagree, and you need both.
- **Using `pyplot` in a logging thread.** It is not thread-safe. `visualise_routing` builds a `Figure` directly precisely so it works headless and off-thread.
- **Trying to `json.dumps` raw tensors.** It fails. `export_routing_stats` converts tensors for you via `.tolist()`/`detach().cpu()`.
- **Feeding wrong shapes.** `compute_routing_entropy` and `visualise_routing` require 2-D `[num_tokens, num_experts]`; `compute_load_imbalance` and `detect_dead_experts` require a non-empty 1-D count tensor. Anything else raises `ValueError`.

## Exercises

1. **Uniform vs one-hot entropy.** With `N = 8`, build a uniform distribution and a one-hot distribution, run each through `compute_routing_entropy`, and confirm the means are `log(8) ≈ 2.079` and `0.0`. Why does the one-hot case *not* return `NaN`?
2. **Spot the imbalance.** Compute `compute_load_imbalance` for balanced counts `[100, 100, 100, 100]` and skewed counts `[280, 100, 10, 10]`. Interpret both the ratio and the CV. Is the skewed case over the acceptable `1.5` ratio?
3. **Detect a dead expert.** Build a count vector for 6 experts where one expert gets `< 1%` of tokens, and confirm `detect_dead_experts` reports exactly that index. Then raise the `threshold` to `0.2` and watch the dead list grow.
4. **Export and re-load a JSONL trace.** Build a list of three per-step stat dicts (each containing at least one *tensor* value), write them with `export_routing_stats`, then re-load the file line by line with `json.loads` and reconstruct the entropy series.
5. **(Stretch) Watch collapse form.** Make three probability matrices that interpolate from near-uniform to near-one-hot, render each with `visualise_routing`, and save the figures. Describe how the stripes appear as entropy falls.

## Solutions

```python
import json
import torch
from moe.utils import (
    compute_routing_entropy, compute_load_imbalance,
    detect_dead_experts, visualise_routing, export_routing_stats,
)

# 1) Uniform vs one-hot entropy
uniform = torch.full((4, 8), 1.0 / 8)          # 4 tokens, uniform over 8 experts
onehot = torch.eye(8)[[0, 1, 2, 3]]            # 4 one-hot tokens
print(float(compute_routing_entropy(uniform).mean))  # ~2.0794 == log(8)
print(float(compute_routing_entropy(onehot).mean))   # 0.0
# No NaN: torch.special.entr defines -p*log(p) = 0 at p = 0, so the exact
# zeros in a one-hot row contribute 0 instead of 0 * -inf.

# 2) Spot the imbalance
bal = compute_load_imbalance(torch.tensor([100, 100, 100, 100]))
skew = compute_load_imbalance(torch.tensor([280, 100, 10, 10]))
print(float(bal.imbalance_ratio), float(bal.coefficient_of_variation))   # 1.0, 0.0
print(float(skew.imbalance_ratio), float(skew.coefficient_of_variation)) # ~2.8, ~1.1
# Mean count = 100; busiest = 280, so ratio = 2.8 — well over the 1.5 ceiling.
# The CV ~1.1 confirms the whole distribution, not just the worst expert, is uneven.

# 3) Detect a dead expert
counts = torch.tensor([500, 500, 500, 500, 500, 3])   # expert 5 gets ~0.12%
print(detect_dead_experts(counts))                    # [5]  (default threshold 0.01)
print(detect_dead_experts(counts, threshold=0.2))     # [5] plus any under 20%

# 4) Export and re-load a JSONL trace
records = [
    {"step": s, "entropy": torch.tensor(2.0 - 0.3 * s), "counts": torch.tensor([4, 4, 4])}
    for s in range(3)
]
path = export_routing_stats(records, "/tmp/moe_trace.jsonl")
series = []
with open(path) as fh:
    for line in fh:
        row = json.loads(line)          # tensors came back as floats / lists
        series.append(row["entropy"])
print(series)                           # [2.0, 1.7, 1.4]

# 5) (Stretch) Watch collapse form
for t, sharpness in enumerate([0.3, 2.0, 12.0]):
    logits = torch.zeros(6, 4)
    logits[:, 0] = sharpness            # push mass onto expert 0 as sharpness grows
    probs = torch.softmax(logits, dim=-1)
    fig = visualise_routing(probs)
    fig.savefig(f"/tmp/routing_{t}.png")
    print(t, float(compute_routing_entropy(probs).mean))
# Entropy falls from ~1.37 toward 0 as a single bright stripe (expert 0) forms
# and the other three columns go dark — the visual signature of collapse.
```

## Key takeaways

- The loss curve is blind to routing collapse; you must instrument the router directly.
- `compute_routing_entropy` (`H = -Σ pᵢ log pᵢ`, via `torch.special.entr` for a safe `0·log0`) is your primary alarm: healthy near `log K … log N`, collapsing toward `0`. Track both `mean` and `min`.
- `compute_load_imbalance` reports the imbalance ratio (`max/mean`, healthy `< ~1.5`) and the population CV (`0` at perfect balance); `(1.0, 0.0)` is the ideal.
- `detect_dead_experts` names the experts below a utilization threshold so you can reinit them with small noise or temporarily lower `K`.
- `visualise_routing` builds a thread-safe, headless heatmap (no `pyplot`); `export_routing_stats` streams tensor-containing metric dicts to JSONL by auto-converting tensors with `.tolist()`.
- The five-metric playbook — entropy, imbalance, overflow fraction, aux-loss fraction, dead experts — is what makes MoE training controllable rather than fragile.

## Next → `08-testing.md`

You now have the instruments. The next lesson shows how the library *tests* this behaviour — turning "I think routing is healthy" into assertions a CI run can enforce.
