# Lesson 02 — The Configuration System

> One validated dataclass, `MoEConfig`, is the single source of truth for every knob the MoE layer touches — so the rest of the library never reaches for global state.

## Learning objectives

By the end of this lesson you will be able to:

- Explain *why* the library funnels every hyperparameter through one dataclass instead of module-level constants or environment variables.
- Identify what each field of `MoEConfig` controls and the trade-offs it encodes.
- Derive the per-expert **capacity** formula and predict what `capacity()` returns, including the `drop_tokens=False` short-circuit.
- Use `validate()` to fail fast, and read its error messages as fix-it instructions.
- Serialise and reload a config with `to_dict` / `from_dict`, and understand forward-compatibility.
- Build configs from the three presets and craft a custom preset of your own.

## Prerequisites

- Comfort with Python dataclasses, type hints, and `@classmethod`.
- Lesson 01 (the conceptual MoE picture: experts `N`, top-`K` routing, the parameter-vs-compute trade-off).
- Basic deep-learning vocabulary: residual stream, feed-forward width, dropout, softmax.

All code below lives in `moe/config.py`. Open it alongside this lesson.

## Why a single validated dataclass?

Imagine the alternative. The router needs `num_experts`, the experts need `d_ff` and `activation`, the loss functions need `alpha` and `beta`, and the dispatch code needs `capacity_factor`. If each of those reached into a global `CONFIG` dict or read `os.environ`, you would have four problems at once: nothing is **thread-safe** (two layers with different settings would clobber each other), nothing is **serialisable** as a unit, nothing is **validated** in one place, and nobody can tell *which* knobs a given layer actually uses.

The module docstring states the design goal directly:

```python
"""Configuration dataclass for the Mixture of Experts (MoE) routing layer.

... Keeping all knobs in one immutable-by-convention
dataclass means the rest of the package never has to reach for module-level
constants or environment variables, which keeps the code thread-safe and easy to
serialise.
"""
```

So `MoEConfig` is the *contract*. You build one, validate it, and hand it to a layer. The layer reads from it but the package never mutates it afterward — that "immutable by convention" promise is what makes it safe to share a config across threads. The class is deliberately **not** `frozen`, though, so that you can build it up incrementally before freezing it in practice:

```python
@dataclass
class MoEConfig:
    ...
    # Notes:
    #   The dataclass is intentionally *not* frozen so that callers may build a
    #   config incrementally, but the package never mutates a config after the
    #   layer is constructed, preserving thread safety.
```

## A tour of the fields

The defaults describe a tiny, fast configuration meant for unit tests. Here are the structural knobs:

```python
d_model: int = 32          # residual-stream width: input/output of the layer
d_ff: int = 64             # hidden width of each expert's two-layer MLP
num_experts: int = 4       # total experts N (parameters scale with this)
top_k: int = 1             # experts each token visits (1 = Switch, 2 = Mixtral)
capacity_factor: float = 1.25  # slack on the per-expert token buffer
```

`d_model` and `d_ff` size the experts; `num_experts` grows total parameters *without* growing per-token compute (only `top_k` of them run per token). The behavioural knobs come next:

```python
router_type: RouterType = "topk"   # "topk" | "switch" | "expert_choice"
activation: Activation = "gelu"    # "gelu" | "swiglu" | "relu2"
expert_dropout: float = 0.0        # dropout inside each expert (train only)
use_noisy_gating: bool = False     # learnable Gaussian noise on gate logits (Shazeer 2017)
noise_std_init: float = 1.0        # init scale for the W_noise projection
jitter_noise: float = 0.0          # multiplicative input jitter (Switch), train only
```

`router_type` selects which router class is instantiated. `activation` picks the expert nonlinearity. `use_noisy_gating` and `jitter_noise` are two different *exploration* tricks — noise on the logits versus jitter on the inputs — both active only during training. Finally the loss and dispatch knobs:

```python
alpha: float = 1e-2                  # weight of the load-balancing aux loss
beta: float = 1e-3                   # weight of the router z-loss
normalize_router_weights: bool = False  # renormalise top-K gates to sum to 1 (Mixtral)
drop_tokens: bool = True             # enforce capacity & drop overflow, or keep every token
dispatch_strategy: DispatchStrategy = "naive"  # "naive" loop | "batch" fixed-capacity
use_bias: bool = True                # bias terms in expert linear layers
router_z_score_norm: bool = False    # divide logits by running std before softmax
eps: float = 1e-8                    # guards divisions and logarithms
```

Two subtle ones deserve a sentence each. `normalize_router_weights` decides whether the `K` selected gate weights are rescaled to sum to 1 (the Mixtral convention) or left as raw masked-softmax probabilities that sum to `<= 1` (the Switch convention). And `dispatch_strategy` is purely a mechanism choice: `"naive"` is a readable per-expert loop, `"batch"` is a fixed-capacity batched path, and the docstring promises they "produce identical outputs when no tokens are dropped."

### Named constants, not magic numbers

Notice the bounds used by validation are named at module scope, so a reader never guesses where a threshold came from:

```python
MIN_CAPACITY_FACTOR: float = 1.0
VALID_ROUTER_TYPES: tuple[str, ...] = ("topk", "switch", "expert_choice")
VALID_ACTIVATIONS: tuple[str, ...] = ("gelu", "swiglu", "relu2")
VALID_DISPATCH: tuple[str, ...] = ("naive", "batch")
```

They are tuples (hashable, immutable) and they double as the `Literal` types `RouterType`, `Activation`, and `DispatchStrategy` for editor autocompletion.

## The derived helpers: `capacity` and `tokens_per_expert`

A real MoE cannot let an expert receive an unbounded number of tokens — buffers must be fixed-size. The **capacity** `C` is the maximum number of tokens one expert will accept in a batch. Let $T$ be the batch token count (`batch * seq`), $N$ the number of experts, $K$ the top-`k`, and $f$ the capacity factor. Then

$$
C = \left\lceil \frac{f \cdot K \cdot T}{N} \right\rceil
$$

The intuition: a perfectly balanced batch sends each expert its fair share of $K\,T/N$ token-assignments (the $K$ accounts for each token producing `top_k` assignments); the factor $f \ge 1$ adds slack for imbalance. Here is the implementation:

```python
def capacity(self, batch_tokens: int) -> int:
    if not self.drop_tokens:
        return batch_tokens
    raw = self.capacity_factor * self.top_k * batch_tokens / self.num_experts
    return min(batch_tokens, max(1, math.ceil(raw)))
```

Three things to read carefully:

1. **The short-circuit.** If `drop_tokens` is `False`, capacity is simply `batch_tokens` — no token can ever overflow, because the buffer is as large as the whole batch. This is how the library guarantees exact parity between dispatch strategies.
2. **The `top_k` factor.** The canonical Switch formula is $f\,T/N$ for top-1. Multiplying by `top_k` provisions enough slots for the `top_k` assignments each token makes; at `top_k == 1` it reduces to the canonical formula exactly.
3. **The clamps.** `max(1, ...)` guarantees at least one slot; `min(batch_tokens, ...)` ensures the buffer never claims to be larger than the batch.

The sibling helper is the fair-share baseline used by the load-balancing loss:

```python
def tokens_per_expert(self, batch_tokens: int) -> float:
    return batch_tokens / self.num_experts
```

With the test defaults (`N=4`, `K=1`, `f=1.25`) and `batch_tokens=1024`: `tokens_per_expert` is `1024/4 = 256.0`, and `capacity` is `ceil(1.25 * 1 * 1024 / 4) = ceil(320.0) = 320`.

## `validate()`: fail fast with a fix-it message

The constructor never raises — you can build a half-finished config. Validation is a separate, explicit step that returns `self` so it chains:

```python
cfg = MoEConfig(d_model=512, d_ff=2048, num_experts=8, top_k=2).validate()
```

The philosophy is **fail fast with a fix-it message**: every error names the offending field, its current value, the constraint, and how to repair it. A few representative checks:

```python
if self.top_k > self.num_experts:
    raise ValueError(
        f"top_k ({self.top_k}) cannot exceed num_experts "
        f"({self.num_experts}). Reduce top_k or add experts."
    )
if self.capacity_factor < MIN_CAPACITY_FACTOR:
    raise ValueError(
        f"capacity_factor must be >= {MIN_CAPACITY_FACTOR}, got "
        f"{self.capacity_factor}. A value below 1.0 cannot hold a "
        "balanced batch; use 1.0-2.0."
    )
if self.router_type == "switch" and self.top_k != 1:
    raise ValueError(
        f"router_type='switch' requires top_k == 1, got top_k="
        f"{self.top_k}. Switch routing is top-1 by definition; use "
        "router_type='topk' for top_k > 1."
    )
```

That last one encodes a *semantic* invariant, not just a range: Switch routing is top-1 by definition, so the config refuses an inconsistent combination rather than silently misbehaving at runtime. Other guards reject `d_model <= 0`, `d_ff <= 0`, `num_experts < 1`, `top_k < 1`, `expert_dropout` outside `[0, 1)`, negative `noise_std_init`/`jitter_noise`/`alpha`/`beta`, unknown `router_type`/`activation`/`dispatch_strategy`, and `eps <= 0`. Catching these at construction is far cheaper than debugging a `nan` thirty minutes into training.

## Serialisation: `to_dict` / `from_dict`

Because everything lives in one dataclass, saving a config is one line:

```python
def to_dict(self) -> dict[str, Any]:
    return asdict(self)
```

Loading is symmetric, with one deliberate twist — **unknown keys are ignored**:

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> MoEConfig:
    if not isinstance(data, dict):
        raise TypeError(f"from_dict expects a dict, got {type(data).__name__}.")
    known = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in known}
    return cls(**filtered)
```

Why drop unknown keys instead of erroring? **Forward compatibility.** A config saved by a newer version of the library (which may have added a field) still loads in an older version: the older `from_dict` simply ignores the field it doesn't recognise rather than crashing. It does still raise `TypeError` if you hand it something that isn't a dict at all.

## The presets: encoded recipes

Realistic architectures are hard to remember field-by-field, so three classmethods encode known recipes — each ends in `.validate()`, so a preset can never hand back an invalid config.

```python
MoEConfig.switch_transformer(num_experts=8, d_model=512)
MoEConfig.mixtral_style(num_experts=8, d_model=512)
MoEConfig.gpt4_style(num_experts=16, d_model=768)
```

- **`switch_transformer()`** (Fedus et al. 2022): `top_k=1`, `router_type="switch"`, `activation="relu2"`, `capacity_factor=1.25`, `jitter_noise=1e-2`, `drop_tokens=True`. Top-1 routing with token dropping and ReLU-squared experts is the paper's recipe.
- **`mixtral_style()`** (Mixtral 8x7B): `top_k=2`, `router_type="topk"`, `activation="swiglu"`, `d_ff=int(d_model * 3.5)`, `normalize_router_weights=True`, `drop_tokens=False`. Mixtral processes *every* token (no dropping) and renormalises the two selected gates to sum to 1.
- **`gpt4_style()`**: `num_experts=16`, `top_k=2`, `activation="gelu"`, `capacity_factor=1.5`, `normalize_router_weights=True`, `drop_tokens=True`, `use_noisy_gating=True`. Its docstring is admirably honest: GPT-4's architecture is not published, so this is "an illustrative large configuration, not a faithful reproduction."

A handy debugging method prints any config compactly:

```python
def summary(self) -> str:
    return (f"MoEConfig(d_model={self.d_model}, d_ff={self.d_ff}, "
            f"N={self.num_experts}, k={self.top_k}, cf={self.capacity_factor}, "
            f"router={self.router_type}, act={self.activation})")
```

## Common pitfalls

- **Forgetting to call `validate()`.** The constructor accepts nonsense happily. Validation is opt-in — always chain `.validate()` (the presets do this for you).
- **Expecting `capacity()` to drop tokens when `drop_tokens=False`.** It returns the full `batch_tokens`; the short-circuit runs *before* the formula.
- **Setting `router_type="switch"` with `top_k=2`.** It will raise — Switch is top-1 by definition; use `"topk"` for `top_k > 1`.
- **Treating `capacity_factor < 1.0` as "tighter packing".** It is rejected: below 1.0 the buffer cannot hold even a balanced batch.
- **Assuming `from_dict` round-trips perfectly across versions.** Unknown keys vanish silently — by design — so a value the older code doesn't know about is simply not restored.

## Exercises

1. **Construct a config that raises on `top_k`.** Build one where `validate()` raises because `top_k` exceeds `num_experts`, and print the message.
2. **Predict then verify `capacity`.** With `num_experts=8`, `top_k=2`, `capacity_factor=1.25`, `drop_tokens=True`, compute `capacity(batch_tokens=4096)` by hand, then check it. Then flip `drop_tokens=False` and predict again.
3. **Trigger the Switch invariant.** Produce the exact `ValueError` raised when `router_type="switch"` is combined with `top_k=2`.
4. **Round-trip with an unknown key.** Take `mixtral_style().to_dict()`, add a bogus key `"future_flag": True`, and confirm `from_dict` still builds a valid config.
5. **Make a custom preset.** Write a function `deepseek_style()` returning a validated 64-expert, top-6, SwiGLU config with `drop_tokens=False` and `normalize_router_weights=True`.

## Solutions

```python
from moe.config import MoEConfig

# 1. top_k > num_experts
try:
    MoEConfig(num_experts=4, top_k=8).validate()
except ValueError as e:
    print(e)  # "top_k (8) cannot exceed num_experts (4). Reduce top_k or add experts."

# 2. capacity. By hand: ceil(1.25 * 2 * 4096 / 8) = ceil(1280.0) = 1280
cfg = MoEConfig(num_experts=8, top_k=2, capacity_factor=1.25, drop_tokens=True)
assert cfg.capacity(4096) == 1280
cfg2 = MoEConfig(num_experts=8, top_k=2, drop_tokens=False)
assert cfg2.capacity(4096) == 4096   # short-circuit: full batch, no drops

# 3. Switch invariant
try:
    MoEConfig(router_type="switch", top_k=2, num_experts=8).validate()
except ValueError as e:
    print(e)  # "router_type='switch' requires top_k == 1, got top_k=2. ..."

# 4. Round-trip with an unknown key
d = MoEConfig.mixtral_style().to_dict()
d["future_flag"] = True
cfg3 = MoEConfig.from_dict(d).validate()   # future_flag is silently dropped
assert not hasattr(cfg3, "future_flag")

# 5. Custom preset
def deepseek_style(num_experts: int = 64, d_model: int = 1024) -> MoEConfig:
    return MoEConfig(
        d_model=d_model,
        d_ff=int(d_model * 2.5),
        num_experts=num_experts,
        top_k=6,
        capacity_factor=1.5,
        router_type="topk",
        activation="swiglu",
        normalize_router_weights=True,
        drop_tokens=False,
    ).validate()

print(deepseek_style().summary())
```

## Key takeaways

- `MoEConfig` is the **single, validated source of truth** — no globals, thread-safe, serialisable.
- `capacity()` implements $C = \lceil f K T / N \rceil$ with a `drop_tokens=False` short-circuit that returns the full batch.
- `validate()` fails fast with field-named, fix-it error messages and enforces semantic invariants like *Switch implies top-1*.
- `to_dict` / `from_dict` give clean serialisation, and `from_dict` ignores unknown keys for forward compatibility.
- The presets `switch_transformer`, `mixtral_style`, and `gpt4_style` are pre-validated, documented recipes you can copy when building your own.

## Next → `03-load-balancing-and-losses.md`

We just met `alpha` and `beta` as numbers. Next we wire them into the actual auxiliary load-balancing loss and the router z-loss — the machinery that keeps experts from collapsing onto a single favourite.
