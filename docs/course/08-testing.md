# Lesson 08 — Testing an MoE Layer

> A sparse router has no single "correct output" you can eyeball, so we test it by pinning the invariants that *must* hold — exact values at analytically-known points, exact sparsity, provable numerical stability, and the training dynamics the layer promises to deliver.

## Learning objectives

By the end of this lesson you will be able to:

- Explain the testing philosophy behind the suite: pin invariants at points where the math gives an exact answer, then check shapes, sparsity, stability, and dynamics around them.
- Use the shared fixtures in `tests/conftest.py` and explain why randomness flows through an explicit `torch.Generator` instead of a global `manual_seed`.
- Recognise the main test categories — shape/sparsity, train-vs-eval determinism, gradient flow, numerical stability, dispatch parity, capacity overflow — and what each one protects.
- Read the integration tests as a small training harness and understand the headline `alpha=0` collapse experiment.
- Write property-based tests with Hypothesis and say *why* they catch bugs that worked examples miss.
- Connect the suite to the `make` targets and the 100% coverage target.

## Prerequisites

- Lessons 02–07: you should know what `MoEConfig`, the routers, the expert bank, the losses, and `MoELayer` do.
- Comfort with `pytest` (fixtures, `parametrize`, `pytest.raises`) and basic PyTorch (`requires_grad`, `.backward()`, `train()`/`eval()`).
- A rough memory of the loss math from Lesson 05: the auxiliary load-balancing loss `L_aux` and the router z-loss `L_z`.

All code below lives under `tests/`. Open the files alongside this lesson and run `make test` as you go.

## Why testing an MoE layer is hard (and how we make it easy)

A dense feed-forward layer is easy to test: feed it a known input, compare against a hand-computed output. An MoE layer is not. Its forward pass branches on an `argmax`, adds Gaussian noise during training, drops tokens that overflow a capacity buffer, and sums a task loss with two auxiliary losses. There is no tidy closed-form "expected output" to assert against.

So the suite does something smarter. Instead of checking outputs everywhere, it **pins invariants at points where the math collapses to a single number**, and checks structural properties (shape, sparsity, finiteness) everywhere else. Three analytically-known anchors recur:

- **Perfect balance → `L_aux = alpha`.** With `N` experts and every expert receiving an equal share, `f_i = P_i = 1/N`, so `sum_i f_i P_i = 1/N` and `L_aux = alpha * N * (1/N) = alpha`.
- **One-hot routing → entropy `0`.** All probability mass on one expert means zero routing entropy.
- **Zero logits → `L_z = beta * (log N)^2`.** `logsumexp(0, ..., 0) = log N`, squared and averaged over tokens.

If you can compute the answer by hand at a special point, the assertion is exact (`pytest.approx(..., abs=1e-7)`) rather than fuzzy. Everything else — shapes, sparsity counts, NaN checks, training trends — is structural and equally pinnable. That is the whole philosophy.

## The shared fixtures: `tests/conftest.py`

Every test starts from a small set of reusable fixtures. The module docstring states the discipline up front:

> All randomness flows through an explicit `torch.Generator` or a locally-scoped `manual_seed` so tests do not leak global RNG state into one another.

There are four fixtures:

```python
@pytest.fixture
def small_config() -> MoEConfig:
    """A tiny, fast config for unit tests: 4 experts, top-1, d_model=32."""
    return MoEConfig(d_model=32, d_ff=64, num_experts=4, top_k=1, ...)

@pytest.fixture
def standard_config() -> MoEConfig:
    """A realistic config for integration-style tests: 8 experts, top-2."""
    return MoEConfig(d_model=512, d_ff=2048, num_experts=8, top_k=2, ...)
```

`small_config` keeps unit tests fast; `standard_config` is realistic enough that integration behaviour is meaningful. The reproducible-input fixture is a **factory**, not a tensor:

```python
@pytest.fixture
def random_input() -> Callable[..., torch.Tensor]:
    def _make(config: MoEConfig, batch: int = 2, seq: int = 16) -> torch.Tensor:
        generator = torch.Generator().manual_seed(INPUT_SEED)  # INPUT_SEED = 42
        return torch.randn(batch, seq, config.d_model, generator=generator)
    return _make
```

Why a factory? Because a test usually needs an input *sized to a specific config*, and often more than one. Returning `_make` lets each test ask for exactly the shape it wants while still getting a deterministic tensor.

Why `torch.Generator().manual_seed(42)` rather than `torch.manual_seed(42)`? Because the global call mutates *process-wide* RNG state. If one test seeds the global RNG and the next test relies on randomness, the second test's behaviour now depends on test ordering — a classic source of "passes alone, fails in the suite" flakiness. A private `Generator` is a self-contained stream: `torch.randn(..., generator=generator)` draws from it and disturbs nothing else. This is also why a *library* should never call the global `manual_seed` internally — it would silently reseed the user's program.

The last fixture gives you a layer in a realistic, non-random-init state:

```python
@pytest.fixture
def trained_layer(standard_config: MoEConfig) -> MoELayer:
    """A MoELayer after 10 gradient steps (non-random-init state)."""
    torch.manual_seed(0)
    layer = MoELayer(standard_config)
    ...
    for _ in range(10):
        out = layer(x)
        loss = task + out.aux_loss + out.z_loss
        loss.backward(); optimiser.step()
    layer.eval()
    return layer
```

Note it combines all three loss terms exactly as a real training loop would — `task + out.aux_loss + out.z_loss` — so anything built on this fixture exercises the real loss path.

## Category 1 — shapes and sparsity

These tests guard the contract that downstream code indexes by. `test_output_shapes` in `tests/test_router.py` is parametrized over all three routers:

```python
@pytest.mark.parametrize("router_type", ["topk", "switch", "expert_choice"])
def test_output_shapes(router_type: str) -> None:
    ...
    assert out.dispatch_weights.shape == (n_tokens, config.num_experts, 1)
    assert out.expert_indices.shape == (n_tokens, top_k)
    assert out.aux_loss.shape == ()
```

The docstring says *why* this matters: "the expert bank and losses index these tensors by exact shape; a wrong dimension would broadcast silently rather than error." A silent broadcast is the worst kind of bug — no exception, just wrong numbers.

Sparsity is the defining property of MoE, so it gets its own pinned tests. `test_topk_sparsity_exactly_k` asserts that with `top_k=2`, **exactly** two experts are non-zero per token:

```python
nonzero_per_token = torch.count_nonzero(out.combine_weights.squeeze(-1), dim=-1)
assert torch.all(nonzero_per_token == 2)
```

A sibling pair pins the weight-normalisation convention: `test_combine_weights_sum_le_one` (masked full-softmax → weights sum to `<= 1`) and `test_normalised_weights_sum_to_one` (with `normalize_router_weights=True` the selected weights renormalise to exactly `1`).

## Category 2 — train-vs-eval determinism

Noisy gating must *help exploration during training* yet be *perfectly reproducible at inference*. `test_train_vs_eval_noise` proves both halves at once:

```python
router.train()
train_runs = torch.stack([router(x).combine_weights for _ in range(50)])
router.eval()
eval_runs = torch.stack([router(x).combine_weights for _ in range(50)])

assert train_runs.std(dim=0).max().item() > 0.0   # varies in train
assert eval_runs.std(dim=0).max().item() == 0.0    # identical in eval
```

The standard deviation across 50 runs is strictly positive in train mode and *exactly* zero in eval. Note the assertion is `== 0.0`, not "small" — eval determinism is binary, so the test demands it exactly. The expert bank has the mirror-image test, `test_dropout_training_only` in `tests/test_experts.py`, using the same 50-run std trick on expert dropout, and the whole layer gets `test_eval_determinism` in `tests/test_layer.py`, which turns on noise *and* dropout *and* jitter and still asserts `torch.equal(layer(x).output, layer(x).output)`.

## Category 3 — gradient flow

Backprop must reach the parts that should learn and *avoid* the parts that should not. `test_gradient_flow_through_router` confirms both the gate and the learnable-noise projection receive gradient:

```python
out.aux_loss.backward()
assert torch.any(router.w_gate.weight.grad != 0)
assert torch.any(router.w_noise.weight.grad != 0)
```

The subtle one is `test_moeloss_gradient_only_through_probs` in `tests/test_losses.py`. Recall `L_aux = alpha * N * sum_i f_i * P_i`, where `P_i` is the differentiable mean router probability but `f_i` is a *counter* of how many tokens went to expert `i` — derived from `argmax`/`topk`, which is non-differentiable and deliberately detached. The test proves the detachment with a clever two-leaf construction:

```python
prob_logits  = torch.randn(8, 4, requires_grad=True)
index_source = torch.randn(8, 4, requires_grad=True)
probs   = torch.softmax(prob_logits, dim=-1)
indices = index_source.topk(2, dim=-1).indices

loss = auxiliary_load_balancing_loss(probs, indices, num_experts=4, alpha=1.0)
loss.backward()

assert torch.any(prob_logits.grad != 0)   # P_i path carries gradient
assert index_source.grad is None          # f_i path is fully detached
```

`probs` and `indices` come from **separate leaves**, so after `backward()` you can read each path independently. The probs' leaf has a gradient; the index source's leaf has `None`. That `None` is the proof that the `f_i` counter contributes no gradient — exactly the design the loss requires. The end-to-end version, `test_gradient_flow_end_to_end` in `tests/test_layer.py`, routes to *all* experts (`top_k == num_experts`) so it can assert every expert's `w1.weight.grad` is non-zero, plus the input's `.grad`.

## Category 4 — numerical stability

Large logits are where softmax-based code goes to die. Two tests pin stability at extreme magnitudes. `test_numerical_stability_large_logits` scales inputs by `1000x` and asserts every output tensor `torch.isfinite(...).all()`. `test_z_loss_numerical_stability` feeds logits of magnitude `1e4`:

```python
logits = torch.full((4, 8), 1e4)
loss = router_z_loss(logits, beta=BETA)
assert torch.isfinite(loss)
```

A naive `log(sum(exp(logits)))` would overflow to `inf` here; the test passes only because the implementation uses `logsumexp`. The test *documents the requirement* — anyone who "optimises" the z-loss back into a naive form gets an immediate failure.

## Category 5 — the analytically-pinned loss values

This is the heart of the suite. Each is exact because the math is exact:

```python
def test_aux_loss_perfect_balance():  # round-robin indices, uniform probs
    assert loss.item() == pytest.approx(ALPHA, abs=1e-7)

def test_aux_loss_collapsed():        # all mass on expert 0
    assert loss.item() == pytest.approx(ALPHA * num_experts, abs=1e-7)

def test_z_loss_at_origin():          # zero logits, N experts
    assert loss.item() == pytest.approx(BETA * math.log(num_experts) ** 2, abs=1e-6)
```

These three nail down the entire dynamic range of each loss: minimum (`alpha`), maximum (`alpha * N`), and the resting value at the origin. The monitoring helpers in `tests/test_utils.py` mirror them — `test_entropy_uniform_and_onehot` pins uniform entropy to `log(N)` and one-hot to `0`; `test_load_imbalance_perfect_balance` pins the imbalance ratio to `1.0`.

## Category 6 — dispatch parity and capacity overflow

The library has two dispatch implementations: a readable `naive` loop and a fused `batch` path that runs in production. They must agree. `test_dispatch_strategies_agree` is parametrized over activations (including SwiGLU) and bias settings:

```python
config.dispatch_strategy = "naive"
out_naive = bank(x, routed.dispatch_weights, routed.expert_indices)
config.dispatch_strategy = "batch"
out_batch = bank(x, routed.dispatch_weights, routed.expert_indices)
assert torch.allclose(out_naive, out_batch, atol=1e-4)
```

The tolerance is `1e-4`, not exact equality, because the fused path reorders floating-point operations. This is the right call: demanding bit-exactness would produce false failures, while `1e-4` still catches any real logic divergence.

Capacity overflow is tested by *construction*. `test_switch_router_capacity_overflow` builds an identity-scaled gating head and one-hot inputs so 90 of 100 tokens route to expert 0; with `capacity_factor=1.0` and 4 experts the capacity is 25, so exactly `90 - 25 = 65` tokens must be dropped:

```python
dropped = int((out.combine_weights.squeeze(-1).sum(dim=-1) == 0).sum())
assert dropped == max(0, 90 - capacity) == 65
```

The expert bank's `test_batch_dispatch_reports_overflow` checks the bookkeeping side: route all 10 tokens to expert 0 with capacity 5, and assert `bank.last_overflow_tokens == 5`.

## Validation tests: every guard must fire

Config and input validation get parametrized "rejection" tables. `INVALID_CASES` in `tests/test_config.py` lists 17 bad configs paired with the substring each error must contain, and `test_validate_rejects_invalid` runs them all through `pytest.raises(ValueError, match=message)`. The router and loss modules do the same: `test_switch_requires_top_1`, `test_flatten_accepts_2d_and_rejects_bad_rank`, `test_aux_loss_rejects_bad_shape`, `test_expert_ffn_input_validation`. The point is coverage of the *unhappy path*: validation is the only guard between a config typo and a silently-wrong training run, so every branch must be proven to raise.

## Integration tests: the training harness

`tests/test_integration.py` exercises the whole stack as a user would. A private `_train` helper builds a two-layer residual network whose FFNs are `MoELayer`s, trains it on a regression target, and returns per-step traces in a `TrainTrace` NamedTuple (`total`, `imbalance`, `entropy`). Crucially, the input is *low-diversity* by design — a shared base direction plus small per-token noise (`INPUT_DIVERSITY = 0.3`) — which manufactures the "rich-get-richer" pressure that collapses routing when load balancing is off.

Three tests then read those traces. `test_training_converges_and_diversifies` smooths the loss with a moving average and asserts it drops to under half its starting value *and* is non-increasing at least 90% of steps, and that routing entropy at step 50 exceeds step 0. The two headline tests form the experiment that justifies the whole auxiliary loss:

```python
def test_routing_collapses_without_aux_loss():
    result = _train(alpha=0.0, steps=200)
    assert sum(result.imbalance[-20:]) / 20 > 2.0   # collapse

def test_aux_loss_prevents_collapse():
    result = _train(alpha=1e-2, steps=200)
    assert sum(result.imbalance[-20:]) / 20 < 1.5    # stays balanced
```

Same network, same data, same seeds — the *only* difference is `alpha`. With `alpha=0` the load imbalance grows past `2.0`; with the default `alpha` it stays below `1.5`. That contrast is the experimental proof that the load-balancing loss is what prevents collapse, turned into a regression test.

Two more integration tests round it out. `test_mixed_precision_bfloat16` runs the layer under `torch.autocast("cpu", dtype=torch.bfloat16)` and asserts the output is `bfloat16` while *both losses stay float32* — routing probabilities and loss accumulation need full precision even when the expert matmuls do not. `test_state_dict_roundtrip` saves and reloads a layer (with noise and z-score norm on) and asserts the reloaded copy reproduces the original eval output bit-for-bit via `torch.equal`.

## Property-based testing with Hypothesis

Worked examples test the cases *you thought of*. Property-based tests, in `tests/test_property_based.py`, test thousands of cases you did not. A `@st.composite` strategy draws a *valid* random config and a matching input:

```python
@st.composite
def _router_case(draw):
    num_experts = draw(st.integers(min_value=2, max_value=8))
    top_k       = draw(st.integers(min_value=1, max_value=num_experts))
    d_model     = draw(st.integers(min_value=16, max_value=64))
    ...
    return config, x
```

Then each test asserts an *invariant* that must hold for every draw — not a specific value, but a property:

```python
@given(case=_router_case())
@_SETTINGS
def test_topk_sparsity_invariant(case):
    config, x = case
    out = TopKRouter(config)(x)
    k = min(config.top_k, config.num_experts)
    nonzero = (out.dispatch_weights.squeeze(-1) != 0).sum(dim=-1)
    assert torch.all(nonzero == k)
```

Companion properties assert outputs are always finite (`test_router_outputs_are_finite`) and that both auxiliary losses are always non-negative (`test_aux_loss_non_negative`, `test_z_loss_non_negative`). These catch the dimension- and dtype-dependent bugs that a hand-picked `num_experts=4, top_k=2` example would sail right past — what if `top_k == num_experts`? what if `seq == 1`? Hypothesis tries those corners for you.

Note the shared settings: `settings(max_examples=25, deadline=None)`. Building a real `nn.Module` per example is not free, so the example count is capped, and the per-example deadline is dropped so that a slow-but-valid draw does not flake by tripping a timeout. That `deadline=None` is a small but important detail — Hypothesis's default deadline will fail tests merely for being slow, which is meaningless noise here.

## How it all runs: coverage and `make`

The suite is wired to be a single command. `make test` runs `python3 -m pytest`, and `pyproject.toml` supplies the flags:

```toml
[tool.pytest.ini_options]
addopts = "-ra -q --cov=moe --cov-report=term-missing"
```

So every run measures coverage and prints the exact line numbers that are *missing*. The suite reaches 100% — the only deliberate exclusion is `moe/bench.py` (a developer timing script, omitted under `[tool.coverage.run]`). One hundred percent line coverage is not the same as "bug-free", but combined with the invariant pins and property tests above, it means no line ships untouched by a test. `make lint` (ruff + mypy) and `make format` complete the quality gate.

## Common pitfalls

- **Global RNG leakage.** Calling `torch.manual_seed(42)` inside a test (or worse, inside library code) reseeds process-wide state, making other tests order-dependent. Prefer a private `torch.Generator().manual_seed(...)` as the fixtures do.
- **Flaky stochastic assertions.** Asserting "the loss went down" at *every* step will fail on optimisation noise. The suite smooths with a moving average and tolerates a 10% slack (`>= 0.9`). Pin trends, not steps.
- **Demanding exact equality where floats reorder.** The naive-vs-batch dispatch test uses `atol=1e-4`, not `==`. Reserve exact equality for things that are genuinely exact: eval determinism and state-dict round-trips.
- **Hypothesis deadline failures.** Without `deadline=None`, a slow-but-correct draw fails for being slow. Set it when each example builds real modules.
- **Testing only the happy path.** The validation tables (`INVALID_CASES`, the `pytest.raises` cases) exist because a guard that never fires is a guard you cannot trust.

## Exercises

1. **A new invalid-config test.** Add a row to `INVALID_CASES` in `tests/test_config.py` for a config with `top_k` greater than `num_experts` but written as a *new* combination not already listed, and confirm the error substring matches.
2. **A new property test.** Write a Hypothesis test asserting that for any `_router_case`, the per-token `combine_weights` sum is always `<= 1 + 1e-6`. Reuse the `_router_case` strategy and `_SETTINGS`.
3. **An analytically-pinned value.** Add a loss test for the half-collapsed case: with `N=4` and exactly half the tokens on expert 0 and half on expert 1 (uniform probs over those two), compute `L_aux` by hand and assert it with `pytest.approx`.
4. **A determinism test for jitter.** Using `MoEConfig(jitter_noise=0.2)`, write a test proving jitter perturbs router logits in train mode but not in eval, mirroring `test_train_vs_eval_noise`.
5. **(Stretch) A capacity-edge test.** Construct inputs so *exactly* the capacity is reached with zero drops, and assert `bank.last_overflow_tokens == 0`.

## Solutions

1. The existing `({"num_experts": 4, "top_k": 5}, "cannot exceed num_experts")` row is the canonical form. A distinct combination such as `({"num_experts": 2, "top_k": 3}, "cannot exceed num_experts")` exercises the same guard with different numbers; the parametrized `test_validate_rejects_invalid` will pick it up automatically.

```python
@given(case=_router_case())
@_SETTINGS
def test_combine_weights_sum_le_one_property(case):
    config, x = case
    out = TopKRouter(config)(x)
    sums = out.combine_weights.squeeze(-1).sum(dim=-1)
    assert torch.all(sums <= 1.0 + 1e-6)
```

3. Half on expert 0, half on expert 1: each of those two experts has `f_i = 1/2` and `P_i = 1/2`, the other two have `f_i = P_i = 0`. So `sum_i f_i P_i = 2 * (1/2)(1/2) = 1/2`, giving `L_aux = alpha * N * 1/2 = alpha * 4 * 1/2 = 2 * alpha`. Build round-robin indices over `{0, 1}` and uniform-over-two probs, then `assert loss.item() == pytest.approx(2 * ALPHA, abs=1e-7)`.

4. Copy `test_train_vs_eval_noise`, swap `use_noisy_gating=True` for `jitter_noise=0.2`, stack 50 runs of `router(x).router_logits` in each mode, and assert the train std is `> 0` while the eval std is `== 0`. (This is exactly the pattern `test_jitter_and_zscore_paths_run` builds on.)

5. Route `capacity` tokens to one expert and the rest elsewhere using the `_all_to_expert0`-style helper, run the bank with `dispatch_strategy="batch"`, and assert `bank.last_overflow_tokens == 0` and `bank.last_dispatched_tokens == capacity`.

## Key takeaways

- **Pin invariants at analytically-known points.** Perfect balance → `alpha`, one-hot → entropy `0`, zero logits → `beta(log N)^2`. Exact math lets you write exact assertions.
- **Isolate randomness with `torch.Generator`.** Never let a test (or the library) touch global RNG; that is the root of order-dependent flakiness.
- **Test structure, not just values.** Shapes, sparsity counts, finiteness, and gradient presence/absence are all pinnable even when the output itself is not.
- **The `alpha=0` collapse experiment is a regression test.** Same network and seeds with only `alpha` changed proves the load-balancing loss earns its keep.
- **Properties catch what examples miss.** Hypothesis sweeps the corners (`top_k == num_experts`, `seq == 1`) you would never enumerate by hand.
- **100% coverage plus invariants plus properties** is a layered guarantee: no line untested, no invariant unpinned, no corner unswept.

## Next → `09-performance-and-scaling.md`

You can now prove the layer is *correct*. Next we make it *fast*: profiling the dispatch paths, the `make bench` harness comparing MoE against an equivalent dense FFN, and how capacity, `top_k`, and dispatch strategy trade compute for quality at scale.
