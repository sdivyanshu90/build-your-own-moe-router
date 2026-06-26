# MoE Routing Layer — Technical Documentation

This document specifies, motivates, and operationalizes the Mixture-of-Experts (MoE) routing layer that accompanies this codebase. It is written to be self-contained: every equation defines its symbols in the surrounding prose, every numerical claim is worked through with concrete numbers, and every design decision is traced back to the literature that introduced it. The conventions used here — the gating equation, the auxiliary load-balancing loss, the router z-loss, the capacity formula, and the combine-weight normalization — are the exact conventions implemented in the accompanying code, so this document doubles as the contract that the implementation must satisfy.

A Mixture of Experts replaces a single feed-forward network (FFN) with a collection of $N$ parallel FFNs, called *experts*, and a small trainable *gating network* (or *router*) that decides, per token, which experts should process that token. Only a few experts run for any given token, so the layer holds far more parameters than it spends compute on. The entire subtlety of making this work in practice lives in the router: how it scores experts, how it keeps load balanced across experts, how it stays numerically stable in low precision, and how it dispatches tokens across devices. Those four concerns organize the body of this document.

---

## 1.1 Conceptual Foundations

### Why MoE: the compute-versus-parameter trade-off

The central economic observation behind MoE is that the quality of a transformer scales with its parameter count, but the cost of running it scales with the *activated* compute per token. A dense model couples those two quantities rigidly: every parameter participates in every token's forward pass. MoE breaks the coupling. By activating only a small subset of experts per token, it lets you grow the parameter budget — and therefore the model's capacity to memorize and specialize — while holding the per-token FLOP budget nearly fixed.

To make this precise we need a FLOP accounting for the dense baseline. A standard transformer FFN maps a hidden vector of dimension $d_{model}$ up to an intermediate width $d_{ff}$ and back down. Let $d_{model}$ be the model (residual stream) dimension and $d_{ff}$ the FFN's inner width. The two matrix multiplications are an up-projection of shape $d_{model} \times d_{ff}$ and a down-projection of shape $d_{ff} \times d_{model}$. Counting a multiply-accumulate as two FLOPs, the dense FFN cost per token is

$$
\text{FLOPs}_{\text{dense}} \approx 2 \cdot d_{model} \cdot d_{ff} \;+\; 2 \cdot d_{ff} \cdot d_{model} \;=\; 4 \cdot d_{model} \cdot d_{ff},
$$

where the first term is the up-projection, the second is the down-projection, and the factor of $2$ in each converts multiply-accumulates into raw FLOPs. This is the familiar $O(d_{ff})$ per-token cost (with $d_{model}$ held fixed). Now replace the single FFN with $N$ experts, each itself an FFN of inner width $d_{ff}$, and route each token to its top $K$ experts. The token only pays for the $K$ experts it actually visits, so its compute is

$$
\text{FLOPs}_{\text{MoE/token}} \approx K \cdot 4 \cdot d_{model} \cdot d_{ff},
$$

while the *parameter* count of the FFN block has grown by a factor of $N$ (there are now $N$ expert FFNs instead of one). Here $N$ is the number of experts and $K$ is the number of experts each token is routed to. The slogan is therefore: **you buy $N\times$ the parameters for $K\times$ the dense per-token FLOPs.** Equivalently, the activated fraction of the expert parameters is $K/N$.

Consider a concrete configuration: $d_{model} = 1024$, $d_{ff} = 4096$, $N = 8$ experts, $K = 2$. The dense FFN costs $4 \cdot 1024 \cdot 4096 = 16{,}777{,}216 \approx 1.68 \times 10^7$ FLOPs per token. The MoE layer holds $8 \times$ those FFN parameters — eight separate $1024\times4096$ up-projections and $4096\times1024$ down-projections — yet each token activates only $K=2$ of them, costing $2 \cdot 16{,}777{,}216 = 33{,}554{,}432 \approx 3.36 \times 10^7$ FLOPs per token. So for $2\times$ the per-token compute of a single dense FFN, the model wields $8\times$ the FFN parameters, of which $2/8 = 25\%$ are activated per token. The router itself adds a negligible $d_{model} \cdot N = 1024 \cdot 8 = 8192$ multiply-accumulates per token to compute the gate logits — about $0.05\%$ of one expert's cost — which is why we can afford to treat routing as "free" relative to the experts.

### Conditional computation and its lineage

The mechanism that lets a network spend different amounts of compute on different inputs is called **conditional computation**: parts of the network are switched on or off as a function of the input, rather than always executing. In an MoE the switch is the router's top-$K$ selection, which is *data-dependent* (it reads the token) and *sparse* (it activates a small subset). Conditional computation is what converts extra parameters into extra capacity without converting them into proportional extra cost.

The idea has a long lineage, and each milestone solved a problem the previous one exposed. **Jacobs, Jordan, Nowlan, and Hinton (1991)** introduced the original "adaptive mixtures of local experts," in which a softmax gating network learns to partition the input space among several expert networks, each specializing on a region. This established the template — a gate plus experts trained jointly — but used a dense softmax over a handful of experts with no notion of sparsity at scale.

**Shazeer et al. (2017), "Outrageously Large Neural Networks,"** scaled the idea to thousands of experts inside an LSTM language model by making the gate genuinely sparse: only the top-$K$ experts run per token. They introduced *noisy top-$K$ gating* to encourage exploration and an explicit load-balancing loss to stop a few experts from monopolizing traffic. This is the paper that turned MoE from a small ensemble trick into a tool for billion-parameter models.

**Lepikhin et al. (2021), GShard,** carried sparse MoE into the transformer and, crucially, into *distributed* training. GShard introduced expert parallelism with an all-to-all dispatch/combine, a per-expert *capacity* limit so that buffers have fixed size, and the engineering scaffolding (sharding annotations) to train a 600-billion-parameter translation model. The capacity-factor idea — pad or drop tokens so every expert receives a fixed-size batch — comes from here.

**Fedus, Zoph, and Shazeer (2022), Switch Transformer,** simplified routing to its extreme: top-1, a single expert per token. This dramatically cut communication and compute, and the paper showed that with the right capacity factor and a simplified auxiliary loss, top-1 routing is stable and scales to a trillion parameters. The auxiliary loss convention this document adopts — with the explicit factor of $N$ — is the Switch convention.

**Zhou et al. (2022), Expert Choice routing,** inverted the selection direction. Instead of each token choosing its top experts, each expert chooses its top-$C$ tokens. This guarantees perfect load balance by construction (every expert gets exactly $C$ tokens) and eliminates dropped tokens, at the cost of allowing a token to be picked by a variable number of experts (possibly zero).

**Jiang et al. (2024), Mixtral 8x7B,** brought sparse MoE into a widely deployed open-weight decoder-only LLM: 8 experts, top-2 routing, with the top-$K$ gate weights renormalized to sum to one over the selected subset. Mixtral demonstrated that MoE is not just a research curiosity but a practical way to ship a model with the quality of a much larger dense model at the inference cost of a smaller one.

### The core MoE equation

With that lineage in place, the layer's forward computation is

$$
y = \sum_{i \in \text{Top-K}} g_i(x) \cdot E_i(x),
$$

where $x \in \mathbb{R}^{d_{model}}$ is the input token representation, $E_i(\cdot)$ is the FFN of expert $i$ mapping $\mathbb{R}^{d_{model}} \to \mathbb{R}^{d_{model}}$, $g_i(x) \in \mathbb{R}$ is the scalar *gating weight* (combine weight) the router assigns to expert $i$ for this token, $\text{Top-K}$ is the set of $K$ expert indices with the highest router logits for $x$, and $y \in \mathbb{R}^{d_{model}}$ is the layer output. The summation runs only over the selected experts; experts outside $\text{Top-K}$ contribute nothing.

The selection induces a *sparsity mask*. Concretely, the router computes a gate weight for every expert, but for experts not in $\text{Top-K}$ the effective combine weight is exactly zero — they are masked out before the sum. This has a precise and important consequence for learning: because the product term for a non-selected expert is $g_i(x)\cdot E_i(x) = 0 \cdot E_i(x)$, both the term and its derivative with respect to that expert's parameters vanish. The gradient $\partial y / \partial \theta_i = g_i(x)\, \partial E_i(x)/\partial \theta_i = 0$ when $g_i(x)=0$. So **non-selected experts receive zero gradient on that token** — they are neither rewarded nor penalized, and they do not move. This is exactly what makes the forward pass sparse in compute *and* the backward pass sparse in gradient, but it is also the seed of the load-balancing problem in Section 1.3: an expert that is never selected is never trained, and an expert that is never trained never becomes worth selecting.

---

## 1.2 The Gating Network in Depth

The gating network is a single learned projection $W_g \in \mathbb{R}^{d_{model} \times N}$ that maps a token to $N$ logits, one per expert, followed by a selection rule and a normalization. Everything interesting about MoE behavior is determined by how those three steps — score, select, normalize — are defined.

**Softmax gating (Jacobs).** The simplest gate computes logits $x W_g \in \mathbb{R}^N$ and a full softmax $G(x) = \text{softmax}(x W_g)$, then uses all experts weighted by these probabilities. This is dense (no sparsity) and serves as the conceptual baseline. The probabilities $p_i = G(x)_i$ are smooth and differentiable in $W_g$, which is what makes the gate trainable by gradient descent at all. In a sparse MoE we keep this softmax as the *probability* model but then restrict the *compute* to the top-$K$ experts.

**Noisy top-$K$ gating (Shazeer).** To both sparsify and improve exploration, Shazeer adds tunable Gaussian noise to the logits before selecting the top-$K$:

$$
G(x) = \text{softmax}\!\big(\text{KeepTopK}(x W_g + \varepsilon \cdot \text{softplus}(x W_{noise}),\, K)\big),
$$

where $W_g \in \mathbb{R}^{d_{model}\times N}$ is the clean gating projection, $W_{noise} \in \mathbb{R}^{d_{model}\times N}$ is a learned per-expert noise-scaling projection, $\varepsilon \sim \mathcal{N}(0, I_N)$ is standard Gaussian noise sampled fresh per token, $\text{softplus}(z) = \log(1+e^z)$ ensures the noise standard deviation is strictly positive, and $\text{KeepTopK}(\cdot, K)$ is the masking operator that keeps the $K$ largest entries of its input vector and sets all others to $-\infty$. The noise matters for two reasons. First, it *aids exploration*: early in training the clean logits are nearly random, and noise lets experts that are slightly behind occasionally win a token, get a gradient, and improve — without noise the initial ordering can freeze. Second, it *aids load balancing*: the noise is a learned, per-expert tie-breaker that the load-balancing loss can shape to spread tokens out. The $\text{softplus}$ wrapper is essential because a standard deviation must be positive; using the raw $xW_{noise}$ could produce negative or zero scales and make the noise meaningless or unstable. Setting the masked logits to $-\infty$ *before* the softmax guarantees that $e^{-\infty} = 0$, so non-selected experts get exactly zero probability and the surviving probabilities renormalize cleanly over the kept set.

**Switch top-1 routing (Fedus).** Switch Transformer takes $K=1$: each token goes to exactly one expert, the argmax of the logits, with combine weight equal to that expert's softmax probability. Top-1 is stable for a simple reason — there is no interaction between co-selected experts, so the gradient signal per token is clean and the all-to-all communication is halved relative to top-2. To keep per-expert buffers a fixed size, Switch uses the **capacity-factor trick**: each expert is allocated a buffer of $C$ token slots (see the formula below), and once an expert's buffer is full, additional tokens routed to it *overflow* — they are dropped from the expert computation and passed through via the residual connection only. Buffer overflow is the price of fixed-size buffers; the capacity factor is the knob that trades memory and padding waste against the dropped-token rate.

**Expert Choice routing (Zhou).** Expert Choice flips the quantifier. Rather than each token selecting its top experts, each expert selects its top-$C$ tokens by gate score. Because every expert fills exactly $C$ slots, **load is perfectly balanced by construction** and no auxiliary loss is needed to enforce balance. It also **eliminates dropped tokens in the per-expert sense** — no expert ever overflows because it stops at exactly $C$. The trade-off is that a given token may be chosen by many experts, by exactly one, or by *none*; a token chosen by zero experts is effectively dropped from the MoE update (again falling back to the residual). The gradient flow changes too: because selection is per-expert, the competition that shapes $W_g$ is among tokens for an expert's slots rather than among experts for a token's budget, which empirically produces different specialization patterns.

**Token dropping and the capacity factor.** The capacity of each expert is

$$
C = \text{capacity\_factor} \times \frac{\text{tokens}}{N},
$$

where $\text{tokens}$ is the number of tokens in the batch being routed (typically $\text{batch} \times \text{seq}$), $N$ is the number of experts, and $\text{capacity\_factor} \ge 1$ is a slack multiplier. The ratio $\text{tokens}/N$ is the load each expert would carry under perfect balance; the capacity factor adds headroom so that experts which are slightly over-subscribed do not immediately overflow. With $\text{tokens}=4096$, $N=8$, and $\text{capacity\_factor}=1.25$, each expert's buffer holds $C = 1.25 \times 4096/8 = 1.25 \times 512 = 640$ tokens. When an expert is assigned more than $C$ tokens, the surplus must be dropped, and the two common policies are **FIFO** (keep the first $C$ tokens in sequence order, drop the rest) and **random** (keep a random subset of size $C$). Dropping hurts training because a dropped token gets no expert transformation at that layer — it propagates only its residual — so its representation is under-processed and its router gradient for that step is degraded; a high drop rate effectively reduces the model's depth for the unlucky tokens and biases learning toward whichever tokens happen to be kept.

---

## 1.3 Load Balancing: The Expert Collapse Problem

This is the most important section in the document, because a sparse MoE that is not actively balanced will reliably destroy itself.

**What expert collapse is.** The failure mode is a positive feedback loop with *rich-get-richer* dynamics. Recall from Section 1.1 that only selected experts receive gradients. Suppose at initialization a few experts are marginally preferred by the router. Those experts get more tokens, hence more gradient, hence they improve faster, hence the router prefers them even more strongly, hence they get still more tokens. Meanwhile the neglected experts receive few or no tokens, never improve, and become permanently uncompetitive. In the limit the router sends almost all tokens to a small clique of experts and the rest of the parameters are dead weight — the model has effectively collapsed back to a dense model with a fraction of its experts, having paid for capacity it cannot use.

**What collapse looks like empirically.** The cleanest signal is the **routing entropy**

$$
H = -\sum_{i=1}^{N} p_i \log p_i,
$$

where $p_i$ is the (batch-averaged) routing probability assigned to expert $i$. A healthy router spreads probability mass, giving entropy near $\log N$ (uniform) or at least near $\log K$ (mass concentrated on a healthy rotating top-$K$); a collapsed router concentrates mass on one or two experts, driving $H$ toward $0$. A second signal is the **load imbalance ratio** $\max_i f_i / \text{mean}_i f_i$, where $f_i$ is the fraction of tokens dispatched to expert $i$; under collapse one expert's load dwarfs the mean and this ratio blows up well above its healthy value near $1$.

**The auxiliary load-balancing loss.** To counteract the feedback loop we add a differentiable penalty that pushes the router toward uniform usage. Using the canonical Switch Transformer form, *with* the factor of $N$,

$$
L_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i,
$$

where $N$ is the number of experts; $f_i$ is the *fraction of tokens dispatched to expert $i$* — a counter obtained by averaging the hard top-$K$ assignment indicator over the batch, which is **detached / non-differentiable** (you cannot backpropagate through the discrete argmax); $P_i$ is the *mean over the batch of the softmax routing probability for expert $i$* — a smooth, **differentiable** quantity; and $\alpha$ is the loss coefficient. The product $f_i P_i$ is summed over experts and scaled by $\alpha N$.

It is worth understanding the two reference values of this loss. At **perfect balance** every $f_i = 1/N$ and every $P_i = 1/N$, so $\sum_i f_i P_i = N \cdot (1/N)(1/N) = 1/N$, and $L_{aux} = \alpha \cdot N \cdot (1/N) = \alpha$. At **full collapse**, where one expert $j$ takes all tokens, $f_j \to 1$ and $P_j \to 1$ while all others vanish, so $\sum_i f_i P_i \to 1$ and $L_{aux} \to \alpha \cdot N \cdot 1 = \alpha N$. The loss therefore ranges from $\alpha$ (balanced) up to $\alpha N$ (collapsed), and the factor of $N$ is exactly what makes the *balanced* value a clean, $N$-independent $\alpha$ — a convenient normalization when tuning across different expert counts. As a worked check with $N=8$: balanced gives $L_{aux}=\alpha$, and a fully collapsed router gives $L_{aux}=8\alpha$, an $8\times$ penalty that the optimizer feels as strong pressure to spread tokens out.

**Why multiplying a detached $f_i$ by a differentiable $P_i$ still trains.** This looks suspicious — why penalize using a counter you cannot differentiate? The key is that the gradient does not need to flow through $f_i$; it flows entirely through $P_i$:

$$
\frac{\partial L_{aux}}{\partial W_g} = \alpha N \sum_{i=1}^{N} f_i \, \frac{\partial P_i}{\partial W_g}.
$$

Here $f_i$ acts as a fixed, per-expert *weight* on how hard we push down the probability $P_i$. Experts that are currently over-used have large $f_i$, so the loss puts large weight on reducing their $P_i$; under-used experts have small $f_i$ and are pushed down little. The net effect of minimizing $\sum_i f_i P_i$ is to *move probability mass off the experts that are already crowded and onto the experts that are starving*, which is precisely the balancing behavior we want — and it is achieved purely through the differentiable $P_i$ channel, with $f_i$ supplying the (non-differentiable but perfectly usable) information about where the crowding currently is.

**Router z-loss for numerical stability and logit regularization.** A second auxiliary term, from ST-MoE (Zoph et al., 2022), penalizes large router logits:

$$
L_z = \beta \cdot \frac{1}{B}\sum_{b=1}^{B}\left(\log\sum_{i=1}^{N} e^{x_b W_g}\right)^2,
$$

where $B$ is the number of tokens in the batch, $x_b W_g \in \mathbb{R}^N$ is the router logit vector for token $b$, $\log\sum_i e^{x_b W_g}$ is the log-sum-exp of those logits (computed in code with `torch.logsumexp` for numerical safety), and $\beta$ is the z-loss coefficient. The quantity inside the square is the log-partition function of the softmax; squaring it and minimizing pushes the logits toward smaller magnitudes, which keeps the subsequent $\exp$ inside the dynamic range of low-precision floats and acts as a gentle regularizer that discourages the router from becoming pathologically overconfident. A useful reference value: when all logits are zero, $\sum_i e^0 = N$, so $\log\sum_i e^{x_b W_g} = \log N$ and $L_z = \beta(\log N)^2$. For $N=8$, $\log 8 \approx 2.079$, so the zero-logit z-loss is $\beta \cdot 2.079^2 \approx 4.32\,\beta$ — small for the recommended $\beta$ range, confirming it does not dominate the main loss at initialization.

**Router z-score normalization as an alternative.** Instead of penalizing large logits after the fact, you can normalize them up front: maintain a running standard deviation of the router logits and divide the logits by it before the softmax, so the softmax always sees a controlled scale. This is *z-score normalization* of the router. Prefer z-score normalization when the instability is primarily about *logit scale drift* across training and you want a parameter-free, always-on guard; prefer the z-loss when you want a soft, tunable pressure that you can anneal and that also serves as a mild regularizer without hard-coding a normalization into the forward pass. In practice many systems use z-loss as the default and reach for z-score normalization only if z-loss alone cannot tame the logits.

**Choosing $\alpha$ and $\beta$.** The auxiliary weight $\alpha$ is typically in $[0.001, 0.01]$ and the z-loss weight $\beta$ in $[0.0001, 0.001]$. Tune them by monitoring three quantities together. If the **routing entropy** is collapsing toward zero or the **imbalance ratio** is climbing above roughly $1.5$, $\alpha$ is too low — raise it toward $0.01$. If the **auxiliary-loss magnitude** is a large fraction of the total loss (a sign it is fighting the language-modeling objective and degrading quality), $\alpha$ is too high — lower it toward $0.001$. For $\beta$, watch the router logits and the z-loss value: if logits grow unbounded or you see overflow in low precision, raise $\beta$; if the z-loss is so strong that the router becomes mushy (entropy pinned near $\log N$ with no specialization), lower it. The healthy operating point is the smallest $\alpha$ that keeps the imbalance ratio under control and the smallest $\beta$ that keeps the logits in range.

---

## 1.4 Numerical Stability

Sparse routing pushes a softmax through a discrete selection in low precision, so numerical care is not optional.

**Softmax overflow and the max-subtraction trick.** A naive softmax computes $e^{x_i}$ directly, and for even moderately large logits $e^{x_i}$ overflows to infinity. The fix is to subtract the per-token maximum logit $m = \max_b x_b$ before exponentiating. This is *mathematically identical* to the naive softmax, not an approximation, because

$$
\frac{e^{a - m}}{\sum_b e^{b - m}} = \frac{e^{a} e^{-m}}{e^{-m}\sum_b e^{b}} = \frac{e^{a}}{\sum_b e^{b}},
$$

where $a$ is the logit being normalized, $b$ ranges over all logits, and $m$ is any constant (the maximum is chosen so the largest exponent becomes $e^0 = 1$ and nothing overflows; the smallest underflow to $0$ harmlessly). The common factor $e^{-m}$ cancels exactly between numerator and denominator, so subtracting the max changes the representable range without changing the value.

**log_softmax + NLL versus log(softmax).** When you need log-probabilities — for example to form a cross-entropy term over expert assignments — never compute `log(softmax(x))` as two steps. The intermediate softmax can underflow to $0$ and then $\log 0 = -\infty$ poisons the gradient. Use a fused `log_softmax` followed by negative-log-likelihood, which computes $\log\text{softmax}(x)_i = x_i - \log\sum_j e^{x_j}$ directly via a stable log-sum-exp and never materializes a tiny probability that it then has to take the log of. The numerical difference is the gap between a clean gradient and a `NaN`.

**Gradient sparsity in routing.** As established in Section 1.1, only routed tokens deliver gradients to their experts. This is correct and intended, but it means a misconfigured router silently starves experts of training signal. Monitor it directly: track the fraction of experts that received at least one token over a window of steps, and the per-expert token counts. If an expert's count is zero across many steps, its parameters are frozen and the symptom will look like a capacity or routing bug rather than what it is — a gradient-flow problem.

**Mixed precision.** Modern training uses bfloat16, whose format has an 8-bit exponent and a 7-bit mantissa. The wide 8-bit exponent (the same range as float32) means bf16 rarely overflows, but the narrow 7-bit mantissa means it carries only about two to three decimal digits of precision. The practical rule for MoE is to keep the precision-sensitive, accumulation-heavy parts in float32 and let the throughput-bound matmuls run in bf16. Specifically, compute the **gating logits**, the **softmax**, and the **auxiliary-loss accumulation** in float32 — these involve sums over experts or over the batch where small errors compound and where a bf16 mantissa would corrupt the probabilities and the balance statistics — while the **expert FFN matmuls** are safe in bf16 because they are large GEMMs whose rounding error averages out and whose dynamic range bf16 handles comfortably.

**Breaking symmetry at initialization.** A subtle but fatal trap: if all experts are initialized with identical weights, then on the very first forward pass every expert computes the identical function, the router has no basis to prefer one over another, routing is arbitrary, and — because identical experts receive symmetric gradients — they *stay* identical. The experts never diversify and the MoE degenerates into a single replicated FFN. The remedy is to initialize each expert with a **different random seed** so that experts start as distinct functions, giving the router real differences to exploit and the load-balancing loss real diversity to shape. Identical init implies identical routing on the first pass implies no diversification ever; distinct init breaks the symmetry from step zero.

---

## 1.5 Distributed Training Considerations

At scale the $N$ experts do not fit on one device, so MoE is trained with **expert parallelism**: experts are sharded across devices, with each device owning a disjoint subset of experts. Because any token may be routed to an expert living on any device, the layer must shuffle tokens to the devices that hold their chosen experts, run the experts locally, and shuffle the results back. This is the **all-to-all dispatch/combine** pattern.

The two collectives bracket the expert computation. The first all-to-all (*dispatch*) sends each token from the device where it currently lives to the device that owns its selected expert. The second all-to-all (*combine*) sends each expert's output back to the token's original device so it can be written into the residual stream. The communication volume per MoE layer per forward pass is

$$
\text{bytes} = 2 \times T \times d_{model} \times \text{sizeof(dtype)},
$$

where $T$ is the number of tokens routed, $d_{model}$ is the hidden size, $\text{sizeof(dtype)}$ is the bytes per element (2 for bf16), and the **factor of 2 counts the two all-to-alls** — one for dispatch and one for combine, each moving roughly one token-vector's worth of data per token. With $T = 4096$ tokens, $d_{model}=1024$, and bf16, that is $2 \times 4096 \times 1024 \times 2 = 16{,}777{,}216$ bytes $\approx 16$ MiB of all-to-all traffic per MoE layer per forward pass — and the same again on the backward pass — which is why interconnect bandwidth is the binding constraint for MoE at scale.

The data layout is transformed around the collectives. Tokens enter as a dense $[\text{batch}, \text{seq}, d_{model}]$ tensor; the router flattens and groups them into a capacity-padded $[\text{num\_experts}, \text{capacity}, d_{model}]$ tensor (every expert gets exactly $C$ slots, padded if under-subscribed, with overflow dropped if over-subscribed) so that the all-to-all moves fixed-size, contiguous blocks; after the experts run, the inverse transform scatters the outputs back to $[\text{batch}, \text{seq}, d_{model}]$ using the saved routing indices. The fixed-size buffer is exactly why capacity (Section 1.2) exists — variable-size all-to-all is far harder to schedule efficiently.

```
   Device 0           Device 1           Device 2           Device 3
 [tokens 0..]       [tokens ..]        [tokens ..]        [tokens ..]
      |                  |                  |                  |
      v                  v                  v                  v
  +----------------------------------------------------------------+
  |   route locally: assign each token its top-K expert id(s)       |
  +----------------------------------------------------------------+
      |   \   \            /   |   \           /   |               |
      |    \   \          /    |    \         /    |   (each token sent
      v     v   v        v     v     v       v     v    to the device that
  =================  ALL-TO-ALL  (DISPATCH)  ====================
      |     |   |        |     |     |       |     |    owns its expert)
      v     v   v        v     v     v       v     v
  [expert 0,1]       [expert 2,3]       [expert 4,5]       [expert 6,7]
   run FFN            run FFN            run FFN            run FFN
      |                  |                  |                  |
  =================  ALL-TO-ALL  (COMBINE)  =====================
      |                  |                  |                  |
      v                  v                  v                  v
  weight by g_i, scatter back to original token positions, add residual
```

Gradient synchronization differs between the router and the experts. The **router weights $W_g$ are replicated** across all data-parallel devices (every device runs the same small router on its local tokens), so their gradients are all-reduced like any data-parallel parameter. The **expert weights are sharded** (each device holds and updates only its own experts), so an expert's gradient is computed only from the tokens that were dispatched to it and is *not* all-reduced across the expert-parallel group. This has a direct implication for the auxiliary loss: $L_{aux}$ depends on global statistics $f_i$ and $P_i$ that span all devices, so those statistics must be reduced across the expert-parallel group before the loss is formed, and the resulting gradient flows back into the replicated router weights consistently on every device.

The choice between **expert parallelism and tensor parallelism** depends on scale. Tensor parallelism splits each matmul across devices and synchronizes with all-reduce on every layer; it has fine-grained, latency-sensitive communication and is best within a single high-bandwidth node. Expert parallelism communicates only twice per MoE layer (the two all-to-alls) but moves whole tokens; it tolerates a bit more latency and scales naturally as you add experts, making it the better fit across nodes and at large expert counts. In practice large MoE models combine them — tensor parallelism inside a node for the dense attention and projections, expert parallelism across nodes for the experts — so that each form of parallelism runs where its communication pattern is cheapest.

---

## 1.6 Hyperparameter Guide

The MoE layer exposes about ten knobs, and they interact strongly: capacity, expert count, and top-$K$ jointly determine the drop rate; the two auxiliary weights jointly determine how hard balance is enforced; and the noise and dropout terms jointly determine how much exploration the router does. The table below is the working reference, and the prose after it explains the couplings you must respect when changing more than one knob at a time.

| Parameter | Typical range | What it controls | How to diagnose if wrong | How to fix |
|---|---|---|---|---|
| `num_experts` ($N$) | 8 – 128 | Total expert parameters / model capacity | Many dead experts (poor utilization); or under-capacity if too few | Match $N$ to data scale; if many die, lower $N$ or raise $\alpha$ |
| `top_k` ($K$) | 1 – 2 | Experts activated per token; per-token FLOPs | $K=1$ unstable/noisy; large $K$ wastes compute | Use 1 (Switch) for stability, 2 (Mixtral) for quality |
| `capacity_factor` | 1.0 – 2.0 | Expert buffer slack; drop rate vs. waste | High overflow fraction (factor too low); high padding waste (too high) | Raise to cut drops; lower to cut memory/padding |
| `expert_dropout` | 0.0 – 0.1 | Regularization inside expert FFNs | Overfitting (too low); underfitting (too high) | Tune like ordinary dropout; default 0.0–0.05 |
| `alpha` (aux weight) | 0.001 – 0.01 | Strength of load-balancing pressure | Collapse / high imbalance (too low); quality loss (too high) | Raise to balance, lower if $L_{aux}/L_{total}$ large |
| `beta` (z weight) | 0.0001 – 0.001 | Logit-magnitude regularization / stability | Logit overflow (too low); mushy router (too high) | Raise for stability, lower if entropy pinned high |
| `noise_std_init` | 0.1 – 1.0 | Initial routing-noise scale (exploration) | Frozen ordering (too low); chaotic routing (too high) | Raise to encourage exploration, anneal over training |
| `d_model` | 512 – 8192 | Residual/hidden width; router input dim | Underfitting (too small); cost blowup (too large) | Set by overall model size budget |
| `d_ff` | 2048 – 32768 | Expert FFN inner width; per-expert FLOPs | Under/over-capacity per expert | Commonly $4\times d_{model}$; tune with $N$ |
| `jitter_noise` | 0.0 – 0.1 | Multiplicative input jitter for exploration | Over-confident routing (too low); instability (too high) | Small value (~0.01) early, decay to 0 |

The couplings are what make tuning subtle. The **drop rate** is a joint function of `capacity_factor`, `num_experts`, and `top_k`: raising `top_k` from 1 to 2 doubles the number of token-expert assignments competing for the same per-expert buffers, so a capacity factor that was comfortable at $K=1$ may overflow at $K=2$ — when you increase $K$, increase `capacity_factor` too. The **balance pressure** couples `alpha` with `num_experts`: because the auxiliary loss carries the factor $N$, its balanced value stays at $\alpha$ regardless of $N$, but the *collapsed* value scales as $\alpha N$, so a given $\alpha$ exerts proportionally more corrective force as you add experts — when you scale $N$ up you can often hold $\alpha$ fixed. The **exploration** terms `noise_std_init` and `jitter_noise` both fight collapse from the other side: rather than penalizing crowding after the fact (what `alpha` does), they perturb the routing decision so that under-used experts occasionally win tokens and start learning. The usual recipe is to keep exploration noise meaningful early and anneal it toward zero, letting `alpha` carry balance once the experts have differentiated. Finally, `d_ff` and `num_experts` together set the expert parameter budget; if you grow $N$ you may shrink $d_{ff}$ to hold total parameters fixed, trading fewer-but-wider experts for more-but-narrower ones.

---

## 1.7 Monitoring and Debugging

A sparse MoE can be silently broken while its loss curve looks fine, so the router must be instrumented with a small set of metrics, each with a known healthy band. The following are the metrics to log every few steps, with their formulas and the ranges that distinguish health from pathology.

**Routing entropy.** Compute $H = -\sum_i p_i \log p_i$ over the batch-averaged routing probabilities $p_i$. A healthy router sits somewhere between $\log K$ and $\log N$ — it concentrates mass on a healthy, rotating set of top experts (lower bound near $\log K$) without ignoring the rest (upper bound near $\log N$). For $N=8$, $\log N \approx 2.08$ and $\log K \approx 0.69$ at $K=2$, so a healthy $H$ lives roughly in $[0.7, 2.1]$. An $H$ collapsing toward $0$ means the router is funneling everything into one expert — collapse in progress — and calls for raising $\alpha$ or the exploration noise.

**Load imbalance ratio.** Compute $\max_i f_i / \text{mean}_i f_i$, where $f_i$ is the fraction of tokens dispatched to expert $i$. Since $\text{mean}_i f_i = 1/N$ by construction, this is just $N \cdot \max_i f_i$. A perfectly balanced router gives $1.0$; the practical acceptable ceiling is about $1.5$ (the busiest expert handles at most 50% more than its fair share). Values climbing past $\sim 2$ signal that balance is failing and $\alpha$ is too low.

**Expert utilization histogram.** Plot $f_i$ across all $N$ experts. A healthy histogram is roughly flat; a spiky histogram with a few tall bars and many near-zero bars is the visual signature of collapse, and it tells you not just *that* balance failed but *which* experts are starving — useful for targeted remediation.

**Overflow fraction.** Compute the dropped-token rate as (tokens dropped due to capacity) / (total tokens). A few percent is tolerable; a persistently high overflow fraction means `capacity_factor` is too low for the current imbalance, and you should either raise the capacity factor or fix the underlying imbalance (which is the real cause when one expert overflows while others sit empty).

**Auxiliary-loss fraction.** Track $L_{aux} / L_{total}$. The auxiliary loss is a means, not an end — it should nudge balance without competing with the language-modeling objective. If $L_{aux}/L_{total}$ exceeds roughly $5\%$, $\alpha$ is too high and is degrading model quality; lower it. If it is essentially zero while imbalance is rising, $\alpha$ is too low; raise it.

**Dead-expert detection and remediation.** An expert is *dead* if it receives less than about $1\%$ of tokens sustained over roughly 100 steps — it is no longer being trained and is wasting its parameters. Detect it by maintaining a running per-expert token count over a sliding window and flagging any expert below the threshold. The two standard remedies are to **reinitialize the dead expert's weights with small random noise** so it becomes a fresh, slightly-different function that the router can rediscover, or to **temporarily reduce $K$** (or, conversely, briefly raise exploration noise) so that routing decisions reshuffle and the starving expert gets a chance to win tokens and climb back into use. Combined with a sufficiently large $\alpha$, these interventions keep the full expert population alive and earning the parameter budget the MoE was built to exploit.

Taken together, these metrics form a tight diagnostic loop: routing entropy and the imbalance ratio tell you *whether* the router is healthy, the utilization histogram and dead-expert detector tell you *where* it is failing, the overflow fraction tells you whether capacity is the bottleneck, and the auxiliary-loss fraction tells you whether your cure has become worse than the disease. Watching all five is what turns MoE training from a fragile art into a controllable engineering process.
