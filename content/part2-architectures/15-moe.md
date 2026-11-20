---
id: ch15-moe
summary: Mixture-of-experts layers grow parameters without growing compute per token. Implement top-k routing, write the load-balancing loss that stops every token piling onto one expert, and predict how many tokens a capacity limit throws away.
udl: "—"
---

## More parameters, same FLOPs

A dense transformer MLP applies every parameter to every token. A **mixture-of-experts (MoE)** layer has $E$ expert MLPs plus a small **router**. Each token is sent to only its top-$k$ experts (usually $k = 1$ or $2$), and their outputs are mixed using the router's weights:
$$y = \sum_{e \in \text{top-}k(x)} \tilde g_e(x)\, \mathrm{Expert}_e(x),\qquad \tilde g = \text{router softmax, renormalised over the chosen } k.$$
So parameters scale with $E$ while compute per token scales with $k$. That's how Switch Transformer, Mixtral and many frontier models reach very large parameter counts.

**The catch is load balance.** Routers collapse easily: a few experts win early, get trained more, and win even more. Two standard tools counter this: an **auxiliary load-balancing loss**, and **expert capacity** limits that drop overflow tokens, which then pass through the residual connection unchanged.

> UDL doesn't cover MoE, so this chapter is self-contained.

::::exercise{id=moe-topk-routing type=implement difficulty=2 time=20 runtime=browser udl="—" tags="moe"}
# Top-k routing

:::prereq
- **Router:** $p = \mathrm{softmax}(x W_g)$, with shape `(T, E)`, giving one probability per expert per token.
- **Top-k:** choose the $k$ largest probabilities per token, then **renormalise** them to sum to 1. That's the Mixtral convention. Switch with $k=1$ uses the raw probability instead.
- **Dispatch by expert, not by token.** For each expert, gather the tokens that chose it, run the expert once on that batch, and scatter-add the weighted outputs back. Experts nobody chose cost nothing.
- `np.take_along_axis(probs, idx, axis=-1)` picks `probs[t, idx[t, j]]`.
:::

:::task
Implement `moe_forward(x, Wg, experts, k=2)`. Here `x` is `(T, D)`, `Wg` is `(D, E)`, and `experts` is a list of `E` matrices of shape `(D, D_out)`; each expert is just a linear map, to keep the focus on routing. Return `(out, idx, gates)`, with `out: (T, D_out)`, `idx: (T, k)` the chosen experts sorted by descending probability, and `gates: (T, k)` the renormalised weights.
:::

:::starter{lang=python}
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def moe_forward(x, Wg, experts, k=2):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def moe_forward(x, Wg, experts, k=2):
    probs = softmax(x @ Wg)
    idx = {{ np.argsort(-probs, axis=-1)[:, :k] | np.argsort(probs, axis=-1)[:, ::-1][:, :k] | np.argsort(-probs, axis=1)[:, :k] ? Indices of the $k$ largest probabilities per token, largest first. `argsort` sorts ascending, so negate. }}
    gates = np.take_along_axis(probs, idx, axis=-1)
    gates = {{ gates / gates.sum(axis=-1, keepdims=True) | gates / gates.sum(-1, keepdims=True) | gates / gates.sum(axis=1, keepdims=True) ? Renormalise over the chosen $k$ so each token's mixing weights sum to 1. }}
    out = np.zeros((x.shape[0], experts[0].shape[1]))
    for e, W in enumerate(experts):
        rows, slot = {{ np.nonzero(idx == e) | np.where(idx == e) ? Which tokens chose expert `e`, and in which of their $k$ slots. Each token picks an expert at most once. }}
        if rows.size:
            out[rows] += {{ gates[rows, slot, None] * (x[rows] @ W) | gates[rows, slot][:, None] * (x[rows] @ W) | (x[rows] @ W) * gates[rows, slot, None] ? Run the expert once on all its tokens, then weight each row by that token's gate. The gate is shaped `(n, 1)` to broadcast over `D_out`. }}
    return out, idx, gates
:::

:::solution{lang=python}
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def moe_forward(x, Wg, experts, k=2):
    probs = softmax(x @ Wg)
    idx = np.argsort(-probs, axis=-1)[:, :k]
    gates = np.take_along_axis(probs, idx, axis=-1)
    gates = gates / gates.sum(axis=-1, keepdims=True)
    out = np.zeros((x.shape[0], experts[0].shape[1]))
    for e, W in enumerate(experts):
        rows, slot = np.nonzero(idx == e)
        if rows.size:
            out[rows] += gates[rows, slot, None] * (x[rows] @ W)
    return out, idx, gates
:::

:::explain
[L9-12] **Route.** The router is a single linear layer, and it's the only part every token pays for in full. Renormalising after top-$k$ means the expert outputs are mixed by a proper convex combination.

[L13-17] **Dispatch per expert.** `np.nonzero(idx == e)` lists the tokens routed to `e`. One batched matmul per *expert*, rather than per token, is what makes MoE efficient on accelerators. The test that sets unused experts to NaN proves they never run. In real systems this step is an all-to-all communication across devices ("expert parallelism"), which is often the true bottleneck.

**Gradients.** The router's top-$k$ choice isn't differentiable, but the *gate values* are. So the router learns through $\tilde g_e$ on the experts it picked, plus the auxiliary loss in the next exercise.
:::

:::tests{lang=python}
import numpy as np

def _setup(seed=0, T=12, D=6, E=5, Dout=4):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, D)), rng.normal(size=(D, E)), [rng.normal(size=(D, Dout)) for _ in range(E)]

def _ref(x, Wg, experts, k):
    probs, out = softmax(x @ Wg), []
    for t in range(len(x)):
        top = np.argsort(-probs[t])[:k]
        g = probs[t, top] / probs[t, top].sum()
        out.append(sum(g[j] * (x[t] @ experts[e]) for j, e in enumerate(top)))
    return np.array(out)

def test_matches_per_token_reference():
    x, Wg, experts = _setup()
    for k in (1, 2, 3):
        out, idx, gates = moe_forward(x, Wg, experts, k)
        assert out.shape == (12, 4) and idx.shape == gates.shape == (12, k)
        assert np.allclose(out, _ref(x, Wg, experts, k)) and np.allclose(gates.sum(-1), 1)

def test_unchosen_experts_never_run():
    x, Wg, experts = _setup(1)
    _, idx, _ = moe_forward(x, Wg, experts, 1)
    poisoned = [W if e in idx else np.full_like(W, np.nan) for e, W in enumerate(experts)]
    assert np.isfinite(moe_forward(x, Wg, poisoned, 1)[0]).all()
:::

:::hints
1. Get `idx` and `gates` right first; the tests check their shapes.
2. In the expert loop, `np.nonzero` on a 2-D boolean array returns `(rows, cols)`. The columns tell you which gate slot to use.
:::

:::interview
- How does MoE decouple parameters from compute? What's the FLOP cost per token for top-2 routing over 8 experts?
- Why renormalise the top-k gates? Why does Switch use $k = 1$?
- What's expert parallelism, and why is the all-to-all the bottleneck?
- Why can MoE models be harder to fine-tune or serve? *(Load imbalance, memory for all experts, and routing instability.)*
:::
::::

::::exercise{id=load-balancing-loss type=implement difficulty=2 time=10 runtime=browser udl="—" tags="moe,custom-losses"}
# The load-balancing auxiliary loss

:::prereq
The Switch Transformer's auxiliary loss, for $E$ experts and a batch of $T$ tokens:
$$\mathcal L_\text{aux} = E\sum_{e=1}^{E} f_e\, P_e,\qquad f_e = \tfrac1T\#\{t : \text{top-1}(t) = e\},\quad P_e = \tfrac1T\textstyle\sum_t p_{t,e}.$$
- $f_e$ is the **fraction of tokens dispatched** to expert $e$. It's a count, so not differentiable.
- $P_e$ is the **mean router probability** for $e$. That's differentiable, and it carries the gradient.

The loss is minimised at **1** when both are uniform, and reaches about $E$ when everything collapses onto one expert. It's added to the main loss with a small weight, around $10^{-2}$.
:::

:::task
Implement `load_balancing_loss(probs, top1)`, where `probs` is `(T, E)` (each row sums to 1) and `top1` is `(T,)` (the chosen expert per token).
:::

:::starter{lang=python}
import numpy as np

def load_balancing_loss(probs, top1):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def load_balancing_loss(probs, top1):
    T, E = probs.shape
    f = {{ np.bincount(top1, minlength=E) / T | np.bincount(top1, minlength=E) / len(top1) | np.bincount(top1, minlength=E) / float(T) ? The fraction of tokens sent to each expert. `minlength` makes sure experts that received nothing still get a 0 entry. }}
    P = {{ probs.mean(axis=0) | probs.mean(0) | np.mean(probs, axis=0) | probs.sum(axis=0) / T ? The mean router probability per expert, averaged over **tokens** (axis 0). }}
    return {{ E * np.sum(f * P) | E * (f * P).sum() | E * np.dot(f, P) | E * f @ P ? Scale by $E$ so a perfectly balanced router scores exactly 1, whatever the number of experts. }}
:::

:::solution{lang=python}
import numpy as np

def load_balancing_loss(probs, top1):
    T, E = probs.shape
    f = np.bincount(top1, minlength=E) / T
    P = probs.mean(axis=0)
    return E * np.sum(f * P)
:::

:::explain
[L5] **Dispatch fractions.** `bincount` counts how many tokens each expert received. Without `minlength=E`, an expert that got nothing would simply be missing, and the shapes would break, or worse, silently misalign.

[L6] **Mean probabilities.** This is the differentiable half. The gradient of $\mathcal L_\text{aux}$ with respect to $P_e$ is $E f_e$, so the router is pushed *away* from experts that are already overloaded, in proportion to their load.

[L7] **Why the product?** Dispatch follows the router (a token goes where its probability is highest), so $f$ and $P$ rise and fall together. The product is therefore large when the same few experts get both the probability mass and the tokens, and as small as about $1/E$ when both are spread evenly. The factor $E$ scales that balanced value to 1.
:::

:::tests{lang=python}
import numpy as np

def test_balanced_router_scores_one():
    E, T = 4, 400
    probs = np.full((T, E), 1 / E)
    assert np.isclose(load_balancing_loss(probs, np.arange(T) % E), 1.0)

def test_collapsed_router_scores_about_E():
    E, T = 8, 100
    probs = np.full((T, E), 1e-3)
    probs[:, 0] = 1 - 7e-3
    assert np.isclose(load_balancing_loss(probs, np.zeros(T, dtype=int)), E * (1 - 7e-3))

def test_handles_experts_that_receive_nothing():
    probs = np.random.default_rng(0).dirichlet(np.ones(6), size=50)
    loss = load_balancing_loss(probs, np.zeros(50, dtype=int))
    assert np.isfinite(loss) and np.isclose(loss, 6 * probs[:, 0].mean())
:::

:::hints
1. Two per-expert vectors of length `E`, then a weighted dot product.
2. What should `f` look like when one expert receives no tokens?
:::

:::interview
- Why is the auxiliary loss a product of $f$ (non-differentiable) and $P$ (differentiable)?
- What happens without it? *(Expert collapse: a rich-get-richer dynamic.)*
- Alternatives? *(Router z-loss for stability, expert-choice routing, and DeepSeek-style bias-based balancing with no auxiliary loss.)*
:::
::::

::::exercise{id=moe-capacity-predict type=predict difficulty=1 time=6 runtime=browser udl="—" tags="moe"}
# How many tokens get dropped?

:::task
An MoE layer has $E = 4$ experts and processes $T = 16$ tokens with **capacity factor 1.25**: each expert accepts at most $\lfloor 1.25 \cdot T/E\rfloor$ tokens, first come, first served. The router (top-1) sends **10** tokens to expert 0 and **2** to each of the other three. How many tokens are dropped?
:::

:::choices
- [ ] 0: capacity factor 1.25 leaves 25% headroom
- [x] 5
- [ ] 6
- [ ] 10
:::

:::experiment{lang=python}
import numpy as np

def dispatch(assign, E, capacity_factor):
    """Top-1 assignments -> (kept mask, capacity). Tokens beyond an expert's capacity are dropped, in order."""
    T = len(assign)
    capacity = int(capacity_factor * T / E)
    kept, load = np.zeros(T, dtype=bool), np.zeros(E, dtype=int)
    for t, e in enumerate(assign):
        if load[e] < capacity:
            kept[t], load[e] = True, load[e] + 1
    return kept, capacity

skewed = np.array([0] * 10 + [1, 1, 2, 2, 3, 3])
balanced = np.arange(16) % 4
for name, assign in [("skewed", skewed), ("balanced", balanced)]:
    for cf in (1.0, 1.25, 2.5):
        kept, cap = dispatch(assign, 4, cf)
        print(f"{name:8s}  capacity factor {cf:<4}  capacity/expert {cap:2d}   dropped {(~kept).sum():2d}")
:::

:::tests{lang=python}
import numpy as np

def test_question_drops_five():
    kept, cap = dispatch(np.array([0] * 10 + [1, 1, 2, 2, 3, 3]), 4, 1.25)
    assert cap == 5 and (~kept).sum() == 5

def test_balanced_routing_drops_nothing_at_factor_one():
    kept, _ = dispatch(np.arange(16) % 4, 4, 1.0)
    assert kept.all()
:::

:::explain
Capacity per expert is $\lfloor 1.25 \times 16/4 \rfloor = 5$. Expert 0 receives 10 tokens, so it accepts 5 and **drops 5**. The other experts receive 2 each, well under capacity, and their spare slots can't be lent out. The 25% headroom only helps when the load is *nearly* balanced.

**What happens to dropped tokens?** They skip the MoE layer, so its contribution is zero for them, and they continue through the **residual connection** unchanged. They aren't lost from the sequence, but they get less computation, and the dropping depends on the other tokens in the batch: an awkward source of non-determinism at inference time.

**Trade-off.** A bigger capacity factor drops less, but pads every expert's buffer up to capacity, which wastes compute and memory, and with expert parallelism, communication too. That's why the load-balancing loss (Exercise 15.2) matters: it makes a small capacity factor safe.
:::

:::interview
- What's the capacity factor, and what do you trade off when choosing it?
- What happens to dropped tokens?
- How do dropless MoE implementations (such as MegaBlocks) avoid capacity limits? *(Block-sparse kernels with variable-sized expert batches.)*
:::
::::
