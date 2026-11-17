---
id: ch13-attention
summary: Attention is a soft, differentiable dictionary lookup. Build it in NumPy with masks, prove why the logits are scaled by √d, then write multi-head attention in PyTorch and match `nn.MultiheadAttention` to five decimal places.
weeks: [5]
udl: "Ch. 12 — Transformers"
---

## A soft dictionary lookup

Each token emits three vectors. A **query** says what it is looking for, a **key** says what it offers, and a **value** is what it hands over if chosen. A query scores every key by dot product. Softmax turns the scores into weights, and the output is the weighted average of the values:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V .
$$

Nothing here depends on token *order*. That's why transformers need positional encodings (Ch. 14), and why masks are how you impose structure such as causality or padding.

| Symbol | Shape here | Meaning |
|---|---|---|
| $Q$ | `(…, Tq, d)` | one query per output position |
| $K$ | `(…, Tk, d)` | one key per input position |
| $V$ | `(…, Tk, dv)` | one value per input position |
| weights | `(…, Tq, Tk)` | each row sums to 1 |

**Convention note.** UDL writes data matrices as $D \times N$, with tokens as **columns**, so its formulas carry transposes in different places. This workbook uses the PyTorch convention: tokens are **rows**, and features are the last axis.

::::exercise{id=sdpa-numpy type=implement difficulty=2 time=20 runtime=browser udl="§12.2–12.3" tags="self-attention,numerical-stability,masking"}
# Scaled dot-product attention with masks

:::prereq
**Batched matmul.** `a @ b` treats every axis but the last two as a batch. `(B, H, T, d) @ (B, H, d, S)` gives `(B, H, T, S)`.

**Transposing the last two axes.** Use `np.swapaxes(k, -1, -2)`. Don't use `k.T`, which reverses *every* axis and scrambles the batch dimensions.

**Masking with $-\infty$.** To stop query $i$ attending to key $j$, set that score to $-\infty$ *before* the softmax. Then $e^{-\infty} = 0$, and the remaining weights still sum to 1. Zeroing the weights *after* the softmax breaks that normalisation.

**Causal masking.** In a decoder, position $i$ may only look at positions $j \le i$. That's the lower triangle of a $T\times T$ matrix, diagonal included.

**Stability.** Scores can be large, so subtract the row max before `exp`, exactly as in Chapter 04.
:::

:::task
Implement `causal_mask(T)` and `scaled_dot_product_attention(q, k, v, mask=None)` in NumPy. Return **both** the output and the attention weights.

| Argument | Shape | Notes |
|---|---|---|
| `q` | `(…, Tq, d)` | any number of leading batch axes |
| `k` | `(…, Tk, d)` | |
| `v` | `(…, Tk, dv)` | |
| `mask` | broadcastable to `(…, Tq, Tk)`, bool | **True = may attend** (same as PyTorch's `F.scaled_dot_product_attention`) |
| **returns** `out` | `(…, Tq, dv)` | |
| **returns** `weights` | `(…, Tq, Tk)` | rows sum to 1 |

Your code must handle 4-D inputs `(B, H, T, d)`, must not let a token see the future under a causal mask, and must stay finite for scores around $10^4$.
:::

:::starter{lang=python}
import numpy as np

def causal_mask(T):
    """-> (T, T) bool. Entry [i, j] is True when query i may attend to key j."""
    raise NotImplementedError

def scaled_dot_product_attention(q, k, v, mask=None):
    """q: (..., Tq, d), k: (..., Tk, d), v: (..., Tk, dv), mask: True = may attend.
    Returns (out: (..., Tq, dv), weights: (..., Tq, Tk))."""
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def causal_mask(T):
    return {{ np.tril(np.ones((T, T), dtype=bool)) | np.tri(T, dtype=bool) | np.tril(np.ones((T, T))).astype(bool) | ~np.triu(np.ones((T, T), dtype=bool), k=1) | ~np.triu(np.ones((T, T), dtype=bool), 1) ? Query $i$ may see keys $0..i$: the lower triangle, *including* the diagonal (a token can attend to itself). }}

def scaled_dot_product_attention(q, k, v, mask=None):
    d = q.shape[-1]
    scores = q @ {{ np.swapaxes(k, -1, -2) | np.swapaxes(k, -2, -1) | k.swapaxes(-1, -2) | k.swapaxes(-2, -1) | np.moveaxis(k, -1, -2) ? `k.T` reverses *every* axis, which scrambles the batch dimensions. Swap only the last two so that `(…, Tq, d) @ (…, d, Tk)` gives `(…, Tq, Tk)`. }} / {{ np.sqrt(d) | d ** 0.5 | np.sqrt(q.shape[-1]) | d**(1/2) ? The dot product of two random $d$-dimensional vectors has variance $d$. Dividing by its standard deviation brings the logits back to unit scale. }}
    if mask is not None:
        scores = np.where(mask, scores, {{ -np.inf | float("-inf") | -float("inf") | -1e9 ? Masked positions must get exactly zero weight *after* the softmax, and $e^{-\infty} = 0$. }})
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis={{ -1 | w.ndim - 1 ? Each **query** distributes its attention over the keys, so normalise over the key axis. That's the last one, whatever the number of batch axes. }}, keepdims=True)
    return w @ v, w
:::

:::solution{lang=python}
import numpy as np

def causal_mask(T):
    return np.tril(np.ones((T, T), dtype=bool))

def scaled_dot_product_attention(q, k, v, mask=None):
    d = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ v, w
:::

:::explain
[L4] **Lower triangle, diagonal included.** Row $i$ is True for columns $0..i$. Using `np.triu(..., k=1)` for the *forbidden* region is equivalent; just negate it.

[L8] **Scores.** `swapaxes(-1, -2)` turns `k` into `(…, d, Tk)`, so the matmul broadcasts over every leading axis. Dividing by $\sqrt d$ keeps the logits at unit variance (Exercise 13.2 proves why).

[L9-10] **Mask before the softmax.** `np.where(mask, scores, -inf)` broadcasts a `(T, T)` mask across any batch axes. A `(B, 1, 1, Tk)` padding mask works the same way.

[L11-13] **Stable softmax over keys.** This is the Chapter 04 trick: subtract the row max, then exponentiate and normalise along the last axis. If an entire row is masked, the max is $-\infty$ and you get NaNs. Real implementations guard against that case.

[L14] **Weighted average of values.** `(…, Tq, Tk) @ (…, Tk, dv)` gives `(…, Tq, dv)`. Returning the weights too is what lets you visualise attention maps.
:::

:::tests{lang=python}
import numpy as np

def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def test_causal_mask():
    m = causal_mask(4)
    assert m.dtype == bool and m.shape == (4, 4)
    assert m[2, 0] and m[2, 2] and not m[2, 3]

def test_matches_reference_4d():
    rng = np.random.default_rng(0)
    q, k, v = rng.normal(size=(2, 3, 5, 8)), rng.normal(size=(2, 3, 7, 8)), rng.normal(size=(2, 3, 7, 4))
    out, w = scaled_dot_product_attention(q, k, v)
    ref_w = _softmax(np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(8))
    assert out.shape == (2, 3, 5, 4) and w.shape == (2, 3, 5, 7)
    assert np.allclose(w, ref_w) and np.allclose(out, ref_w @ v)

def test_causal_blocks_the_future():
    x = np.random.default_rng(1).normal(size=(6, 4))
    out, w = scaled_dot_product_attention(x, x, x, causal_mask(6))
    assert np.allclose(np.triu(w, k=1), 0.0) and np.allclose(w.sum(-1), 1.0)
    x2 = x.copy()
    x2[-1] += 10.0
    out2, _ = scaled_dot_product_attention(x2, x2, x2, causal_mask(6))
    assert np.allclose(out[:-1], out2[:-1])

def test_large_scores_stay_finite():
    q, k = np.full((1, 2, 4), 100.0), np.full((1, 3, 4), 100.0)
    v = np.arange(12.0).reshape(1, 3, 4)
    out, w = scaled_dot_product_attention(q, k, v)
    assert np.all(np.isfinite(out)) and np.allclose(w, 1 / 3)
:::

:::hints
1. Write the shapes next to every line: `scores` must be `(…, Tq, Tk)`.
2. Masking is a `np.where` on the scores, not a multiplication on the weights.
3. If the 4-D test fails but 2-D works, look for a `.T` or a hard-coded axis number.
:::

:::interview
- Why mask with $-\infty$ *before* the softmax rather than zeroing weights after? *(Zeroing afterwards breaks the normalisation, so rows no longer sum to 1.)*
- What happens to a row where every key is masked? How do libraries handle it?
- What's the time and memory cost in $T$? *($O(T^2 d)$ time and $O(T^2)$ memory for the weights. FlashAttention avoids materialising them.)*
- How do you build a padding mask for a batch of sequences of different lengths?
:::
::::

## Why divide by √d?

This is a guaranteed interview question. The one-line answer is "to keep the softmax out of saturation". The derivation shows *why it's √d and not d*.

::::exercise{id=attention-sqrt-d type=derive difficulty=2 time=8 runtime=browser udl="§12.3" tags="self-attention,sqrt-d-derivation,initialisation"}
# Why attention is scaled by √d

:::prereq
- **Independence:** if $X \perp Y$, then $\mathbb E[XY] = \mathbb E[X]\,\mathbb E[Y]$.
- **Variance of a sum** of independent terms is the sum of their variances.
- **Scaling:** $\mathrm{Var}(cX) = c^2\,\mathrm{Var}(X)$.
- **Softmax saturation:** if one logit exceeds the rest by a lot, softmax is nearly one-hot. Its Jacobian $\mathrm{diag}(p) - pp^\top$ then goes to $0$, and so do the gradients.
:::

:::task
Let $q, k \in \mathbb R^d$ have independent components with mean $0$ and variance $1$, which is roughly the situation at initialisation. **Derive the mean and variance of $q\cdot k$**, explain what goes wrong in the softmax without scaling, and justify $\sqrt d$ over $d$.

Say it out loud in under five minutes.
:::

:::scaffold
Let $s = q\cdot k = \sum_{i=1}^d q_i k_i$.

**1. Mean.** $\mathbb E[s] = \sum_i \mathbb E[q_i]\,\mathbb E[k_i] =$ {{ 0 ? Each factor has mean zero, and independence lets the expectation split. }}

**2. One term.** $\mathrm{Var}(q_i k_i) = \mathbb E[q_i^2]\,\mathbb E[k_i^2] - (\mathbb E[q_i k_i])^2 =$ {{ 1 | 1 - 0 | 1·1 ? $\mathbb E[q_i^2] = \mathrm{Var}(q_i) + \mathbb E[q_i]^2 = 1$, and the subtracted square is zero by step 1. }}

**3. The sum.** $\mathrm{Var}(s) =$ {{ d | d·1 | \sum_i 1 ? There are $d$ independent terms, each with variance 1, and variances of independent terms add. }}, so the logits' standard deviation grows like $\sqrt d$.

**4. The fix.** Divide the scores by {{ \sqrt{d} | sqrt(d) | √d | d^{1/2} | d^0.5 | d**0.5 | \sqrt d ? $\mathrm{Var}(s/c) = \mathrm{Var}(s)/c^2$. You want this to equal 1, so $c^2 = d$. }} to restore unit variance.

**5. Without it**, a large $d$ makes the softmax nearly {{ one-hot | onehot | re:one-?hot|argmax|saturat|peak|spik|hard ? The logits' spread grows like $\sqrt d$, so one key dominates. The softmax Jacobian then vanishes, and so do the gradients. }}
:::

:::derivation
**Set up.** The score is a sum of $d$ products, $s = \sum_{i=1}^d q_i k_i$, whose components are independent with mean $0$ and variance $1$.

---step---

**Mean.** By independence, $\mathbb E[q_i k_i] = \mathbb E[q_i]\,\mathbb E[k_i] = 0$, so $\mathbb E[s] = 0$.

---step---

**Variance.** For one term,
$$\mathrm{Var}(q_i k_i) = \mathbb E[q_i^2 k_i^2] - \mathbb E[q_i k_i]^2 = \mathbb E[q_i^2]\,\mathbb E[k_i^2] - 0 = 1 .$$
The $d$ terms are independent, so the variances add: $\mathrm{Var}(s) = d$, and the standard deviation is $\sqrt d$.

---step---

**Choose the scale.** $\mathrm{Var}(s/\sqrt d) = d/d = 1$, so the logits keep unit scale whatever the head size.

Dividing by $d$ would *over*-correct: the variance becomes $1/d$, every logit collapses towards $0$, and attention becomes uniform averaging.

---step---

**Say why it matters.**

- With $d = 512$ and no scaling, the logits have standard deviation of about $22.6$. The largest one wins outright, and the softmax is nearly one-hot.
- Then $\mathrm{diag}(p) - pp^\top \approx 0$, so almost no gradient flows to the queries and keys. Training stalls or becomes unstable.
- This is an argument about **initialisation**. Later, learned weights can rescale things, which is why some models add QK-LayerNorm or a learnable temperature.
:::

:::solution{lang=python}
import numpy as np

def logit_std(d, n=20000, scale=True, seed=0):
    rng = np.random.default_rng(seed)
    q, k = rng.normal(size=(n, d)), rng.normal(size=(n, d))
    s = (q * k).sum(axis=1)
    return (s / np.sqrt(d) if scale else s).std()

def max_attention_weight(d, T=64, scale=True, seed=0):
    rng = np.random.default_rng(seed)
    q, k = rng.normal(size=(T, d)), rng.normal(size=(T, d))
    s = q @ k.T / (np.sqrt(d) if scale else 1.0)
    w = np.exp(s - s.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)
    return w.max(axis=1).mean()
:::

:::explain
[L3-7] **Measure the spread.** Draw $n$ random query–key pairs and take row-wise dot products. Unscaled, the standard deviation is about $\sqrt d$ (16 for $d=256$). Scaled, it's about $1$ for every $d$.

[L9-15] **Measure saturation.** For $T=64$ keys, compute each query's largest attention weight. Uniform attention would give $1/64 \approx 0.016$. Unscaled at $d=512$, the largest weight is about $0.9$: one key takes almost everything. Scaled, it stays near $0.1$.

Run the tests, then try other values of `d` in your head. The unscaled maximum weight climbs towards 1 as `d` grows, while the scaled one stays flat.
:::

:::tests{lang=python}
import numpy as np

def test_unscaled_std_grows_like_sqrt_d():
    assert abs(logit_std(256, scale=False) - 16.0) < 0.8

def test_scaled_std_is_one():
    for d in (16, 256):
        assert abs(logit_std(d) - 1.0) < 0.05

def test_unscaled_softmax_saturates():
    assert max_attention_weight(512, scale=False) > 0.8
    assert max_attention_weight(512) < 0.2
:::

:::hints
1. Treat $q\cdot k$ as a sum of $d$ independent random variables.
2. Variances add, and standard deviations add in quadrature. Which of those tells you the scale?
:::

:::interview
- Why $\sqrt d$ and not $d$? *(Dividing by $d$ makes the variance $1/d$, which pushes attention towards uniform.)*
- Does the argument still hold after training? What do QK-norm and temperature parameters address?
- Where else does the same "keep the variance at 1" argument appear? *(Xavier and He initialisation, in Ch. 06.)*
:::
::::

## Many heads

One head computes one weighted average per position. **Multi-head attention** runs $h$ smaller attentions in parallel, each of dimension $d_k = d_{\text{model}}/h$, so different heads can attend to different things. The trick is purely a matter of **reshaping**: the projections are a single matmul each, and the heads are just a reshape of the feature axis.

::::exercise{id=multi-head-attention type=implement difficulty=3 time=30 runtime=local-torch udl="§12.3" tags="multi-head-attention,pytorch,tensor-shapes"}
# Multi-head self-attention in PyTorch

:::prereq
**Splitting heads.** `(B, T, D)` → `view(B, T, h, dk)` → `transpose(1, 2)` → `(B, h, T, dk)`. You *must* split the **last** axis, then move the head axis forward. Reshaping straight to `(B, h, T, dk)` mixes up tokens (see Exercise 13.4).

**Merging heads.** Reverse it: `(B, h, T, dk)` → `transpose(1, 2)` → `(B, T, h, dk)` → `reshape(B, T, D)`. After a transpose the tensor isn't contiguous, so `.view` raises an error. Use `.reshape` or `.contiguous().view`.

**Fused QKV.** A single `nn.Linear(D, 3D)` computes Q, K and V in one matmul. `.chunk(3, dim=-1)` splits them.

**`masked_fill(cond, value)`** writes `value` wherever `cond` is **True**.
:::

:::task
Implement `MultiHeadAttention(d_model, n_heads)` as an `nn.Module` whose `forward(x, mask=None)` maps `(B, T, d_model)` to `(B, T, d_model)`. `mask` is a `(T, T)` bool tensor, with **True = may attend**.

**Required attribute names.** The tests copy your weights into `torch.nn.MultiheadAttention` and compare outputs, so use exactly these:

| Attribute | Value |
|---|---|
| `self.h` | number of heads |
| `self.qkv` | `nn.Linear(d_model, 3 * d_model)`, output order **q, k, v** |
| `self.out` | `nn.Linear(d_model, d_model)` |

Don't use `F.scaled_dot_product_attention`. Write the attention yourself.
:::

:::starter{lang=python}
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """Self-attention. x: (B, T, d_model) -> (B, T, d_model); mask: (T, T) bool, True = may attend."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, mask=None):
        raise NotImplementedError
:::

:::scaffold{lang=python}
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, {{ self.h, self.dk | self.h, -1 | self.h, D // self.h ? Split the **last** axis $D$ into $(h, d_k)$. Tokens stay on axis 1 for now. }}).transpose({{ 1, 2 | 2, 1 ? Move heads in front of time, `(B, T, h, dk) → (B, h, T, dk)`, so the matmuls batch over `(B, h)`. }}) for t in (q, k, v))
        scores = q @ k.transpose({{ -2, -1 | -1, -2 | 2, 3 | 3, 2 ? You need `(B, h, T, dk) @ (B, h, dk, T)`, so swap only the last two axes of `k`. }}) / self.dk ** 0.5
        if mask is not None:
            scores = scores.masked_fill({{ ~mask | mask == 0 | mask.logical_not() | mask == False | ~mask.bool() ? `masked_fill` writes where the condition is **True**, but our mask is True where attention is *allowed*. Fill the complement. }}, float("-inf"))
        w = scores.softmax(dim={{ -1 | 3 ? Each query's weights over the **keys** must sum to 1. Keys are the last axis. }})
        y = (w @ v).transpose(1, 2).{{ reshape(B, T, D) | contiguous().view(B, T, D) | reshape(B, T, -1) | flatten(2) | flatten(-2) | reshape(B, T, self.h * self.dk) ? Merge `(B, T, h, dk)` back into `(B, T, D)`. After `transpose` the memory isn't contiguous, so plain `.view` fails. }}
        return self.out(y)
:::

:::solution{lang=python}
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / self.dk ** 0.5
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        w = scores.softmax(dim=-1)
        y = (w @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(y)
:::

:::explain
[L7-10] **Parameters.** There are $4D^2 + 4D$ of them, whatever $h$ is: more heads means *narrower* heads, not more weights. The fused `qkv` layer lays out its output as `[q | k | v]`, the same as `nn.MultiheadAttention.in_proj_weight`. That's why the tests can copy weights straight across.

[L14] **One matmul for all three.** `chunk(3, dim=-1)` returns views of the `(B, T, 3D)` output, so nothing is copied.

[L15] **Split heads correctly.** `view(B, T, h, dk)` cuts each token's feature vector into $h$ slices. `transpose(1, 2)` then moves heads forward. Each head sees *every token*, but only its own slice of features.

[L16] **Scale by $\sqrt{d_k}$, not $\sqrt{D}$.** Each head's dot product sums over $d_k$ features, so that's the variance to correct for (Exercise 13.2).

[L17-19] **Mask, then softmax over keys.** `~mask` flips "may attend" into "must block". Note that `nn.MultiheadAttention`'s boolean `attn_mask` uses the *opposite* convention, True = block, while `F.scaled_dot_product_attention` uses ours. This mismatch causes real bugs.

[L20-21] **Merge and project.** `transpose(1, 2)` makes the memory non-contiguous, which is why this line uses `reshape`. The output projection mixes information across heads.
:::

:::tests{lang=python}
import torch

def _ref(m, x, mask=None):
    ref = torch.nn.MultiheadAttention(x.shape[-1], m.h, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(m.qkv.weight)
        ref.in_proj_bias.copy_(m.qkv.bias)
        ref.out_proj.weight.copy_(m.out.weight)
        ref.out_proj.bias.copy_(m.out.bias)
    return ref(x, x, x, attn_mask=None if mask is None else ~mask, need_weights=False)[0]

def test_shapes():
    m = MultiHeadAttention(32, 4)
    assert m(torch.randn(2, 5, 32)).shape == (2, 5, 32)

def test_matches_torch_mha():
    torch.manual_seed(0)
    m, x = MultiHeadAttention(32, 4), torch.randn(3, 7, 32)
    assert torch.allclose(m(x), _ref(m, x), atol=1e-5)

def test_causal_mask_matches_and_blocks_future():
    torch.manual_seed(1)
    m, x = MultiHeadAttention(16, 2), torch.randn(2, 6, 16)
    mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    y = m(x, mask)
    assert torch.allclose(y, _ref(m, x, mask), atol=1e-5)
    x2 = x.clone()
    x2[:, -1] += 5.0
    assert torch.allclose(m(x2, mask)[:, :-1], y[:, :-1], atol=1e-6)

def test_gradients_reach_every_parameter():
    m, x = MultiHeadAttention(8, 2), torch.randn(1, 4, 8, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None and all(p.grad is not None for p in m.parameters())
:::

:::hints
1. Write the shape after every line, e.g. `# (B, h, T, dk)`. Most bugs here are shape bugs that don't crash.
2. If `test_matches_torch_mha` fails but the shapes are right, check the order of `view` and `transpose` when splitting heads.
3. A `RuntimeError: view size is not compatible…` means you called `.view` on a transposed tensor.
:::

:::interview
- How many parameters does MHA have, and how does that depend on $h$? *($4D^2 + 4D$, independent of $h$.)*
- Why use several heads instead of one big one?
- What's a KV cache, and which tensors in this code would it store?
- What changes for cross-attention? *(Queries come from one sequence, keys and values from another, so `Tq ≠ Tk`.)*
:::
::::

::::exercise{id=split-heads-bug type=debug difficulty=2 time=10 runtime=local-torch udl="§12.3" tags="debugging,tensor-shapes,multi-head-attention"}
# Heads that leak across tokens

:::prereq
**`view` never moves data.** It reinterprets the same row-major memory with a new shape. `view(B, h, T, dk)` on a `(B, T, D)` tensor chops the *flattened* `T·D` numbers into `h` blocks of `T·dk`, so each "head" gets whole tokens' worth of features from several positions.

**`transpose` changes the view, not the memory.** To put heads before time, you must first split the feature axis *where it lives* (last), then transpose.
:::

:::task
A colleague's transformer has all the right shapes and trains, a bit. But the causal language model is suspiciously good on the training set and useless at generation. Probing shows that **changing token 5 changes the output at position 0**.

They wrote these helpers. Find the bug, explain how it lets information leak *across tokens* despite the causal mask, and fix it.
:::

:::starter{lang=python}
import torch

def split_heads(x, h):
    """x: (B, T, D) -> (B, h, T, D // h)"""
    B, T, D = x.shape
    return x.view(B, h, T, D // h)

def merge_heads(x):
    """x: (B, h, T, dk) -> (B, T, h * dk)"""
    B, h, T, dk = x.shape
    return x.transpose(1, 2).reshape(B, T, h * dk)
:::

:::scaffold{lang=python}
import torch

def split_heads(x, h):
    """x: (B, T, D) -> (B, h, T, D // h)"""
    B, T, D = x.shape
    return x.{{ view(B, T, h, D // h).transpose(1, 2) | reshape(B, T, h, D // h).transpose(1, 2) | view(B, T, h, -1).transpose(1, 2) | reshape(B, T, h, -1).transpose(1, 2) | unflatten(-1, (h, D // h)).transpose(1, 2) | view(B, T, h, D // h).permute(0, 2, 1, 3) | reshape(B, T, h, D // h).permute(0, 2, 1, 3) ? `view(B, h, T, dk)` slices the *flattened* memory, so "position $t$ of head 0" contains features from several different tokens. Split the feature axis first, `(B, T, h, dk)`, then move heads forward. }}

def merge_heads(x):
    """x: (B, h, T, dk) -> (B, T, h * dk)"""
    B, h, T, dk = x.shape
    return x.transpose(1, 2).reshape(B, T, h * dk)
:::

:::solution{lang=python}
import torch

def split_heads(x, h):
    """x: (B, T, D) -> (B, h, T, D // h)"""
    B, T, D = x.shape
    return x.view(B, T, h, D // h).transpose(1, 2)

def merge_heads(x):
    """x: (B, h, T, dk) -> (B, T, h * dk)"""
    B, h, T, dk = x.shape
    return x.transpose(1, 2).reshape(B, T, h * dk)
:::

:::explain
[L6] **Split where the features live.** `view(B, T, h, dk)` cuts each token's own $D$ features into $h$ slices, and `transpose(1, 2)` then reorders the axes without mixing anything.

**Why the causal mask didn't help.** The mask stops position $i$ attending to *positions* $j > i$. But with the buggy `view`, "position 0" of a head is assembled from features of tokens 0, 1, 2, …. The future has leaked in before attention even runs, so the mask has nothing to catch. Training loss looks great because the model is partly reading the answer.

[L8-11] **`merge_heads` was right all along.** That's what made the bug sneaky: it is the correct inverse of the *correct* split, so the round trip doesn't restore $x$ when paired with the buggy split.

**General lesson.** Test with **structured inputs**. Make one token non-zero and check exactly where its values end up. Random inputs with plausible statistics hide layout bugs.
:::

:::tests{lang=python}
import torch

def test_round_trip_is_identity():
    x = torch.randn(2, 5, 12)
    assert torch.equal(merge_heads(split_heads(x, 3)), x)

def test_each_position_holds_only_its_own_token():
    x = torch.zeros(1, 4, 8)
    x[0, 2] = 1.0
    s = split_heads(x, 2)
    assert s.shape == (1, 2, 4, 4)
    assert torch.all(s[:, :, 2] == 1.0) and s.sum() == 8.0
:::

:::hints
1. Use `torch.arange(2 * 3 * 4).view(1, 3, 4)` as `x` with `h = 2`, and print `split_heads(x, 2)`. Which numbers end up in head 0, position 0?
2. `view` reads memory in row-major order. Which axis of `(B, T, D)` changes fastest?
:::

:::interview
- Why doesn't the causal mask protect against this bug?
- When does `.view` fail, and when does it silently do the wrong thing?
- How would `einops.rearrange(x, "b t (h d) -> b h t d", h=h)` have prevented this?
:::
::::
