---
id: ch14-transformers
summary: Assemble attention into a full transformer. Encode positions two ways (sinusoidal and RoPE), build a pre-LN block, make generation fast with a KV cache, and get straight which masks make a model encoder-only, decoder-only or encoder–decoder.
udl: "Ch. 12 — Transformers"
---

## From attention to a transformer

Chapter 13 built attention. A **transformer block** wraps it with three more ingredients:

```
x ─┬─► LayerNorm ─► MultiHeadAttention ─►(+)─┬─► LayerNorm ─► MLP ─►(+)─► out
   └──────────────────────────────────────────┘└──────────────────────────┘
                  residual                                 residual
```

- **Position information.** Attention is permutation-invariant, so order must be injected (exercise 14.1).
- **Residual connections plus normalisation** keep a 100-layer stack trainable (Chapter 10). Pre-LN, shown above, normalises *inside* each branch. Post-LN, as in the original 2017 paper, normalises after the addition, and needs warmup to train.
- **A position-wise MLP**, usually $D \to 4D \to D$ with GELU. It holds about two-thirds of the parameters.

**Variants** differ only in *masks* and *what attends to what*. Encoder-only (BERT) uses bidirectional self-attention. Decoder-only (GPT) uses causal self-attention. Encoder–decoder (T5, the original transformer) adds cross-attention from the decoder to the encoder (exercise 14.4).

::::exercise{id=positional-encodings type=implement difficulty=2 time=20 runtime=browser udl="§12.4" tags="transformers"}
# Sinusoidal encodings and RoPE

:::prereq
**Sinusoidal (absolute) encodings** are *added* to the token embeddings. For position $p$ and pair index $i$:
$$\mathrm{PE}[p, 2i] = \sin\!\big(p/10000^{2i/d}\big),\qquad \mathrm{PE}[p, 2i+1] = \cos\!\big(p/10000^{2i/d}\big).$$
Each pair is a clock running at its own frequency, from fast (small $i$) to very slow (large $i$).

**RoPE (rotary embeddings)** instead *rotates* the queries and keys. Each feature pair $(x_{2i}, x_{2i+1})$ at position $p$ is rotated by the angle $p\,\theta_i$, where $\theta_i = 10000^{-2i/d}$. Rotations compose, so
$$\langle R_m q,\, R_n k\rangle = \langle q,\, R_{n-m} k\rangle :$$
attention scores depend only on the **relative** offset $n - m$.
:::

:::task
Implement:

- `sinusoidal_pe(T, d)`, returning `(T, d)` for even `d`;
- `rope(x, positions, base=10000.0)`, which rotates the interleaved pairs of `x` (shape `(T, d)`) by `positions[t] * theta_i`.
:::

:::starter{lang=python}
import numpy as np

def sinusoidal_pe(T, d):
    raise NotImplementedError

def rope(x, positions, base=10000.0):
    """x: (T, d) with d even; positions: (T,) -> (T, d), pairs (2i, 2i+1) rotated by positions * theta_i."""
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def sinusoidal_pe(T, d):
    pos = np.arange(T)[:, None]
    i = np.arange(d // 2)[None, :]
    angles = pos / {{ 10000 ** (2 * i / d) | 10000 ** (2*i/d) | np.power(10000, 2 * i / d) | 10000.0 ** (2 * i / d) ? Each pair $i$ runs at frequency $1/10000^{2i/d}$, from one radian per position down to very slow. }}
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = {{ np.cos(angles) ? The odd features carry the cosine of the same angle, so each pair is a point on a circle. }}
    return pe

def rope(x, positions, base=10000.0):
    d = x.shape[-1]
    theta = base ** (-np.arange(0, d, 2) / d)
    ang = positions[:, None] * theta[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = {{ x1 * cos - x2 * sin | cos * x1 - sin * x2 ? A 2-D rotation by angle $a$ maps $(x_1, x_2)$ to $(x_1\cos a - x_2\sin a,\; x_1\sin a + x_2\cos a)$. }}
    out[:, 1::2] = {{ x1 * sin + x2 * cos | sin * x1 + cos * x2 | x2 * cos + x1 * sin ? The second component of the same rotation. }}
    return out
:::

:::solution{lang=python}
import numpy as np

def sinusoidal_pe(T, d):
    pos = np.arange(T)[:, None]
    i = np.arange(d // 2)[None, :]
    angles = pos / 10000 ** (2 * i / d)
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe

def rope(x, positions, base=10000.0):
    d = x.shape[-1]
    theta = base ** (-np.arange(0, d, 2) / d)
    ang = positions[:, None] * theta[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out
:::

:::explain
[L4-6] **An outer product of positions and frequencies.** `(T, 1) / (1, d/2)` broadcasts to `(T, d/2)`: one angle per position per pair.

[L7-9] **Interleave sin and cos.** Because each pair is $(\sin, \cos)$ of the same angle, $\mathrm{PE}[p + k]$ is a fixed *rotation* of $\mathrm{PE}[p]$. That was the original paper's argument that relative positions are easy to learn.

[L13-15] **The RoPE angles.** These are the same frequencies, applied to queries and keys rather than added to embeddings.

[L16-19] **Rotate each pair.** The norm is preserved, so RoPE never changes a vector's length, only its phase. (LLaMA-style code pairs feature $j$ with $j + d/2$ instead of interleaving. It's the same idea with a different layout, and mixing the two up silently breaks pretrained weights.)

**Why RoPE won.** Scores depend only on relative offsets, it adds no parameters, and with frequency rescaling ("NTK" or "YaRN" scaling) it extends to longer contexts than seen in training.
:::

:::tests{lang=python}
import numpy as np

def test_pe_shape_and_first_rows():
    pe = sinusoidal_pe(10, 8)
    assert pe.shape == (10, 8) and np.allclose(pe[0], [0, 1, 0, 1, 0, 1, 0, 1])
    assert np.isclose(pe[1, 0], np.sin(1.0)) and np.isclose(pe[1, 1], np.cos(1.0))
    assert np.isclose(pe[3, 2], np.sin(3 / 10000 ** (2 / 8)))

def test_rope_preserves_norms_and_position_zero():
    x = np.random.default_rng(0).normal(size=(6, 8))
    r = rope(x, np.arange(6.0))
    assert np.allclose(np.linalg.norm(r, axis=1), np.linalg.norm(x, axis=1))
    assert np.allclose(r[0], x[0])

def test_rope_scores_depend_only_on_offset():
    rng = np.random.default_rng(1)
    q, k = rng.normal(size=(1, 16)), rng.normal(size=(1, 16))
    s = lambda m, n: (rope(q, np.array([m])) @ rope(k, np.array([n])).T).item()
    assert np.isclose(s(3, 7), s(10, 14)) and np.isclose(s(0, 4), s(100, 104))
    assert not np.isclose(s(3, 7), s(3, 8))
:::

:::hints
1. Build the angle table by broadcasting a column of positions against a row of frequencies.
2. Write the 2×2 rotation matrix and apply it to each pair `(x1, x2)`.
:::

:::interview
- Why do transformers need positional information at all? *(Self-attention is permutation-equivariant.)*
- Absolute versus relative positions: what does RoPE make relative, and how?
- How do models extend the context length past training (RoPE scaling, ALiBi)?
- Learned versus sinusoidal absolute embeddings: which extrapolates, and does either do it well?
:::
::::

::::exercise{id=transformer-block type=implement difficulty=2 time=20 runtime=local-torch udl="§12.4" tags="transformers,pytorch,residuals,normalisation"}
# A pre-LN transformer block

:::prereq
**Pre-LN block.** $x \leftarrow x + \mathrm{Attn}(\mathrm{LN}_1(x))$, then $x \leftarrow x + \mathrm{MLP}(\mathrm{LN}_2(x))$.

**`nn.MultiheadAttention(d, h, batch_first=True)`** takes `(query, key, value)` and returns `(output, weights)`. Its boolean `attn_mask` uses **True = not allowed**, the *opposite* of the mask convention in Chapter 13. For a causal mask, block the strict upper triangle.

**The MLP** is `Linear(d, d_ff) → GELU → Linear(d_ff, d)`.
:::

:::task
Implement `Block(d, n_heads, d_ff)` with the attributes `ln1`, `ln2`, `attn` (an `nn.MultiheadAttention` with `batch_first=True`) and `mlp` (an `nn.Sequential` of the three layers above). `forward(x, causal=False)` maps `(B, T, d)` to `(B, T, d)`.
:::

:::starter{lang=python}
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, causal=False):
        raise NotImplementedError
:::

:::scaffold{lang=python}
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, x, causal=False):
        T = x.shape[1]
        mask = {{ torch.triu(torch.ones(T, T, dtype=torch.bool), 1) | torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1) | torch.ones(T, T, dtype=torch.bool).triu(1) | ~torch.tril(torch.ones(T, T, dtype=torch.bool)) ? For `nn.MultiheadAttention`, True means **blocked**. Block the future: the strict upper triangle, above the diagonal. }} if causal else None
        h = {{ self.ln1(x) ? Pre-LN: normalise the input of the branch, not the residual stream itself. }}
        x = {{ x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0] | x + self.attn(h, h, h, attn_mask=mask)[0] ? Self-attention uses the normalised `h` as query, key and value. Add the result back onto the **un-normalised** `x`. }}
        return {{ x + self.mlp(self.ln2(x)) | self.mlp(self.ln2(x)) + x ? The second residual branch: normalise, apply the MLP, and add. }}
:::

:::solution{lang=python}
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, x, causal=False):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1) if causal else None
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))
:::

:::explain
[L7-9] **Parameters.** Attention has $4d^2 + 4d$, the MLP $2d\,d_\text{ff} + d_\text{ff} + d$, and the two LayerNorms $4d$. With $d_\text{ff} = 4d$, the MLP holds about two-thirds of the block.

[L13] **The causal mask, in this API's convention.** `triu(…, 1)` marks every key *after* the query as blocked. Getting this backwards gives a model that sees only the future, and it still trains, just badly.

[L14-15] **Pre-LN.** The residual stream `x` is never normalised directly. Each branch reads a normalised copy and *adds* its contribution, so the identity path from input to output stays clean. That's why pre-LN trains stably without warmup at depth, and why a final LayerNorm is needed after the last block.

[L16] **The second branch.** It's the same pattern. The tests check that zeroing both branches' output layers turns the whole block into the identity.
:::

:::tests{lang=python}
import torch

def test_shape_and_param_count():
    b = Block(32, 4, 64)
    assert b(torch.randn(2, 5, 32)).shape == (2, 5, 32)
    assert sum(p.numel() for p in b.parameters()) == 8544

def test_causal_block_ignores_the_future():
    torch.manual_seed(0)
    b, x = Block(16, 2, 32).eval(), torch.randn(1, 6, 16)
    x2 = x.clone()
    x2[:, -1] += 3.0 * torch.randn(16)  # not a constant shift: LayerNorm would erase that
    assert torch.allclose(b(x, causal=True)[:, :-1], b(x2, causal=True)[:, :-1], atol=1e-6)
    assert not torch.allclose(b(x)[:, 0], b(x2)[:, 0], atol=1e-4)

def test_zeroed_branches_give_identity():
    b, x = Block(16, 2, 32), torch.randn(2, 4, 16)
    with torch.no_grad():
        for layer in (b.attn.out_proj, b.mlp[2]):
            layer.weight.zero_()
            layer.bias.zero_()
    assert torch.allclose(b(x), x)
:::

:::hints
1. Two residual branches, each starting with its own LayerNorm.
2. If the causal test fails, check the mask convention: in this API, True means *blocked*.
:::

:::interview
- Pre-LN versus post-LN: where does the norm sit, and why does pre-LN train more stably?
- Where are most of a transformer's parameters and FLOPs? *(Parameters: in the MLPs for typical sizes. FLOPs: attention's $O(T^2 d)$ term dominates once $T$ is larger than about $d$.)*
- Why GELU or SwiGLU rather than ReLU in modern transformers?
:::
::::

::::exercise{id=kv-cache type=implement difficulty=2 time=15 runtime=browser udl="§12.5" tags="transformers,self-attention"}
# A KV cache for fast generation

:::prereq
**Autoregressive generation** produces one token at a time. Naively, step $t$ recomputes attention for the whole prefix, so a sequence costs $O(T^2)$ projections overall.

**The cache.** Under a causal mask, the keys and values of earlier tokens **never change**: token $j$ only sees positions $\le j$. So store each new $k_t$ and $v_t$ as they're produced. At step $t$, only the new query needs computing:
$$o_t = \mathrm{softmax}\!\left(\frac{q_t K_{\le t}^\top}{\sqrt d}\right)V_{\le t}.$$
No mask is needed, because the cache only ever contains the past.
:::

:::task
`causal_attention(Q, K, V)` (the full, masked version) is given as the reference. Implement `KVCache` with `step(q, k, v)`. Each argument has shape `(d,)` for the newest token. The method appends `k` and `v` to the cache and returns that token's attention output, shape `(d,)`. Feeding tokens one at a time must reproduce `causal_attention` row by row.
:::

:::starter{lang=python}
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def causal_attention(Q, K, V):
    T, d = Q.shape
    s = Q @ K.T / np.sqrt(d)
    s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, -np.inf)
    return softmax(s) @ V

class KVCache:
    def __init__(self):
        self.K, self.V = [], []

    def step(self, q, k, v):
        raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def causal_attention(Q, K, V):
    T, d = Q.shape
    s = Q @ K.T / np.sqrt(d)
    s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, -np.inf)
    return softmax(s) @ V

class KVCache:
    def __init__(self):
        self.K, self.V = [], []

    def step(self, q, k, v):
        {{ self.K.append(k) ? The new token's key joins the cache **before** attending: a token attends to itself. }}
        self.V.append(v)
        K, V = np.stack(self.K), np.stack(self.V)
        w = softmax({{ q @ K.T / np.sqrt(q.shape[-1]) | K @ q / np.sqrt(len(q)) | q @ K.T / np.sqrt(len(q)) | K @ q / np.sqrt(q.shape[-1]) ? One query against every cached key: `(d,) @ (d, t)` gives `t` scores. Scale by $\sqrt d$ as always. }})
        return {{ w @ V | V.T @ w | np.dot(w, V) ? Weighted average of the cached values: `(t,) @ (t, d)` gives `(d,)`. }}
:::

:::solution{lang=python}
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def causal_attention(Q, K, V):
    T, d = Q.shape
    s = Q @ K.T / np.sqrt(d)
    s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, -np.inf)
    return softmax(s) @ V

class KVCache:
    def __init__(self):
        self.K, self.V = [], []

    def step(self, q, k, v):
        self.K.append(k)
        self.V.append(v)
        K, V = np.stack(self.K), np.stack(self.V)
        w = softmax(q @ K.T / np.sqrt(q.shape[-1]))
        return w @ V
:::

:::explain
[L18-19] **Append first.** Token $t$'s own key and value are part of what it attends to, so they're the diagonal of the causal mask.

[L20] **The cache is the past.** Stacking gives `(t, d)`. Real implementations preallocate a `(T_max, d)` buffer per layer and per head, and write into it, rather than re-stacking every step.

[L21-22] **Just one row of attention.** This is $O(t\,d)$ work per step instead of recomputing all $t$ rows. There's no mask: there's nothing in the cache to hide.

**Memory is the catch.** The cache stores $2 \times \text{layers} \times \text{heads} \times d_\text{head} \times T$ numbers per sequence. For long contexts that dominates GPU memory, which motivates **multi-query** and **grouped-query attention** (sharing K and V across heads) and paged KV caches.
:::

:::tests{lang=python}
import numpy as np

def test_incremental_matches_full_causal():
    rng = np.random.default_rng(0)
    Q, K, V = rng.normal(size=(7, 8)), rng.normal(size=(7, 8)), rng.normal(size=(7, 8))
    full, cache = causal_attention(Q, K, V), KVCache()
    for t in range(7):
        assert np.allclose(cache.step(Q[t], K[t], V[t]), full[t])
    assert len(cache.K) == 7

def test_first_token_attends_only_to_itself():
    rng = np.random.default_rng(1)
    q, k, v = rng.normal(size=4), rng.normal(size=4), rng.normal(size=4)
    assert np.allclose(KVCache().step(q, k, v), v)
:::

:::hints
1. The order inside `step` is: cache, then attend.
2. The shapes are `q: (d,)` and `K: (t, d)`, so the scores are `(t,)`.
:::

:::interview
- Why can keys and values be cached but not queries?
- What's the memory cost of a KV cache? How do multi-query and grouped-query attention reduce it?
- Why does a KV cache only work for causal (decoder) attention?
- Prefill versus decode: why is decode memory-bandwidth-bound?
:::
::::

::::exercise{id=transformer-variants-predict type=predict difficulty=1 time=6 runtime=browser udl="§12.6–12.7" tags="transformer-variants,transformers"}
# Encoder, decoder, or both?

:::task
An encoder–decoder transformer (like T5, or the original translation model) reads a **5-token** source sentence and has generated a **3-token** target prefix so far. In one decoder layer, what's the shape of one head's **cross-attention** weight matrix?
:::

:::choices
- [x] `(3, 5)`: one row per target position, one column per source token
- [ ] `(5, 3)`
- [ ] `(8, 8)`: source and target are concatenated
- [ ] `(3, 3)`
:::

:::experiment{lang=python}
import numpy as np

def show(name, M):
    print(f"{name}  shape {M.shape}")
    for row in M:
        print("    " + " ".join("■" if v else "·" for v in row))
    print()

S, T = 5, 3
encoder_self = np.ones((S, S), dtype=bool)                  # BERT / encoder: everyone sees everyone
decoder_self = np.tril(np.ones((T, T), dtype=bool))         # target side: causal
cross = np.ones((T, S), dtype=bool)                         # each target position reads the whole source
decoder_only = np.tril(np.ones((S + T, S + T), dtype=bool)) # GPT: one causal stream
prefix_lm = decoder_only.copy()
prefix_lm[:S, :S] = True                                    # prefix LM: bidirectional over the prompt

for name, M in [("encoder self-attention", encoder_self), ("decoder self-attention", decoder_self),
                ("decoder cross-attention", cross), ("decoder-only (GPT)", decoder_only), ("prefix LM", prefix_lm)]:
    show(name, M)
:::

:::tests{lang=python}
def test_cross_attention_shape():
    assert cross.shape == (3, 5)

def test_decoder_only_is_causal_over_everything():
    assert decoder_only.sum() == 8 * 9 // 2

def test_prefix_lm_adds_bidirectional_prompt():
    assert prefix_lm.sum() == decoder_only.sum() + 5 * 4 // 2
:::

:::explain
In cross-attention the **queries come from the decoder** (3 target positions) and the **keys and values come from the encoder output** (5 source tokens). The weight matrix is always (queries × keys), so it's `(3, 5)`, and no mask is needed, because every target position may read the whole source.

| Variant | Self-attention mask | Cross-attention? | Typical use |
|---|---|---|---|
| Encoder-only (BERT) | bidirectional | no | classification, embeddings, masked-LM pretraining |
| Decoder-only (GPT) | causal over the whole sequence | no | generation, prompting, almost every modern LLM |
| Encoder–decoder (T5) | encoder: bidirectional; decoder: causal | yes, decoder → encoder | translation, summarisation |
| Prefix LM | bidirectional over the prompt, causal after it | no | a hybrid of the two |

**Why decoder-only won for LLMs.** It's one stack, one objective (next-token prediction on everything) and one KV cache, and prompting turns every task into continuation. Encoders are still the better tool when you want a fixed embedding of a whole input.
:::

:::interview
- What exactly differs between BERT, GPT and T5? Masks, objectives, and when you'd use each.
- In cross-attention, where do Q, K and V come from?
- Can a decoder-only model do translation? *(Yes: put the source in the prompt. That's a decoder-only model with a "prefix".)*
:::
::::
