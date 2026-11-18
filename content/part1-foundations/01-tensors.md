---
id: ch01-tensors
summary: Most deep-learning bugs are shape bugs that don't crash. Predict what broadcasting does, compute pairwise distances without a single loop, rewrite matrix code with `einsum`, and find an aliasing bug that quietly corrupts your data.
udl: "Appendix (notation)"
---

## Shapes are the type system

In an interview, the quickest way to show fluency is to **say the shape of every tensor out loud** as you write it. NumPy and PyTorch share the same rules, so everything here carries straight over to `torch`.

| Operation | Rule |
|---|---|
| Elementwise `a + b` | broadcast (see below), then combine elementwise |
| `a @ b` | last axis of `a` meets the second-to-last of `b`; all other axes are batch axes and broadcast |
| Reductions `x.sum(axis=k)` | remove axis `k`, unless `keepdims=True` |
| `reshape` / `view` | reinterpret the same memory in row-major order; nothing moves |
| `transpose` / `swapaxes` | reorder axes; the memory layout is unchanged, so the result may not be contiguous |

**Broadcasting** lines shapes up **from the right**. Two axes are compatible when they're equal or one is $1$, and missing leading axes count as $1$. The result takes the larger size along each axis.

::::exercise{id=broadcast-predict type=predict difficulty=1 time=5 runtime=browser udl="—" tags="broadcasting,tensor-shapes"}
# What shape does broadcasting give?

:::task
`a` has shape `(4, 1, 3)` and `b` has shape `(2, 3)`. What is the shape of `a - b`?
:::

:::choices
- [ ] `(4, 3)`: the size-1 axis is squeezed away
- [x] `(4, 2, 3)`
- [ ] `(4, 1, 3)`: the result keeps the shape of the larger operand
- [ ] It raises an error: the number of dimensions differs
:::

:::experiment{lang=python}
import numpy as np

def show(sa, sb):
    a, b = np.zeros(sa), np.zeros(sb)
    try:
        print(f"{str(sa):12} - {str(sb):10} -> {(a - b).shape}")
    except ValueError as e:
        print(f"{str(sa):12} - {str(sb):10} -> ValueError")

show((4, 1, 3), (2, 3))     # the question
show((4, 3), (3,))          # subtract a per-column vector
show((4, 3), (4,))          # the classic mistake: per-row vector without keepdims
show((4, 3), (4, 1))        # ...done right
show((5, 1), (1, 6))        # an "outer" difference
show((2, 1, 5), (3, 1))     # try predicting this one first!
:::

:::tests{lang=python}
import numpy as np

def test_question_shape():
    assert (np.zeros((4, 1, 3)) - np.zeros((2, 3))).shape == (4, 2, 3)

def test_outer_difference():
    assert (np.zeros((5, 1)) - np.zeros((1, 6))).shape == (5, 6)
:::

:::explain
Align from the right and compare the axes pairwise:

```
a:  4  1  3
b:     2  3      ← a missing leading axis counts as 1
→   4  2  3
```

The last axes are $3$ and $3$, which are equal. The middle axes are $1$ and $2$, so the $1$ stretches to $2$. The first axes are $4$ and a missing axis, so the result is $4$. Nothing is squeezed and nothing is copied: broadcasting is a *virtual* repeat.

The `(4, 3) - (4,)` case in the experiment is the one that bites in practice. Aligned from the right, $3$ meets $4$ and it errors. Worse, if the two sizes happen to be **equal** (a square batch), it silently subtracts along the wrong axis. Chapter 04's debug exercise is exactly this bug.

**Interview habit:** when you write a broadcast, say it out loud as "(B, 1, D) against (1, T, D) gives (B, T, D)".
:::

:::interview
- What shape is `x[:, None, :] - x[None, :, :]` for `x` of shape `(N, D)`? What does it compute?
- Does broadcasting allocate memory for the stretched operand? *(No, it uses stride 0. The **result** is fully allocated, though.)*
:::
::::

## Pairwise distances: the canonical broadcasting problem

Nearest neighbours, k-means, RBF kernels and contrastive losses all start from a matrix of pairwise distances. There are two loop-free ways to compute one, and interviewers like to ask about the trade-off between them.

::::exercise{id=pairwise-distances type=implement difficulty=2 time=15 runtime=browser udl="—" tags="broadcasting,tensor-shapes,numerical-stability"}
# Pairwise squared distances without loops

:::prereq
For rows $x_i$ of $X \in \mathbb R^{N\times D}$ and $y_j$ of $Y\in\mathbb R^{M\times D}$:

$$\|x_i - y_j\|^2 = \|x_i\|^2 + \|y_j\|^2 - 2\,x_i^\top y_j.$$

- **Broadcast version:** form the `(N, M, D)` tensor of differences, square it, and sum over $D$. It's simple, but uses $O(NMD)$ memory.
- **Expansion version:** use the identity above. The cross term is one matmul `X @ Y.T`, giving `(N, M)`, so memory is $O(NM)$. Floating-point cancellation can make tiny distances come out slightly **negative**, so clamp at $0$.
:::

:::task
Implement both:

- `sq_dists_broadcast(X, Y)` using a `(N, M, D)` intermediate;
- `sq_dists_expand(X, Y)` using the expansion identity, never forming `(N, M, D)`, and clamping negatives to $0$.

Both take `X: (N, D)` and `Y: (M, D)`, and return `(N, M)`.
:::

:::starter{lang=python}
import numpy as np

def sq_dists_broadcast(X, Y):
    """X: (N, D), Y: (M, D) -> (N, M), via an (N, M, D) difference tensor."""
    raise NotImplementedError

def sq_dists_expand(X, Y):
    """X: (N, D), Y: (M, D) -> (N, M), via |x|^2 + |y|^2 - 2 x.y (no (N, M, D) tensor)."""
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def sq_dists_broadcast(X, Y):
    diff = {{ X[:, None, :] - Y[None, :, :] | X[:, None] - Y[None] | X[:, None, :] - Y[None] | X[:, np.newaxis, :] - Y[np.newaxis, :, :] | X[:, None] - Y ? Give `X` a singleton axis in position 1 and `Y` one in position 0: `(N, 1, D) − (1, M, D)` broadcasts to `(N, M, D)`. }}
    return (diff ** 2).sum(axis={{ -1 | 2 ? Sum the squared differences over the *feature* axis $D$, which is the last one. }})

def sq_dists_expand(X, Y):
    xx = (X ** 2).sum(axis=1)[:, None]
    yy = {{ (Y ** 2).sum(axis=1)[None, :] | (Y ** 2).sum(axis=1)[None] | (Y ** 2).sum(axis=1) | (Y**2).sum(1) | np.sum(Y ** 2, axis=1) | (Y * Y).sum(axis=1) ? Squared norms of the rows of `Y` must lie along the **column** axis: shape `(M,)` or `(1, M)` broadcasts against `(N, 1)`. }}
    d = xx + yy - 2 * {{ X @ Y.T | X @ Y.transpose() | np.dot(X, Y.T) | X.dot(Y.T) | np.matmul(X, Y.T) ? All cross terms $x_i^\top y_j$ at once: `(N, D) @ (D, M)` gives `(N, M)`. }}
    return {{ np.maximum(d, 0) | np.maximum(d, 0.0) | np.clip(d, 0, None) | d.clip(min=0) | d.clip(0) ? Cancellation can produce tiny negative values (e.g. $-10^{-15}$) for identical points. A squared distance can't be negative, so clamp. }}
:::

:::solution{lang=python}
import numpy as np

def sq_dists_broadcast(X, Y):
    diff = X[:, None, :] - Y[None, :, :]
    return (diff ** 2).sum(axis=-1)

def sq_dists_expand(X, Y):
    xx = (X ** 2).sum(axis=1)[:, None]
    yy = (Y ** 2).sum(axis=1)[None, :]
    d = xx + yy - 2 * X @ Y.T
    return np.maximum(d, 0)
:::

:::explain
[L4] **Insert singleton axes.** `X[:, None, :]` is `(N, 1, D)` and `Y[None, :, :]` is `(1, M, D)`. Broadcasting fills in the `(N, M, D)` grid of differences. This is easy to read, but costly: with $N = M = 10^4$ and $D = 512$ it's 200 GB in float32.

[L5] **Reduce over features.** Summing the squares over the last axis leaves `(N, M)`.

[L8-9] **Squared norms as a column and a row.** `(N, 1) + (1, M)` broadcasts to the full `(N, M)` grid. This is the "outer sum" pattern.

[L10] **One matmul does the heavy lifting.** `X @ Y.T` is $O(NMD)$ *compute* but only $O(NM)$ *memory*, and it runs on optimised BLAS. That's how every production k-NN and kernel method does it.

[L11] **Clamp.** For $x_i = y_j$, the three terms are large and nearly cancel, so floating-point error can leave $-10^{-14}$. Without the clamp, `np.sqrt` would return NaN.
:::

:::tests{lang=python}
import numpy as np

def _loops(X, Y):
    return np.array([[((x - y) ** 2).sum() for y in Y] for x in X])

def test_broadcast_matches_loops():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(5, 3)), rng.normal(size=(7, 3))
    assert sq_dists_broadcast(X, Y).shape == (5, 7)
    assert np.allclose(sq_dists_broadcast(X, Y), _loops(X, Y))

def test_expand_matches_loops():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(6, 4)), rng.normal(size=(2, 4))
    assert sq_dists_expand(X, Y).shape == (6, 2)
    assert np.allclose(sq_dists_expand(X, Y), _loops(X, Y))

def test_self_distance_is_never_negative():
    X = np.random.default_rng(2).normal(size=(50, 8)) * 1e3
    d = sq_dists_expand(X, X)
    assert (d >= 0).all() and np.allclose(np.diag(d), 0, atol=1e-6 * 1e6)
:::

:::hints
1. Write the target shape first: `(N, M)`. Which axes do `X` and `Y` each need to "own"?
2. In the expansion, `xx` must vary down the rows and `yy` across the columns.
3. If the self-distance test fails, print `d.min()`.
:::

:::interview
- Memory and compute of each version? When would you pick the broadcast one? *(Small inputs, or when you need the difference vectors themselves.)*
- Why can the expansion be numerically worse? *(Catastrophic cancellation when $\|x\| \approx \|y\|$ is large and the points are close.)*
- How would you find the nearest neighbour of each row of `X` among `Y`? *(`argmin` over axis 1. You don't need the square root, because it's monotonic.)*
:::
::::

## `einsum`: say the indices, get the code

`np.einsum` and `torch.einsum` express any combination of products and sums by **naming axes**. Indices that appear in the output are kept; indices that don't are summed over. It's the clearest way to write batched linear algebra, and interviewers often accept it as "showing you understand the shapes".

::::exercise{id=einsum-drill type=implement difficulty=2 time=12 runtime=browser udl="—" tags="tensor-shapes"}
# Rewrite with einsum

:::prereq
`einsum("ij,jk->ik", A, B)` is a matrix product: $j$ appears in both inputs but not in the output, so it's summed.

| Pattern | Meaning |
|---|---|
| `"ii->"` | trace |
| `"ij->ji"` | transpose |
| `"bij,bjk->bik"` | batched matmul |
| `"i,j->ij"` | outer product |
| `"bi,ij,bj->b"` | batched bilinear form $x_b^\top W y_b$ |
:::

:::task
Implement each function with a **single** `np.einsum` call:

| Function | Inputs | Output |
|---|---|---|
| `batch_matmul(A, B)` | `(B, N, K)`, `(B, K, M)` | `(B, N, M)` |
| `batch_trace(A)` | `(B, N, N)` | `(B,)` |
| `bilinear(x, W, y)` | `(B, D)`, `(D, E)`, `(B, E)` | `(B,)`, entries $x_b^\top W y_b$ |
| `attention_scores(q, k)` | `(B, H, T, D)`, `(B, H, S, D)` | `(B, H, T, S)` |
:::

:::starter{lang=python}
import numpy as np

def batch_matmul(A, B):
    raise NotImplementedError

def batch_trace(A):
    raise NotImplementedError

def bilinear(x, W, y):
    raise NotImplementedError

def attention_scores(q, k):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def batch_matmul(A, B):
    return np.einsum({{ "bnk,bkm->bnm" | "bij,bjk->bik" | "bik,bkj->bij" | "...ij,...jk->...ik" ? Keep the batch index in both inputs *and* the output. The shared inner index is summed because it's missing from the output. }}, A, B)

def batch_trace(A):
    return np.einsum({{ "bii->b" | "...ii->..." | "bjj->b" ? A repeated index *within one operand* walks the diagonal. Dropping it from the output sums it. }}, A)

def bilinear(x, W, y):
    return np.einsum({{ "bi,ij,bj->b" | "bd,de,be->b" ? Three operands: $x_{bi} W_{ij} y_{bj}$ summed over $i$ and $j$. Only $b$ survives. }}, x, W, y)

def attention_scores(q, k):
    return np.einsum({{ "bhtd,bhsd->bhts" | "bhqd,bhkd->bhqk" | "...td,...sd->...ts" | "bhid,bhjd->bhij" ? Contract the feature index $d$ and keep both positions $t$ and $s$. This is $QK^\top$ without any transpose. }}, q, k)
:::

:::solution{lang=python}
import numpy as np

def batch_matmul(A, B):
    return np.einsum("bnk,bkm->bnm", A, B)

def batch_trace(A):
    return np.einsum("bii->b", A)

def bilinear(x, W, y):
    return np.einsum("bi,ij,bj->b", x, W, y)

def attention_scores(q, k):
    return np.einsum("bhtd,bhsd->bhts", q, k)
:::

:::explain
[L4] **Batched matmul.** `b` is kept and `k` is contracted. It's identical to `A @ B`.

[L7] **Trace.** Writing `ii` makes einsum read only the diagonal, and leaving `i` out of the output sums it. There's no Python loop and no `np.diagonal` copy.

[L10] **Bilinear form.** Three operands in one call. Without einsum you'd write `((x @ W) * y).sum(-1)`, which is correct, but you have to think harder to check it.

[L13] **Attention scores.** This is $QK^\top$ per head. Naming `t` and `s` separately makes it obvious the result is (queries × keys), with no `swapaxes` needed. Compare Chapter 13.

**A habit worth having:** after writing an einsum string, read it back as "for each b, h, t and s, sum over d of q times k". If you can say it, it's right.
:::

:::tests{lang=python}
import numpy as np

def test_batch_matmul():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(3, 4, 5)), rng.normal(size=(3, 5, 2))
    assert np.allclose(batch_matmul(A, B), A @ B)

def test_batch_trace():
    A = np.random.default_rng(1).normal(size=(6, 4, 4))
    assert np.allclose(batch_trace(A), np.trace(A, axis1=1, axis2=2))

def test_bilinear():
    rng = np.random.default_rng(2)
    x, W, y = rng.normal(size=(5, 3)), rng.normal(size=(3, 4)), rng.normal(size=(5, 4))
    assert np.allclose(bilinear(x, W, y), [x[b] @ W @ y[b] for b in range(5)])

def test_attention_scores():
    rng = np.random.default_rng(3)
    q, k = rng.normal(size=(2, 3, 4, 8)), rng.normal(size=(2, 3, 6, 8))
    assert np.allclose(attention_scores(q, k), q @ np.swapaxes(k, -1, -2))
:::

:::hints
1. Give every axis a letter. Letters in the output are kept; the rest are summed.
2. A repeated letter inside one operand (`ii`) means "the diagonal".
:::

:::interview
- Is einsum slower than `@`? *(It can be. Modern backends route common patterns to BLAS, and `np.einsum(..., optimize=True)` chooses a contraction order.)*
- Write multi-head attention's output projection in einsum. *(`"bhtd,hde->bte"` merges the heads and projects in one step.)*
:::
::::

## Views, copies and a silent bug

::::exercise{id=inplace-alias-bug type=debug difficulty=2 time=10 runtime=browser udl="—" tags="debugging,tensor-shapes"}
# The normaliser that edits your dataset

:::prereq
**Assignment never copies an array.** `a = x` makes `a` another name for the same buffer. Slices are **views** too.

**In-place operators modify the buffer.** `x -= m` and `x /= s` modify the existing array. `x = x - m` creates a new array and rebinds the name.

**Function arguments are passed by reference.** If a function does `x -= ...`, the *caller's* array changes. The same is true in PyTorch, where in-place ops end with an underscore: `x.sub_(m)`.
:::

:::task
`standardise` is meant to return a zero-mean, unit-variance copy of a dataset. Training works for the first epoch. On the second epoch the "raw" data already has mean 0, and a downstream check that compares the raw data with a saved checksum fails.

Find the bug and fix it **without** changing the function's interface.
:::

:::starter{lang=python}
import numpy as np

def standardise(X, eps=1e-8):
    """X: (N, D) -> standardised copy with per-feature mean 0 and std 1. Must not modify X."""
    X -= X.mean(axis=0)
    X /= X.std(axis=0) + eps
    return X
:::

:::scaffold{lang=python}
import numpy as np

def standardise(X, eps=1e-8):
    """X: (N, D) -> standardised copy with per-feature mean 0 and std 1. Must not modify X."""
    Z = {{ X - X.mean(axis=0) | X - X.mean(0) | X.copy() - X.mean(axis=0) | X - np.mean(X, axis=0) ? `X -= …` writes into the caller's buffer. Build a **new** array with an out-of-place operation. After that, in-place edits on the new array are safe. }}
    Z /= Z.std(axis=0) + eps
    return Z
:::

:::solution{lang=python}
import numpy as np

def standardise(X, eps=1e-8):
    """X: (N, D) -> standardised copy with per-feature mean 0 and std 1. Must not modify X."""
    Z = X - X.mean(axis=0)
    Z /= Z.std(axis=0) + eps
    return Z
:::

:::explain
[L5] **The fix: the first subtraction is out-of-place.** `X - X.mean(axis=0)` allocates a fresh array `Z`, so the caller's `X` is never touched.

[L6] **In-place on your own buffer is fine.** `Z` belongs to this function, so `Z /= …` saves an allocation. That's the right use of in-place ops.

**Why it was hard to spot.** The function's *output* was correct every time. Only a side effect was wrong, and it surfaced an epoch later and somewhere else. Tests that only check return values miss it, so **assert that inputs are unchanged**.

**In PyTorch** the same mistake can also break autograd. An in-place op on a tensor that's needed for the backward pass raises `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`.
:::

:::tests{lang=python}
import numpy as np

def test_output_is_standardised():
    X = np.random.default_rng(0).normal(3.0, 2.0, size=(200, 4))
    Z = standardise(X)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-10) and np.allclose(Z.std(axis=0), 1, atol=1e-6)

def test_input_is_not_modified():
    X = np.random.default_rng(1).normal(5.0, 3.0, size=(100, 3))
    before = X.copy()
    standardise(X)
    assert np.array_equal(X, before)
:::

:::hints
1. Call it on an array, then print the array you passed in.
2. Which operators write into an existing buffer, and which create a new one?
:::

:::interview
- Name three NumPy or PyTorch operations that return views. *(Slicing, `reshape`/`view` when possible, `transpose`, `expand`.)*
- When does PyTorch refuse an in-place op during training, and why?
- How do you check whether two arrays share memory? *(`np.shares_memory(a, b)`; in PyTorch, compare `x.data_ptr()`.)*
:::
::::
