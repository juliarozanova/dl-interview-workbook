---
id: ch04-losses
summary: Every standard loss is a negative log-likelihood in disguise. Implement softmax cross-entropy so it never overflows, derive its gradient on a whiteboard, and hunt a broadcasting bug that hides behind a passing sanity check.
weeks: [2]
udl: "Ch. 5 — Loss functions"
---

## The one recipe behind every loss

A network doesn't predict *the answer*. It predicts the **parameters of a distribution** over answers. Training then maximises the likelihood of the observed data under that distribution. Equivalently, it minimises the **negative log-likelihood** (NLL):

$$
\hat\phi = \arg\min_\phi \; \Big[-\sum_{i} \log \Pr\big(y_i \mid f(x_i;\phi)\big)\Big].
$$

Choose the distribution and you get the loss:

| Task | Distribution over $y$ | Network outputs | Loss you get |
|---|---|---|---|
| Regression | Normal $\mathcal N(\mu, \sigma^2)$ | $\mu$ (fixed $\sigma$) | mean squared error |
| Binary classification | Bernoulli$(\lambda)$ | logit $z$, with $\lambda = \sigma(z)$ | binary cross-entropy |
| Multiclass classification | Categorical$(p)$ | logits $z \in \mathbb R^C$, with $p = \mathrm{softmax}(z)$ | softmax cross-entropy |

This chapter focuses on the last row, the loss behind almost every classifier and language model.

::::exercise{id=softmax-ce-stable type=implement difficulty=2 time=15 runtime=browser udl="§5.5, §5.7" tags="custom-losses,numerical-stability"}
# Numerically stable softmax cross-entropy

:::prereq
**Softmax** turns a row of logits $z \in \mathbb R^C$ into probabilities:
$$\mathrm{softmax}(z)_c = \frac{e^{z_c}}{\sum_{j} e^{z_j}}.$$

**Shift invariance.** Adding the same constant $m$ to every logit changes nothing, because $e^{m}$ cancels top and bottom. Choosing $m = -\max_j z_j$ makes the largest exponent $e^0 = 1$, so nothing overflows.

**Why we need it.** `float64` overflows at $e^{710}$; `float32` overflows at $e^{89}$. Logits of a few hundred are routine in a badly scaled network.

**Log space.** Taking logs of the softmax gives
$$\log \mathrm{softmax}(z)_c = z_c - \underbrace{\log \textstyle\sum_j e^{z_j}}_{\text{log-sum-exp}}.$$
Never compute `log(softmax(z))`. A tiny probability underflows to $0$, and $\log 0 = -\infty$.

**Cross-entropy** for integer labels $y_i$ is the mean NLL of the true class:
$$L = -\frac{1}{B}\sum_{i=1}^{B} \log \mathrm{softmax}(z_i)_{y_i}.$$
:::

:::task
Implement `log_softmax` and `softmax_ce` in NumPy. Your functions must survive logits of magnitude $10^4$.

| Argument | Shape | dtype | Notes |
|---|---|---|---|
| `z`, `logits` | `(B, C)` | float | unnormalised scores |
| `y` | `(B,)` | int | class indices in `[0, C)` |
| **returns** `log_softmax` | `(B, C)` | float | each row's `exp` sums to 1 |
| **returns** `softmax_ce` | `()` | float | mean over the batch |

Test against these edge cases: very large logits, a *confidently wrong* prediction (true-class logit far below the rest), and uniform logits (the loss should be $\log C$).
:::

:::starter{lang=python}
import numpy as np

def log_softmax(z):
    """z: (B, C) logits -> (B, C) log-probabilities."""
    raise NotImplementedError

def softmax_ce(logits, y):
    """logits: (B, C) float, y: (B,) int in [0, C) -> scalar mean loss."""
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def log_softmax(z):
    m = {{ z.max(axis=1, keepdims=True) | z.max(1, keepdims=True) | np.max(z, axis=1, keepdims=True) | z.max(axis=-1, keepdims=True) | z.max(-1, keepdims=True) ? Subtract each row's largest logit. Keep the class axis (`keepdims=True`) so the `(B, 1)` column broadcasts against `(B, C)`. }}
    z = z - m
    lse = np.log(np.exp(z).sum(axis=1, keepdims=True))
    return {{ z - lse ? In log space, softmax is $z_c - \log\sum_j e^{z_j}$. No division and no second `exp`. }}

def softmax_ce(logits, y):
    logp = log_softmax(logits)
    B = logits.shape[0]
    picked = logp[{{ np.arange(B) | range(B) | np.arange(len(y)) ? Integer-array indexing pairs row $i$ with column $y_i$, so you need one row index per sample. }}, y]
    return {{ -picked.mean() | -np.mean(picked) | (-picked).mean() | -picked.sum() / B ? The loss is the *negative* log-likelihood, averaged over the batch. }}
:::

:::solution{lang=python}
import numpy as np

def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

def softmax_ce(logits, y):
    logp = log_softmax(logits)
    picked = logp[np.arange(len(y)), y]
    return -picked.mean()
:::

:::explain
[L4] **Shift by the row max.** Afterwards the largest logit in each row is exactly $0$, so `exp` returns at most $1$ and can't overflow. `keepdims=True` keeps the max as shape `(B, 1)`, so it broadcasts across classes and not across samples.

[L5] **Stay in log space.** $\log p_c = z_c - \mathrm{LSE}(z)$. The sum inside the log is at least $1$ (the max term contributes $e^0$), so the log is always finite. You never form a probability that could underflow to zero.

[L8-9] **Pick the true class.** `logp[np.arange(B), y]` is NumPy's integer-array ("fancy") indexing. It reads element `(i, y[i])` for every `i`, which does the job of the one-hot dot product without allocating a `(B, C)` matrix.

[L10] **Negate and average.** Using a mean rather than a sum keeps the gradient scale independent of batch size. That is why changing the batch size doesn't force you to retune the learning rate. `F.cross_entropy` uses `reduction="mean"` by default.
:::

:::tests{lang=python}
import numpy as np

def _ref(logits, y):
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return -np.log(p[np.arange(len(y)), y]).mean()

def test_matches_reference():
    rng = np.random.default_rng(0)
    z, y = rng.normal(size=(8, 5)), rng.integers(0, 5, size=8)
    assert np.isclose(softmax_ce(z, y), _ref(z, y))

def test_log_softmax_rows_normalise():
    z = np.random.default_rng(1).normal(size=(4, 7))
    assert log_softmax(z).shape == (4, 7)
    assert np.allclose(np.exp(log_softmax(z)).sum(axis=1), 1.0)

def test_huge_logits_are_finite():
    z, y = np.array([[1e4, 0.0, -1e4], [-1e4, 1e4, 0.0]]), np.array([0, 1])
    assert np.isclose(softmax_ce(z, y), 0.0)

def test_confidently_wrong_is_large_but_finite():
    z, y = np.array([[0.0, 1000.0]]), np.array([0])
    assert np.isclose(softmax_ce(z, y), 1000.0)

def test_uniform_logits_give_log_C():
    z, y = np.zeros((3, 10)), np.array([0, 4, 9])
    assert np.isclose(softmax_ce(z, y), np.log(10))
:::

:::hints
1. You only need one identity: softmax doesn't change if you add the same constant to every logit in a row.
2. Work in log space: $\log p_c = z_c - \log\sum_j e^{z_j}$. Only exponentiate *after* shifting by the row max.
3. The confidently-wrong test fails if you compute `np.log(softmax(z))`: $e^{-1000}$ underflows to exactly $0$.
:::

:::interview
- Why is `log(softmax(z))` still unsafe even after max-subtraction? *(The true class can underflow to probability 0.)*
- PyTorch's `F.cross_entropy` takes **logits**, not probabilities. Why? *(It fuses `log_softmax` with `nll_loss`, for stability and a cheaper backward pass.)*
- How would you add label smoothing to this function?
- What's the loss of an untrained classifier with 1000 classes? *(About $\log 1000 \approx 6.9$. It's a great first sanity check.)*
:::
::::

## From loss to gradient

In an interview, the most common follow-up to "implement cross-entropy" is "now differentiate it". The result is short enough to memorise, but you should be able to *derive* it out loud.

::::exercise{id=softmax-ce-grad type=derive difficulty=2 time=10 runtime=browser udl="§5.7, Ch. 7" tags="backprop,softmax-ce-derivation"}
# Backprop through softmax + cross-entropy

:::prereq
- **Kronecker delta:** $\delta_{ij} = 1$ if $i=j$, otherwise $0$.
- **One-hot target:** $e_y$ is the vector with a $1$ at index $y$, so $(e_y)_i = \delta_{iy}$.
- **Quotient rule:** $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$.
- **Chain rule:** $\frac{\partial L}{\partial z_i} = \sum_j \frac{\partial L}{\partial p_j}\frac{\partial p_j}{\partial z_i}$.
:::

:::task
For a single example with logits $z\in\mathbb R^C$, true class $y$, $p=\mathrm{softmax}(z)$ and $L=-\log p_y$, **derive $\partial L/\partial z_i$**.

Say each step out loud as if at a whiteboard, and aim for under five minutes. Then give the batched gradient for the *mean* loss over $B$ samples.

Strong answers mention **two routes**: differentiate the log-sum-exp form directly, or use the softmax Jacobian with the chain rule.
:::

:::scaffold
Let $p=\mathrm{softmax}(z)$ and $L=-\log p_y$. Type answers in plain text or LaTeX. For example, `\delta_{ij}` and `δ_ij` are both accepted.

**1. Softmax Jacobian.** $\dfrac{\partial p_j}{\partial z_i} =$ {{ p_j(\delta_{ij} - p_i) | p_j(δ_ij - p_i) | p_jδ_ij - p_jp_i | p_j\delta_{ij} - p_jp_i | re:^p_?\{?j\}?\((δ|\\delta)_?\{?ij\}?-p_?\{?i\}?\)$ ? Apply the quotient rule to $e^{z_j}/\sum_k e^{z_k}$. The numerator depends on $z_i$ only when $i=j$; the denominator always does. }}

**2. Chain rule.** $\dfrac{\partial L}{\partial z_i} = -\dfrac{1}{p_y}\dfrac{\partial p_y}{\partial z_i} =$ {{ p_i - \delta_{iy} | p_i - δ_iy | p_i - δ_yi | p_i - y_i | p_i - \delta_{yi} | re:^p_?\{?i\}?-(δ|\\delta)_?\{?(iy|yi)\}?$ ? Set $j=y$ in step 1. The $p_y$ cancels, leaving the predicted probability minus the target. }}

**3. Vector form.** $\nabla_z L =$ {{ p - e_y | \mathrm{softmax}(z) - e_y | softmax(z) - e_y | p - y ? Stack the components from step 2 into a vector, using the one-hot target $e_y$. }}

**4. Batch, mean reduction.** Let $P$ be the $(B,C)$ softmax matrix and $Y$ the one-hot targets. Then $\dfrac{\partial \bar L}{\partial Z} =$ {{ (P - Y)/B | \frac{1}{B}(P - Y) | (1/B)(P - Y) | \frac{P - Y}{B} ? Each sample contributes its own row, and averaging over the batch adds a factor of $1/B$. }}
:::

:::derivation
**Write the loss in log-sum-exp form.** This is the fastest route at a whiteboard.
$$L = -\log p_y = -z_y + \log\sum_k e^{z_k}.$$

---step---

**Differentiate term by term.** The first term gives $-\delta_{iy}$. The second gives the softmax itself:
$$\frac{\partial L}{\partial z_i} = -\delta_{iy} + \frac{e^{z_i}}{\sum_k e^{z_k}} = p_i - \delta_{iy}.$$
That's the whole result. Notice you never divided by $p_y$.

---step---

**Cross-check with the softmax Jacobian.** Interviewers often ask for the Jacobian itself. Applying the quotient rule to $p_j = e^{z_j}/S$ with $S=\sum_k e^{z_k}$:
$$\frac{\partial p_j}{\partial z_i} = \frac{\delta_{ij}e^{z_j}S - e^{z_j}e^{z_i}}{S^2} = p_j(\delta_{ij} - p_i),$$

or in matrix form, $J = \mathrm{diag}(p) - pp^\top$. The chain rule then gives

$$\frac{\partial L}{\partial z_i} = -\frac{1}{p_y}\,p_y(\delta_{iy}-p_i) = p_i - \delta_{iy}.\;\checkmark$$

---step---

**Vector and batch form.**
$$\nabla_z L = p - e_y, \qquad \frac{\partial \bar L}{\partial Z} = \frac{1}{B}(P - Y)\ \text{ for } \bar L = \tfrac1B\textstyle\sum_i L_i.$$

---step---

**Say why it matters.**

- Each component lies in $[-1, 1]$, and the gradient shrinks to zero only as $p \to e_y$. It doesn't saturate the way MSE on a sigmoid does.
- The fused backward pass costs $O(C)$, because it never builds the $C\times C$ Jacobian. That, together with stability, is why frameworks fuse softmax with cross-entropy.
- The components sum to zero: $\sum_i (p_i - \delta_{iy}) = 1 - 1$. The gradient doesn't change the logits' common offset, which matches shift invariance.
:::

:::solution{lang=python}
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def softmax_ce_grad(logits, y):
    B, C = logits.shape
    G = softmax(logits)
    G[np.arange(B), y] -= 1.0
    return G / B
:::

:::explain
[L9-10] **$P - Y$ without the one-hot.** Start from the softmax matrix and subtract $1$ at each sample's true class. This is the same fancy-indexing trick as in the forward pass.

[L11] **Mean reduction.** The forward loss averaged over the batch, so each row of the gradient carries a factor of $1/B$. If your gradient check is off by exactly a factor of $B$, this line is the reason.

The tests compare against **central finite differences**, $\frac{L(z+\epsilon) - L(z-\epsilon)}{2\epsilon}$. That error is $O(\epsilon^2)$, much smaller than the $O(\epsilon)$ of one-sided differences. Learn the pattern: you'll reuse it every time you write a backward pass.
:::

:::tests{lang=python}
import numpy as np

def _loss(z, y):
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -logp[np.arange(len(y)), y].mean()

def test_matches_finite_differences():
    rng = np.random.default_rng(0)
    z, y = rng.normal(size=(4, 6)), rng.integers(0, 6, size=4)
    g = softmax_ce_grad(z.copy(), y)
    eps, num = 1e-6, np.zeros_like(z)
    for idx in np.ndindex(*z.shape):
        zp, zm = z.copy(), z.copy()
        zp[idx] += eps
        zm[idx] -= eps
        num[idx] = (_loss(zp, y) - _loss(zm, y)) / (2 * eps)
    assert np.allclose(g, num, atol=1e-6)

def test_rows_sum_to_zero():
    z, y = np.random.default_rng(2).normal(size=(5, 3)), np.array([0, 1, 2, 0, 1])
    assert np.allclose(softmax_ce_grad(z, y).sum(axis=1), 0.0)
:::

:::hints
1. Rewrite $-\log p_y$ using $\log(a/b) = \log a - \log b$ before differentiating anything.
2. $\frac{\partial}{\partial z_i}\log\sum_k e^{z_k}$ is a function you already know.
:::

:::interview
- Write down the softmax Jacobian in matrix form. *($\mathrm{diag}(p) - pp^\top$.)*
- Why don't frameworks backprop through softmax and log separately?
- What's the gradient for *binary* cross-entropy with a sigmoid? *(It has the same form: $\sigma(z) - y$.)*
- Your gradient check fails by exactly a factor of $B$. What happened?
:::
::::

## A bug that passes the sanity check

::::exercise{id=softmax-broadcast-bug type=debug difficulty=1 time=8 runtime=browser udl="§5.5" tags="debugging,broadcasting"}
# The softmax whose rows still sum to one

:::prereq
**NumPy broadcasting rules.**

1. Line the shapes up **from the right**.
2. Two axes are compatible when they're equal or one of them is $1$.
3. A missing leading axis counts as $1$.

So `(B, C) - (C,)` subtracts along columns, `(B, C) - (B, 1)` subtracts along rows, and `(B, C) - (B,)` only works when $B = C$. When it does run, it's wrong.

**Reductions drop the axis.** `z.max(axis=1)` has shape `(B,)`, not `(B, 1)`, unless you pass `keepdims=True`. A 1-D array always lines up with the **last** axis, so `(B,)` behaves like a row `(1, B)`, not a column. Here it is on a square batch, where nothing errors:

@figure(broadcast-keepdims.svg) "With `keepdims=True` the maxima form a column and each row loses its own max. Without it they lie down as a row and stretch down the columns."
:::

:::task
A teammate's softmax passes the "every row sums to 1" check. But:

- With **32 classes and batch size 32**, training plateaus at chance accuracy.
- With **batch size 16**, it crashes with `ValueError: operands could not be broadcast together with shapes (16,32) (16,)`.

Find the bug, explain why the sum-to-one check didn't catch it, and fix it.
:::

:::starter{lang=python}
import numpy as np

def softmax(z):
    """z: (B, C) logits -> (B, C) probabilities."""
    z = z - z.max(axis=1)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
:::

:::scaffold{lang=python}
import numpy as np

def softmax(z):
    """z: (B, C) logits -> (B, C) probabilities."""
    z = z - {{ z.max(axis=1, keepdims=True) | z.max(1, keepdims=True) | z.max(axis=1)[:, None] | z.max(axis=1)[:, np.newaxis] | z.max(-1, keepdims=True) | z.max(axis=-1, keepdims=True) | np.max(z, axis=1, keepdims=True) ? `z.max(axis=1)` has shape `(B,)`. Broadcasting aligns *trailing* axes, so `(B, C) − (B,)` pairs the B row maxima with the C **columns**. It only runs at all when B = C. }}
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
:::

:::solution{lang=python}
import numpy as np

def softmax(z):
    """z: (B, C) logits -> (B, C) probabilities."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
:::

:::explain
[L5] **The fix: keep the reduced axis.** With `keepdims=True` the max has shape `(B, 1)` and broadcasts across each row. The buggy `(B,)` vector was laid along the last axis instead, so column $j$ had row $j$'s maximum subtracted. That's a different constant per *column*, which softmax is **not** invariant to.

[L7] **Why the sanity check passed.** The normalisation divides each row by its own sum, so the output always sums to one, *whatever* was subtracted earlier. Sum-to-one tests the last line; it says nothing about the shift.

**General lesson.** Shape bugs hide when axes have equal sizes. In tests, give every axis a **distinct** size (e.g. `(3, 5)`), compare against a reference implementation, and test the defining property (here, invariance to per-row shifts).
:::

:::tests{lang=python}
import numpy as np

def _ref(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def test_square_batch_matches_reference():
    z = np.random.default_rng(0).normal(size=(4, 4)) * 3
    assert np.allclose(softmax(z), _ref(z))

def test_non_square_batch():
    z = np.random.default_rng(1).normal(size=(16, 32))
    assert np.allclose(softmax(z), _ref(z))

def test_per_row_shift_invariance():
    z = np.random.default_rng(2).normal(size=(3, 5))
    assert np.allclose(softmax(z), softmax(z + np.array([[100.0], [-50.0], [7.0]])))
:::

:::hints
1. Print the shape of every intermediate result for an input of shape `(3, 5)`.
2. The bug is on the line that *protects* against overflow, not the normalisation.
:::

:::interview
- Why did "rows sum to one" pass? What test *would* have caught this?
- What's the PyTorch version of this pitfall? *(`z.max(dim=1)` returns `(values, indices)` and drops the dim. Use `keepdim=True`.)*
- How do you make shape bugs loud in a larger codebase? *(Shape asserts, `einops` patterns, distinct axis sizes in tests, `jaxtyping`-style annotations.)*
:::
::::
