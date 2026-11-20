---
id: ch10-normalisation
summary: Normalisation layers and residual connections are what make deep networks trainable at all. Derive and code batch norm's backward pass, implement LayerNorm and RMSNorm, then predict what a residual stack does without normalisation. The answer surprises most people.
udl: "Ch. 11 — Residual networks"
---

## Two ideas that made depth work

Chapter 06 showed that signal scale compounds multiplicatively with depth. Good initialisation only fixes that **at step 0**. Two architectural ideas keep it under control **throughout training**:

- **Normalisation** re-standardises activations inside the network. **BatchNorm** normalises each feature *across the batch*. **LayerNorm** normalises each example *across its features*. **RMSNorm** is LayerNorm without the mean subtraction.
- **Residual connections**, $h_{\ell+1} = h_\ell + f(h_\ell)$, give gradients an identity path: $\partial h_{\ell+1}/\partial h_\ell = I + \partial f/\partial h_\ell$, so the product over layers can't collapse to zero.

| | Normalises over | Train vs eval differ? | Typical home |
|---|---|---|---|
| BatchNorm | the batch (per feature) | **yes**: running statistics at eval | CNNs |
| LayerNorm | features (per example) | no | transformers, RNNs |
| RMSNorm | features, no centring | no | modern LLMs (LLaMA etc.) |

::::exercise{id=batchnorm-numpy type=implement difficulty=3 time=25 runtime=browser udl="§11.4" tags="normalisation,backprop"}
# Batch norm: forward and backward

:::prereq
**Forward** (training mode), per feature $j$, over a batch of $N$:
$$\mu_j = \tfrac1N\textstyle\sum_i x_{ij},\quad \sigma_j^2 = \tfrac1N\textstyle\sum_i (x_{ij}-\mu_j)^2,\quad \hat x_{ij} = \dfrac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}},\quad y_{ij} = \gamma_j\hat x_{ij} + \beta_j.$$

**Backward.** Every $\hat x_{ij}$ depends on *every* $x_{kj}$ in the batch, through $\mu_j$ and $\sigma_j$. Carrying the chain rule through both gives a compact result, with $g = \partial L/\partial \hat x = \gamma \odot \partial L/\partial y$:
$$\frac{\partial L}{\partial x} = \frac{1}{\sqrt{\sigma^2+\epsilon}}\Big(g - \overline{g} - \hat x\odot\overline{g\odot\hat x}\Big),$$
where the bar means the mean over the batch, per feature. Read it as: the upstream gradient, minus its component along the "shift everything" direction, minus its component along the "scale everything" direction. BatchNorm is invariant to both, so it can't pass those components back.
:::

:::task
Implement `batchnorm_forward(x, gamma, beta, eps=1e-5)`, returning `(out, cache)`, and `batchnorm_backward(dout, cache)`, returning `(dx, dgamma, dbeta)`. `x` has shape `(N, D)`, and `gamma` and `beta` have shape `(D,)`. Use the population variance (divide by $N$), as frameworks do in training mode.
:::

:::starter{lang=python}
import numpy as np

def batchnorm_forward(x, gamma, beta, eps=1e-5):
    """x: (N, D) -> (out: (N, D), cache)."""
    raise NotImplementedError

def batchnorm_backward(dout, cache):
    """dout: (N, D) -> (dx: (N, D), dgamma: (D,), dbeta: (D,))."""
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def batchnorm_forward(x, gamma, beta, eps=1e-5):
    mu = {{ x.mean(axis=0) | x.mean(0) | np.mean(x, axis=0) ? BatchNorm averages over the **batch**, axis 0, giving one mean per feature. }}
    var = x.var(axis=0)
    xhat = (x - mu) / {{ np.sqrt(var + eps) | (var + eps) ** 0.5 | np.sqrt(eps + var) ? Divide by the standard deviation, with ε **inside** the square root so it never divides by 0. }}
    return gamma * xhat + beta, (xhat, var, gamma, eps)

def batchnorm_backward(dout, cache):
    xhat, var, gamma, eps = cache
    dbeta = {{ dout.sum(axis=0) | dout.sum(0) | np.sum(dout, axis=0) ? `beta` is added to every row, so its gradient sums over the batch. }}
    dgamma = {{ (dout * xhat).sum(axis=0) | (xhat * dout).sum(axis=0) | np.sum(dout * xhat, axis=0) | (dout * xhat).sum(0) ? `gamma` multiplies `xhat`, so its gradient is upstream × `xhat`, summed over the batch. }}
    g = dout * gamma
    dx = (g - {{ g.mean(axis=0) | g.mean(0) | np.mean(g, axis=0) ? Remove the component of the gradient that would shift every example equally. BatchNorm subtracts the mean, so it is blind to that direction. }} - xhat * {{ (g * xhat).mean(axis=0) | (xhat * g).mean(axis=0) | np.mean(g * xhat, axis=0) | (g * xhat).mean(0) ? Remove the component along $\hat x$ itself: the "scale everything" direction that the division by σ cancels out. }}) / np.sqrt(var + eps)
    return dx, dgamma, dbeta
:::

:::solution{lang=python}
import numpy as np

def batchnorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma * xhat + beta, (xhat, var, gamma, eps)

def batchnorm_backward(dout, cache):
    xhat, var, gamma, eps = cache
    dbeta = dout.sum(axis=0)
    dgamma = (dout * xhat).sum(axis=0)
    g = dout * gamma
    dx = (g - g.mean(axis=0) - xhat * (g * xhat).mean(axis=0)) / np.sqrt(var + eps)
    return dx, dgamma, dbeta
:::

:::explain
[L4-6] **Normalise per feature, across the batch.** `x.var` divides by $N$, which matches PyTorch in training mode. (PyTorch's *running* variance uses the unbiased $N-1$ estimate, which is a classic trick question.)

[L7] **Cache what backward needs.** `xhat` and `var` are enough. You don't need to recompute $\mu$.

[L11-12] **Parameter gradients.** These are the bias and scale rules from Chapter 05, summed over the batch.

[L13-14] **The input gradient.** This is the projection form. Two consequences fall straight out of it, and the tests check both: `dx` sums to zero over the batch, and it's orthogonal to `xhat` (exactly as ε → 0, and up to about ε otherwise). Adding a constant to every example, or rescaling them all, can't change the loss, so the gradient has no component in those directions.

**Eval mode.** At test time there may be a single example, so BatchNorm uses *running averages* of $\mu$ and $\sigma^2$ collected during training. Forgetting `model.eval()` therefore changes predictions, and batches of size 1 in training mode give $\hat x = 0$.
:::

:::tests{lang=python}
import numpy as np

def test_forward_normalises_each_feature():
    x = np.random.default_rng(0).normal(3, 5, size=(64, 4))
    out, _ = batchnorm_forward(x, np.ones(4), np.zeros(4))
    assert np.allclose(out.mean(axis=0), 0, atol=1e-7) and np.allclose(out.std(axis=0), 1, atol=1e-3)

def test_gamma_beta_applied():
    x = np.random.default_rng(1).normal(size=(32, 3))
    out, _ = batchnorm_forward(x, np.array([2.0, 3.0, 4.0]), np.array([1.0, -1.0, 0.5]))
    assert np.allclose(out.mean(axis=0), [1.0, -1.0, 0.5], atol=1e-7)
    assert np.allclose(out.std(axis=0), [2.0, 3.0, 4.0], atol=1e-3)

def test_backward_matches_finite_differences():
    rng = np.random.default_rng(2)
    x, gamma, beta, R = rng.normal(size=(6, 3)), rng.normal(size=3), rng.normal(size=3), rng.normal(size=(6, 3))
    loss = lambda x, g, b: (batchnorm_forward(x, g, b)[0] * R).sum()
    _, cache = batchnorm_forward(x, gamma, beta)
    dx, dg, db = batchnorm_backward(R, cache)
    eps = 1e-6
    for arr, grad, which in [(x, dx, 0), (gamma, dg, 1), (beta, db, 2)]:
        num = np.zeros_like(arr)
        for idx in np.ndindex(*arr.shape):
            args = [x.copy(), gamma.copy(), beta.copy()]
            args[which][idx] += eps
            hi = loss(*args)
            args[which][idx] -= 2 * eps
            num[idx] = (hi - loss(*args)) / (2 * eps)
        assert np.allclose(grad, num, atol=1e-5), which

def test_input_gradient_is_projected():
    rng = np.random.default_rng(3)
    x, R = rng.normal(size=(10, 2)), rng.normal(size=(10, 2))
    _, cache = batchnorm_forward(x, np.ones(2), np.zeros(2))
    dx, _, _ = batchnorm_backward(R, cache)
    assert np.allclose(dx.sum(axis=0), 0, atol=1e-10) and np.allclose((dx * cache[0]).sum(axis=0), 0, atol=1e-4)
:::

:::hints
1. The forward pass is three lines: the mean and variance over axis 0, then normalise.
2. In backward, `dbeta` and `dgamma` are the same "sum over the batch" pattern as any bias and scale.
3. For `dx`, trust the projection formula in the prerequisites and translate it term by term.
:::

:::interview
- Why does BatchNorm behave differently at train and test time? What breaks with batch size 1?
- Why can BatchNorm let you drop the preceding layer's bias? *(The mean subtraction cancels any constant.)*
- Why is BatchNorm awkward for transformers and RNNs? *(Variable lengths, small batches, and statistics leaking between examples.)*
- Running variance: biased or unbiased? *(PyTorch normalises with the biased estimate during training, but stores the unbiased one in its running statistics.)*
:::
::::

::::exercise{id=layernorm-rmsnorm type=implement difficulty=1 time=10 runtime=browser udl="§11.4, §12.4" tags="normalisation,transformers"}
# LayerNorm and RMSNorm

:::prereq
**LayerNorm** standardises each example over its **feature** axis (the last one), then applies a learned per-feature $\gamma$ and $\beta$:
$$\mathrm{LN}(x) = \gamma\odot\frac{x-\mu(x)}{\sqrt{\sigma^2(x)+\epsilon}} + \beta .$$
It doesn't look at other examples in the batch, so it behaves identically at train and test time, for any batch size or sequence length.

**RMSNorm** drops the centring and the bias: $\mathrm{RMSNorm}(x) = \gamma\odot x\,/\sqrt{\overline{x^2}+\epsilon}$. That's cheaper, and it works just as well in large language models.
:::

:::task
Implement `layer_norm(x, gamma, beta, eps=1e-5)` and `rms_norm(x, gamma, eps=1e-6)` for inputs of any shape `(…, D)`, normalising over the last axis.
:::

:::starter{lang=python}
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    raise NotImplementedError

def rms_norm(x, gamma, eps=1e-6):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis={{ -1, keepdims=True | -1, keepdims=1 ? Normalise each example over its **own features**: the last axis, kept so the result broadcasts back. }})
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def rms_norm(x, gamma, eps=1e-6):
    rms = {{ np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps) | np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps) | np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) | ((x ** 2).mean(axis=-1, keepdims=True) + eps) ** 0.5 ? The root of the mean square over the feature axis, with no mean subtraction. Keep the axis so it broadcasts. }}
    return {{ gamma * x / rms | x / rms * gamma | gamma * (x / rms) ? Rescale by the RMS, then apply the learned gain. RMSNorm has no bias. }}
:::

:::solution{lang=python}
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def rms_norm(x, gamma, eps=1e-6):
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return gamma * x / rms
:::

:::explain
[L4-6] **Per-example statistics.** `keepdims=True` makes `mu` and `var` shape `(…, 1)`, so they broadcast across features. `gamma` and `beta` have shape `(D,)` and broadcast across everything else.

[L9-10] **RMSNorm.** There's no centring, so the result is invariant to *scaling* the input but not to *shifting* it. The tests check both properties for each layer.

**Batch independence.** Normalising one example on its own gives exactly the same result as normalising it inside a batch, which BatchNorm can't offer. That's why transformers use LayerNorm or RMSNorm.
:::

:::tests{lang=python}
import numpy as np

def test_layernorm_rows_standardised_3d():
    x = np.random.default_rng(0).normal(4, 3, size=(2, 5, 16))
    y = layer_norm(x, np.ones(16), np.zeros(16))
    assert y.shape == x.shape and np.allclose(y.mean(-1), 0, atol=1e-7) and np.allclose(y.std(-1), 1, atol=1e-3)

def test_layernorm_independent_of_batch():
    x = np.random.default_rng(1).normal(size=(8, 6))
    g, b = np.full(6, 1.5), np.full(6, 0.2)
    assert np.allclose(layer_norm(x, g, b)[3], layer_norm(x[3:4], g, b)[0])

def test_layernorm_shift_and_scale_invariant():
    x = np.random.default_rng(2).normal(size=(4, 10))
    g, b = np.ones(10), np.zeros(10)
    assert np.allclose(layer_norm(3 * x + 7, g, b), layer_norm(x, g, b), atol=1e-4)

def test_rmsnorm_scale_but_not_shift_invariant():
    x = np.random.default_rng(3).normal(size=(4, 10))
    g = np.ones(10)
    assert np.allclose(rms_norm(5 * x, g), rms_norm(x, g), atol=1e-5)
    assert not np.allclose(rms_norm(x + 3, g), rms_norm(x, g), atol=1e-2)
    assert np.allclose(np.sqrt((rms_norm(x, g) ** 2).mean(-1)), 1, atol=1e-4)
:::

:::hints
1. Everything reduces over `axis=-1` with `keepdims=True`.
2. RMSNorm is LayerNorm minus two things: the centring and $\beta$.
:::

:::interview
- BatchNorm versus LayerNorm: what does each normalise over, and why do transformers use LayerNorm?
- Why do many large language models use RMSNorm?
- Pre-LN versus post-LN transformers: where does the norm go, and why does pre-LN train more stably? *(Chapter 14.)*
:::
::::

## Residuals: a free lunch?

::::exercise{id=residual-scale-predict type=predict difficulty=2 time=8 runtime=browser udl="§11.2–11.3" tags="residuals,vanishing-gradients"}
# A deep residual stack with no normalisation

:::task
Take Chapter 06's 50-layer ReLU network with the "too small" initialisation, $W \sim \mathcal N(0, 1/D)$, where the **gradient at the input** shrank by about $10^{-8}$. Now make every layer **residual**: $h_{\ell+1} = h_\ell + \mathrm{ReLU}(h_\ell W_\ell)$, with no normalisation layers. Relative to the gradient at the output, what does the gradient at the input do?
:::

:::choices
- [ ] Still vanishes, by roughly the same factor
- [ ] Stays about the same size: the identity path preserves it
- [x] Grows by many orders of magnitude
- [ ] Becomes exactly equal to the output gradient
:::

:::experiment{lang=python}
import numpy as np

def input_grad_ratio(depth=50, D=64, residual=False, alpha=1.0, seed=0):
    """‖dL/dh_0‖ / ‖dL/dh_L‖ for a ReLU stack, weights ~ N(0, 1/D)."""
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(0, np.sqrt(1 / D), (D, D)) for _ in range(depth)]
    h, masks = rng.normal(size=(1, D)), []
    for W in Ws:
        z = h @ W
        masks.append(z > 0)
        h = h + alpha * np.maximum(z, 0) if residual else np.maximum(z, 0)
    g0 = rng.normal(size=(1, D))
    g = g0
    for W, m in zip(reversed(Ws), reversed(masks)):
        through = (g * m) @ W.T
        g = g + alpha * through if residual else through
    return np.linalg.norm(g) / np.linalg.norm(g0)

print("plain                     :", f"{input_grad_ratio():.2e}")
print("residual, alpha = 1       :", f"{input_grad_ratio(residual=True):.2e}")
print("residual, alpha = 1/sqrt(L):", f"{input_grad_ratio(residual=True, alpha=1 / np.sqrt(50)):.2e}")
:::

:::tests{lang=python}
import numpy as np

def test_plain_vanishes():
    assert input_grad_ratio() < 1e-6

def test_naive_residual_explodes():
    assert input_grad_ratio(residual=True) > 1e4

def test_scaled_residual_is_well_behaved():
    assert 0.3 < input_grad_ratio(residual=True, alpha=1 / np.sqrt(50)) < 10
:::

:::explain
The residual connection does fix **vanishing**: the Jacobian of each block is $I + J_f$, so there's always an identity path. But the *variance* now compounds **additively**. Each block adds a branch output of comparable size to what's already there, so the scale grows by roughly $(1 + c)$ per block, and $(1.5)^{50} \approx 6\times 10^{8}$. Gradients ride the same growth on the way back.

That's why real residual networks always pair the skip connection with **something that controls the branch scale**:

- **Normalisation inside the branch** (pre-LN transformers, and BN in ResNets). The branch then adds a bounded amount at each layer.
- **Down-weighting the branch.** Scale it by about $1/\sqrt{L}$, or **initialise its last layer to zero** (Fixup, ReZero, "zero-init residual" in ResNets). At step 0 each block is then roughly the identity, and the network can only grow from there. The experiment's $\alpha = 1/\sqrt{L}$ row shows the effect: the gradient stays $O(1)$.

**Interview takeaway.** "Residuals fix vanishing gradients" is half the story. They give gradients a *path*; normalisation or branch scaling keeps the *magnitude* sane.
:::

:::interview
- Why do residual connections help optimisation? What's the Jacobian of $h + f(h)$?
- Why do residual networks still need normalisation or careful branch initialisation?
- What do zero-init and ReZero do at initialisation? *(Each block starts as the identity.)*
- Residual networks as ensembles of shallow paths: what's the argument?
:::
::::
