---
id: ch08-generalisation
summary: Test error = noise + bias² + variance. Derive the decomposition at a whiteboard and measure each term on real fits, see why "L2 regularisation" and "weight decay" stop being the same thing under Adam, and implement inverted dropout.
udl: "Ch. 8–9 — Measuring performance, Regularisation"
---

## Where test error comes from

Train a model on a random dataset $\mathcal D$, then evaluate it on a fresh point $x$ with noisy target $y = f(x) + \epsilon$. The test error varies with *which* dataset you happened to get. Averaged over datasets, it splits into three parts with very different cures:

| Term | Meaning | What reduces it |
|---|---|---|
| **Noise** $\sigma^2$ | randomness in the targets themselves | nothing: it's the floor |
| **Bias²** | the *average* model is wrong | more capacity, better features |
| **Variance** | the model changes a lot with the training set | more data, regularisation, ensembling, smaller models |

The classical U-shaped test curve is these two forces trading off. UDL Chapter 8 also covers **double descent**: past the point where the model can interpolate the training data, test error can fall again. So the "trade-off" isn't the whole story for large networks.

::::exercise{id=bias-variance-derive type=derive difficulty=2 time=10 runtime=browser udl="§8.2" tags="bias-variance,bias-variance-derivation,generalisation"}
# The bias–variance decomposition

:::prereq
- **Notation.** $\hat f_{\mathcal D}(x)$ is the model trained on dataset $\mathcal D$, and $\bar f(x) = \mathbb E_{\mathcal D}[\hat f_{\mathcal D}(x)]$ is the **average model**.
- **Assumptions.** The test noise $\epsilon$ has mean $0$ and variance $\sigma^2$, and is independent of the training set.
- **The trick.** Add and subtract the same quantity inside a square, $(a - c)^2 = ((a - b) + (b - c))^2$, then show that the cross term has expectation zero.
:::

:::task
Show that the expected squared error at a point $x$ decomposes as
$$\mathbb E_{\mathcal D,\epsilon}\big[(y - \hat f_{\mathcal D}(x))^2\big] = \sigma^2 + \big(f(x) - \bar f(x)\big)^2 + \mathbb E_{\mathcal D}\big[(\hat f_{\mathcal D}(x) - \bar f(x))^2\big],$$
name each term, and say how each one moves as model capacity grows. Practise it out loud. It's one of the most commonly asked derivations.
:::

:::scaffold
Drop the $(x)$ for brevity, and write $\hat f$ for $\hat f_{\mathcal D}$.

**1. Split off the noise.** $(y - \hat f)^2 = (\epsilon + (f - \hat f))^2$. The cross term $2\,\mathbb E[\epsilon\,(f - \hat f)]$ equals {{ 0 | zero ? $\epsilon$ is independent of $\mathcal D$, so the expectation factorises, and $\mathbb E[\epsilon] = 0$. }}

**2. The noise term.** $\mathbb E[\epsilon^2] =$ {{ σ^2 | \sigma^2 | σ² | sigma^2 ? $\mathbb E[\epsilon^2] = \mathrm{Var}(\epsilon) + \mathbb E[\epsilon]^2$, and the mean is zero. }}

**3. Split around the average model.** $f - \hat f = (f - \bar f) + (\bar f - \hat f)$. The deterministic part $(f - \bar f)^2$ is the {{ squared bias | bias^2 | bias squared | bias² | re:bias ? It measures how far the *average* model is from the truth, and doesn't depend on which dataset you drew. }}

**4. The random part.** $\mathbb E_{\mathcal D}[(\hat f - \bar f)^2]$ is the {{ variance | re:^var ? The spread of the trained model around its own average, across datasets. }}. Its cross term vanishes because $\mathbb E_{\mathcal D}[\bar f - \hat f] = 0$.

**5. Capacity.** As capacity grows, bias usually {{ decreases | falls | goes down | re:^(decreas|fall|drop|shrink|goes?down|lower|reduc) ? A more flexible model class can get closer to $f$ on average. }} and variance usually {{ increases | grows | goes up | rises | re:^(increas|grow|ris|goesup|goup|higher) ? A more flexible model can fit the particular noise in each training set, so it changes more from one dataset to the next. }}.
:::

:::derivation
**Separate the noise.** With $y = f + \epsilon$,
$$\mathbb E[(y-\hat f)^2] = \mathbb E[\epsilon^2] + 2\,\mathbb E[\epsilon]\,\mathbb E[f - \hat f] + \mathbb E[(f-\hat f)^2] = \sigma^2 + \mathbb E_{\mathcal D}[(f-\hat f)^2].$$
The cross term vanishes because the test noise is independent of the training set and has zero mean.

---step---

**Add and subtract the average model** $\bar f = \mathbb E_{\mathcal D}[\hat f]$:
$$\mathbb E_{\mathcal D}[(f-\hat f)^2] = (f-\bar f)^2 + 2\,(f - \bar f)\,\mathbb E_{\mathcal D}[\bar f - \hat f] + \mathbb E_{\mathcal D}[(\hat f - \bar f)^2].$$
$f - \bar f$ is a constant with respect to $\mathcal D$, and $\mathbb E_{\mathcal D}[\bar f - \hat f] = \bar f - \bar f = 0$, so the middle term drops out.

---step---

**Name the terms.**
$$\mathbb E[(y-\hat f)^2] = \underbrace{\sigma^2}_{\text{noise}} + \underbrace{(f-\bar f)^2}_{\text{bias}^2} + \underbrace{\mathbb E_{\mathcal D}[(\hat f-\bar f)^2]}_{\text{variance}}.$$
Average this over test points $x$ to get the expected test MSE.

---step---

**Interpret it.** More capacity usually means lower bias and higher variance, which gives the classical U-shaped curve. More *data* lowers variance and leaves bias alone. Regularisation trades a little bias for a big drop in variance.

The decomposition is exact for squared error. For 0-1 loss and cross-entropy there are analogous but messier decompositions.

---step---

**Caveat for interviews.** Over-parameterised networks show **double descent**: test error can fall again past the interpolation threshold. Implicit regularisation (SGD, initialisation, architecture) keeps the effective variance low. Mention this, and you show you know where the textbook picture stops.
:::

:::solution{lang=python}
import numpy as np

def bias_variance(degree, n_datasets=300, n=20, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.05, 0.95, 50)
    f = lambda x: np.sin(2 * np.pi * x)
    preds = np.empty((n_datasets, xs.size))
    for k in range(n_datasets):
        x = rng.uniform(0, 1, n)
        y = f(x) + noise * rng.normal(size=n)
        preds[k] = np.polyval(np.polyfit(x, y, degree), xs)
    bias2 = ((preds.mean(axis=0) - f(xs)) ** 2).mean()
    var = preds.var(axis=0).mean()
    y_test = f(xs) + noise * rng.normal(size=(n_datasets, xs.size))
    err = ((y_test - preds) ** 2).mean()
    return bias2, var, noise ** 2, err
:::

:::explain
[L8-11] **Many training sets.** Each iteration draws a fresh set of 20 noisy samples of $\sin(2\pi x)$, fits a polynomial, and records its predictions on a fixed grid. Row $k$ of `preds` is $\hat f_{\mathcal D_k}$.

[L12-13] **Bias² and variance.** These are estimated directly from their definitions: the average model's squared error against the truth, and the spread around the average model.

[L14-15] **Test error, measured independently.** It uses fresh noisy targets. The tests check that $\text{bias}^2 + \text{variance} + \sigma^2$ matches this measured error, and that the error is U-shaped in the polynomial degree. Try printing `bias_variance(d)` for `d` in 0–7: bias collapses by degree 3, while variance explodes by degree 7.
:::

:::tests{lang=python}
import numpy as np

def test_decomposition_matches_measured_error():
    for d in (1, 3, 5):
        b, v, n, e = bias_variance(d)
        assert abs((b + v + n) - e) / e < 0.05

def test_bias_falls_with_capacity():
    assert bias_variance(3)[0] < 0.1 * bias_variance(1)[0]

def test_variance_rises_with_capacity():
    assert bias_variance(3)[1] < bias_variance(5)[1] < bias_variance(7)[1]

def test_u_shaped_test_error():
    e1, e3, e7 = (bias_variance(d)[3] for d in (1, 3, 7))
    assert e3 < e1 and e3 < e7
:::

:::hints
1. Write $y - \hat f = \epsilon + (f - \bar f) + (\bar f - \hat f)$, and expand the square.
2. For each cross term, ask: what is independent of what, and what has zero mean?
:::

:::interview
- Derive the decomposition. Which expectations are over what?
- How do more data, more regularisation and ensembling each move bias and variance?
- What is double descent, and how does it complicate the classical picture?
:::
::::

## "L2 regularisation" and "weight decay" are not the same thing

::::exercise{id=weight-decay-predict type=predict difficulty=3 time=10 runtime=browser udl="§9.1, §6.4" tags="weight-decay,regularisation,optimisers"}
# Adam + L2 versus AdamW

:::prereq
- **Plain SGD.** Adding $\tfrac\lambda2\|w\|^2$ to the loss contributes $\lambda w$ to the gradient, so each step does $w \leftarrow w - \eta\lambda w - \eta g$. That's the same as multiplying the weights by $(1-\eta\lambda)$, which is why the two names became interchangeable.
- **Adam** divides the whole gradient by $\sqrt{\hat v}$, including the $\lambda w$ part.
:::

:::task
Two weights both start at $1.0$. Their loss gradients are pure zero-mean noise: weight **A**'s has standard deviation $10$ and weight **B**'s has standard deviation $0.01$. Train for 3,000 steps with Adam ($\eta = 10^{-3}$), adding the L2 term $\tfrac\lambda2\|w\|^2$, with $\lambda = 0.1$, to the loss. Which weight ends up closer to zero?
:::

:::choices
- [ ] A: it has the larger gradients, so it moves more
- [x] B: the small-gradient weight is regularised much harder
- [ ] They shrink by about the same amount
- [ ] Neither shrinks noticeably
:::

:::experiment{lang=python}
import numpy as np

def run(decoupled, steps=3000, lr=1e-3, wd=0.1, seed=0):
    rng = np.random.default_rng(seed)
    w, scale = np.array([1.0, 1.0]), np.array([10.0, 0.01])   # weights A, B
    m, v, b1, b2 = np.zeros(2), np.zeros(2), 0.9, 0.999
    for t in range(1, steps + 1):
        g = scale * rng.normal(size=2)          # loss gradient: pure noise
        if not decoupled:
            g = g + wd * w                      # Adam + L2: decay enters the gradient
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
        if decoupled:
            w = w * (1 - lr * wd)               # AdamW: decay applied directly
        w = w - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return w

print("Adam + L2   final [A, B]:", np.round(run(decoupled=False), 3))
print("AdamW       final [A, B]:", np.round(run(decoupled=True), 3))
:::

:::tests{lang=python}
def test_adam_l2_spares_the_large_gradient_weight():
    a, b = run(decoupled=False)
    assert a > 0.85 and abs(b) < 0.05

def test_adamw_decays_both_alike():
    a, b = run(decoupled=True)
    assert 0.55 < a < 0.95 and 0.55 < b < 0.95 and abs(a - b) < 0.2
:::

:::explain
Adam's step is roughly $\eta\,\hat m/\sqrt{\hat v}$, which normalises *everything* in the gradient by that parameter's typical gradient size.

- **Weight A:** $\sqrt{\hat v}\approx 10$, so the decay term $\lambda w = 0.1$ is divided by 10 and makes up about 1% of the signal. It barely registers, and A stays near 1.
- **Weight B:** the noise is tiny, so $\lambda w$ *dominates* the gradient. Adam normalises it to a step of about $\eta$ every iteration. B marches to zero in about 1,000 steps.

So with Adam + L2, the effective regularisation strength is **inversely proportional to the gradient scale**. Parameters with large or frequent gradients are barely regularised, and rarely updated ones are crushed.

**AdamW** (Loshchilov & Hutter) *decouples* the decay: $w \leftarrow w(1-\eta\lambda)$ happens outside the adaptive normalisation. Every weight then shrinks by the same factor per step, about $e^{-\eta\lambda T} = e^{-0.3} \approx 0.74$ here, plus a small random walk from the noise. That's the behaviour people *mean* by weight decay, and it's why AdamW is the default for training transformers.

**For L1 versus L2:** L2 shrinks large weights most and rarely makes them exactly 0. L1's gradient has constant magnitude, so it drives small weights to exactly 0, which gives sparsity.
:::

:::interview
- Are L2 regularisation and weight decay equivalent? *(For SGD, yes, up to rescaling λ by the learning rate. For adaptive methods, no.)*
- Why is AdamW preferred for transformers?
- L1 versus L2: which gives sparsity, and why? *(L1. Its gradient doesn't shrink as the weight approaches 0.)*
- Should LayerNorm gains and biases be weight-decayed? *(Usually not, and they're typically excluded in a separate parameter group.)*
:::
::::

::::exercise{id=dropout-numpy type=implement difficulty=1 time=8 runtime=browser udl="§9.3" tags="regularisation,generalisation"}
# Inverted dropout

:::prereq
**Dropout** zeroes each activation independently with probability $p$ during training, so the network can't rely on any single unit.

**Inverted dropout** scales the survivors by $1/(1-p)$ *during training*, which keeps the expected activation unchanged: $\mathbb E[\tilde h] = (1-p)\cdot\frac{h}{1-p} = h$. At **evaluation** time the layer is the identity, so there's no extra work at inference. That's why every framework uses this variant.

The classic bug is forgetting to switch modes, i.e. leaving dropout on at test time. In PyTorch that means forgetting `model.eval()`.
:::

:::task
Implement `dropout(x, p, rng, train=True)` for $0 \le p < 1$. In training mode, drop each entry with probability `p` (using `rng.random`) and rescale the rest. In evaluation mode, or when `p == 0`, return `x` unchanged.
:::

:::starter{lang=python}
import numpy as np

def dropout(x, p, rng, train=True):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def dropout(x, p, rng, train=True):
    if {{ not train or p == 0 | p == 0 or not train | not train or p == 0.0 ? Dropout is only active while training. In evaluation it must be the identity, and with `p = 0` there's nothing to do. }}:
        return x
    keep = {{ rng.random(x.shape) >= p | rng.random(x.shape) > p | rng.random(size=x.shape) >= p | rng.uniform(size=x.shape) >= p ? Each entry survives with probability $1-p$: compare a uniform draw against `p`. }}
    return {{ x * keep / (1 - p) | x * keep / (1.0 - p) | np.where(keep, x / (1 - p), 0) | keep * x / (1 - p) ? Scale the survivors by $1/(1-p)$ so the expected activation is unchanged. That's what makes evaluation mode a plain identity. }}
:::

:::solution{lang=python}
import numpy as np

def dropout(x, p, rng, train=True):
    if not train or p == 0:
        return x
    keep = rng.random(x.shape) >= p
    return x * keep / (1 - p)
:::

:::explain
[L4-5] **Evaluation is the identity.** Because training already applied the rescaling, nothing needs to change at test time. Older "non-inverted" dropout instead multiplied by $1-p$ at test time, which is easy to forget.

[L6] **A Bernoulli mask.** `rng.random` draws uniform values in $[0, 1)$, so `>= p` keeps each entry with probability $1-p$. The same mask must be used in the backward pass: the gradient flows only through the kept units, scaled by $1/(1-p)$.

[L7] **Rescale the survivors.** The tests check that the mean is preserved on average, that about a fraction $p$ is zeroed, and that every survivor equals $x/(1-p)$ exactly.
:::

:::tests{lang=python}
import numpy as np

def test_eval_mode_is_identity():
    x = np.random.default_rng(0).normal(size=(4, 5))
    assert np.array_equal(dropout(x, 0.5, np.random.default_rng(1), train=False), x)

def test_drop_fraction_and_mean_preserved():
    x = np.ones((400, 500))
    y = dropout(x, 0.3, np.random.default_rng(2))
    assert abs((y == 0).mean() - 0.3) < 0.01 and abs(y.mean() - 1.0) < 0.01

def test_survivors_are_rescaled():
    x = np.random.default_rng(3).normal(size=1000) + 5
    y = dropout(x, 0.25, np.random.default_rng(4))
    kept = y != 0
    assert np.allclose(y[kept], x[kept] / 0.75)

def test_p_zero_is_identity():
    x = np.arange(6.0)
    assert np.array_equal(dropout(x, 0.0, np.random.default_rng(5)), x)
:::

:::hints
1. A boolean mask times an array zeroes the `False` entries.
2. Without the rescaling, what is the expected value of a dropped-out activation?
:::

:::interview
- Why rescale during training rather than at test time?
- What happens if you forget `model.eval()`? *(The predictions are noisy, and on average slightly wrong where the network is nonlinear.)*
- Dropout as an ensemble: what is it approximately averaging over? MC dropout for uncertainty?
- Why is dropout rarely used in large pre-trained transformers today?
:::
::::
