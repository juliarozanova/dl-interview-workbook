---
id: ch20-autoregressive
summary: Two ways to get an exact likelihood. Autoregressive models factor p(x) into conditionals, enforced by masks; normalising flows push a simple density through an invertible map. Build MADE's masks, derive the change-of-variables formula, and implement an affine coupling layer with an exact inverse and log-determinant.
udl: "Ch. 16 — Normalizing flows"
---

## Exact likelihoods

VAEs bound the likelihood (Chapter 17), and GANs don't have one (Chapter 18). Two families compute $\log p(x)$ **exactly**.

**Autoregressive models** use the chain rule of probability, $p(x) = \prod_{d=1}^D p(x_d \mid x_{<d})$. One network outputs every conditional at once, provided output $d$ can't see inputs $\ge d$. That's enforced with **masks**: causal attention in GPT (Chapter 14), masked convolutions in PixelCNN, and masked weight matrices in **MADE**. Training is parallel. Sampling is sequential, one dimension at a time.

**Normalising flows** use an invertible $f$ that maps data $x$ to a simple latent $z = f(x)$. The density follows by the **change of variables**,
$$\log p_x(x) = \log p_z(f(x)) + \log\left|\det \frac{\partial f}{\partial x}\right| ,$$
so the design challenge is making $f$ expressive while keeping both its inverse and its log-determinant cheap. **Coupling layers** do exactly that.

::::exercise{id=made-masks type=implement difficulty=3 time=20 runtime=browser udl="—" tags="autoregressive,masking"}
# MADE: autoregressive masks for an MLP

:::prereq
**Degrees.** Give every unit an integer **degree**:

- input $d$ has degree $d$ (1-based);
- hidden units get random degrees in $\{1, \dots, D-1\}$;
- output $d$ has degree $d$.

**Connection rules**, for weights stored `(in, out)` as in this workbook:

- input or hidden → hidden: connect if $\deg(\text{out}) \ge \deg(\text{in})$;
- hidden → output: connect if $\deg(\text{out}) > \deg(\text{in})$, **strictly**.

Then any path from input $j$ to output $d$ has non-decreasing degrees and ends with a strict step up, so $j < d$. Output $d$ depends only on $x_{<d}$, and output 1 depends on nothing (it models $p(x_1)$).
:::

:::task
Implement `made_masks(D, hidden_sizes, seed=0)`, returning a list of 0/1 masks, one per weight matrix, with shapes `(D, h1), (h1, h2), …, (h_last, D)`.
:::

:::starter{lang=python}
import numpy as np

def made_masks(D, hidden_sizes, seed=0):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def made_masks(D, hidden_sizes, seed=0):
    rng = np.random.default_rng(seed)
    degrees = [np.arange(1, D + 1)]
    for h in hidden_sizes:
        degrees.append({{ rng.integers(1, D, size=h) | rng.integers(1, D, h) | rng.integers(low=1, high=D, size=h) ? Hidden degrees are uniform on $\{1, \dots, D-1\}$. A hidden unit of degree $D$ could never feed any output, and `integers` excludes the upper bound. }})
    masks = [{{ (degrees[l][:, None] <= degrees[l + 1][None, :]).astype(float) | (degrees[l][:, None] <= degrees[l + 1]).astype(float) | (degrees[l + 1][None, :] >= degrees[l][:, None]).astype(float) ? Rows are inputs and columns are outputs. Connect when the output's degree is at least the input's. }} for l in range(len(hidden_sizes))]
    masks.append({{ (degrees[-1][:, None] < degrees[0][None, :]).astype(float) | (degrees[-1][:, None] < degrees[0]).astype(float) | (degrees[0][None, :] > degrees[-1][:, None]).astype(float) ? The output layer uses a **strict** inequality, so output $d$ never sees input $d$ itself. }})
    return masks
:::

:::solution{lang=python}
import numpy as np

def made_masks(D, hidden_sizes, seed=0):
    rng = np.random.default_rng(seed)
    degrees = [np.arange(1, D + 1)]
    for h in hidden_sizes:
        degrees.append(rng.integers(1, D, size=h))
    masks = [(degrees[l][:, None] <= degrees[l + 1][None, :]).astype(float) for l in range(len(hidden_sizes))]
    masks.append((degrees[-1][:, None] < degrees[0][None, :]).astype(float))
    return masks
:::

:::explain
[L4-7] **Degrees per layer.** The inputs and outputs share degrees $1..D$, and hidden units are random so different masks give different orderings of dependence. MADE trains over many masks as a kind of ensemble.

[L8] **Hidden masks by broadcasting.** `(in, 1) <= (1, out)` builds the full connectivity matrix in one comparison. That's Chapter 01's outer-comparison trick.

[L9] **The strict output mask.** The single `<` is what removes the diagonal. The tests check the defining property with an actual Jacobian: a masked random network's output $d$ has exactly zero derivative with respect to every input $j \ge d$.

**Compare with GPT.** A causal attention mask is the same idea for sequences. Position $t$ predicts token $t+1$ from tokens $\le t$, enforced by the triangular mask from Chapter 13.
:::

:::tests{lang=python}
import numpy as np

def _net(masks, seed=1):
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(size=m.shape) * m for m in masks]
    def f(x):
        h = x
        for W in Ws[:-1]:
            h = np.tanh(h @ W)
        return h @ Ws[-1]
    return f

def test_shapes():
    ms = made_masks(5, [16, 12])
    assert [m.shape for m in ms] == [(5, 16), (16, 12), (12, 5)]

def test_jacobian_is_strictly_lower_triangular():
    D = 6
    f, x, eps = _net(made_masks(D, [32, 32])), np.random.default_rng(2).normal(size=D), 1e-6
    J = np.zeros((D, D))
    for j in range(D):
        dx = np.zeros(D)
        dx[j] = eps
        J[:, j] = (f(x + dx) - f(x - dx)) / (2 * eps)
    assert np.allclose(np.triu(J), 0, atol=1e-9)
    assert np.abs(J[-1, :-1]).sum() > 0
:::

:::hints
1. Store each layer's degrees, and build each mask with one broadcast comparison.
2. Only the final mask uses a strict inequality.
:::

:::interview
- How does MADE make one network output all the autoregressive conditionals?
- Why is autoregressive sampling slow but training fast? *(Teacher forcing gives every conditional in parallel; sampling needs $D$ sequential passes.)*
- Connect MADE, PixelCNN's masked convolutions, and GPT's causal mask.
:::
::::

::::exercise{id=change-of-variables-derive type=derive difficulty=2 time=8 runtime=browser udl="§16.1" tags="autoregressive,normalising-flows"}
# The change-of-variables formula

:::prereq
- **Probability is conserved.** If $z = f(x)$ with $f$ invertible, a small region around $x$ of volume $dV_x$ maps to a region of volume $dV_z = |\det J_f(x)|\,dV_x$, and both contain the same probability mass.
- **1-D intuition.** If $z = a x + b$, then stretching by $|a|$ dilutes the density by $|a|$.
- **The determinant of a triangular matrix** is the product of its diagonal.
:::

:::task
Derive $p_x(x) = p_z(f(x))\,\left|\det \partial f/\partial x\right|$, write the log-likelihood for a composition of flows, and say why triangular Jacobians make flows practical.
:::

:::scaffold
**1. Conservation of mass.** $p_x(x)\,dV_x = p_z(z)\,dV_z$ and $dV_z = |\det J_f|\,dV_x$, so $p_x(x) = p_z(f(x))\,\cdot$ {{ |det J_f(x)| | |\det J_f(x)| | |det J| | |\det J| | |det(∂f/∂x)| | re:^\|\\?det.*\|$ ? Volumes scale by the absolute Jacobian determinant. The density must scale inversely to keep the mass fixed. }}

**2. 1-D check.** For $z = ax + b$ with $z\sim\mathcal N(0,1)$, the density is $p_x(x) = \mathcal N(ax+b;\,0,1)\,\cdot$ {{ |a| | abs(a) | \|a\| ? The Jacobian of a 1-D affine map is just the slope. }}

**3. Compositions.** For $f = f_K\circ\dots\circ f_1$ with intermediate values $h_k$, $\log p_x(x) = \log p_z(z) + \sum_k$ {{ log|det J_{f_k}| | \log|\det J_{f_k}| | log|det J_k| | \log|\det J_k| | re:^\\?log\|\\?det.*\|$ ? Determinants of a product multiply, so their logs add, one term per layer. }}

**4. Cost.** For a general $D\times D$ Jacobian, the determinant costs $O(D^3)$. If $J$ is triangular, it's the product of the {{ diagonal | diagonal entries | diagonal elements | re:diag ? Triangular matrices have their eigenvalues on the diagonal. }}, which is $O(D)$.
:::

:::derivation
**Conserve mass.** $f$ maps an infinitesimal box around $x$ to a parallelepiped around $z$ whose volume is $|\det J_f(x)|$ times bigger. Equal mass in both gives
$$p_x(x) = p_z\big(f(x)\big)\,\Big|\det \frac{\partial f}{\partial x}\Big| .$$

---step---

**Take logs and compose.** For $f = f_K\circ\dots\circ f_1$, the chain rule makes the Jacobian a product, and $\det$ of a product is the product of $\det$s:
$$\log p_x(x) = \log p_z(z) + \sum_{k=1}^K \log\big|\det J_{f_k}(h_{k-1})\big| .$$
Training maximises this exactly. There's no bound and no adversary.

---step---

**Make the determinant cheap.** A dense $\det$ costs $O(D^3)$. **Coupling layers** split $x = (x_a, x_b)$ and set $y_a = x_a$, $y_b = x_b \odot e^{s(x_a)} + t(x_a)$. The Jacobian is block-triangular, with diagonal $(1, \dots, 1, e^{s})$, so $\log|\det| = \sum s(x_a)$. The functions $s$ and $t$ can be arbitrary networks: they never need inverting.

---step---

**The trade-offs.** Flows need invertible, dimension-preserving maps, which limits architectures. Autoregressive flows (MAF and IAF) trade fast density evaluation against fast sampling. Continuous flows and flow matching avoid the coupling constraint, and connect back to diffusion (Chapter 19).
:::

:::solution{lang=python}
import numpy as np

def std_normal_logpdf(z):
    return -0.5 * (z ** 2 + np.log(2 * np.pi))

def affine_flow_logpdf(x, a, b):
    return std_normal_logpdf(a * x + b) + np.log(np.abs(a))
:::

:::explain
[L3-4] **The base density.** A standard normal, evaluated in log space.

[L6-7] **The change of variables in 1-D.** It's the base density at $z = ax+b$, plus $\log|a|$. The tests check that it integrates to 1 numerically, and that it equals the Gaussian $\mathcal N(-b/a,\,1/a^2)$ it should represent. Drop the $\log|a|$ term and the integral comes out as $1/|a|$.
:::

:::tests{lang=python}
import numpy as np

def test_integrates_to_one():
    xs = np.linspace(-30, 30, 200001)
    for a, b in [(2.0, 1.0), (0.3, -2.0), (-1.5, 0.5)]:
        p = np.exp(affine_flow_logpdf(xs, a, b))
        assert abs(((p[1:] + p[:-1]) / 2 * np.diff(xs)).sum() - 1) < 1e-6

def test_equals_the_implied_gaussian():
    a, b, x = 2.0, 1.0, np.linspace(-3, 3, 7)
    mean, var = -b / a, 1 / a ** 2
    ref = -0.5 * ((x - mean) ** 2 / var + np.log(2 * np.pi * var))
    assert np.allclose(affine_flow_logpdf(x, a, b), ref)
:::

:::hints
1. Start from "equal probability mass in corresponding small regions".
2. Logs turn the product of determinants into a sum.
:::

:::interview
- Derive the change-of-variables formula. Why the absolute value of the determinant?
- What makes a flow layer practical? *(A cheap inverse and a cheap log-determinant.)*
- Flows versus VAEs versus diffusion: what does each give up?
:::
::::

::::exercise{id=affine-coupling type=implement difficulty=2 time=15 runtime=browser udl="§16.3" tags="normalising-flows,autoregressive"}
# An affine coupling layer

:::prereq
For even $D$, split $x = (x_a, x_b)$ in half. With any functions $s(\cdot)$ and $t(\cdot)$ of $x_a$:
$$y_a = x_a,\qquad y_b = x_b\odot e^{s(x_a)} + t(x_a),\qquad \log|\det J| = \textstyle\sum s(x_a).$$
**Inverse.** $x_a = y_a$, then $x_b = (y_b - t(y_a))\odot e^{-s(y_a)}$. This only works because $x_a$ passes through unchanged, so $s$ and $t$ can be recomputed from $y_a$.
:::

:::task
Implement `coupling_forward(x, s_fn, t_fn)`, returning `(y, logdet)`, and `coupling_inverse(y, s_fn, t_fn)`, returning `x`. `x` has shape `(B, D)`, and `s_fn` and `t_fn` map `(B, D/2) → (B, D/2)`. `logdet` has shape `(B,)`.
:::

:::starter{lang=python}
import numpy as np

def coupling_forward(x, s_fn, t_fn):
    raise NotImplementedError

def coupling_inverse(y, s_fn, t_fn):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def coupling_forward(x, s_fn, t_fn):
    d = x.shape[1] // 2
    xa, xb = x[:, :d], x[:, d:]
    s, t = s_fn(xa), t_fn(xa)
    yb = {{ xb * np.exp(s) + t | np.exp(s) * xb + t | t + xb * np.exp(s) ? Scale the second half elementwise by $e^{s}$, which is always positive and so invertible, then shift it by $t$. }}
    return np.concatenate([xa, yb], axis=1), {{ s.sum(axis=1) | s.sum(1) | np.sum(s, axis=1) | s.sum(axis=-1) ? The Jacobian is triangular with diagonal $(1, \dots, 1, e^{s})$. Its log-determinant is the sum of the $s$ values, per example. }}

def coupling_inverse(y, s_fn, t_fn):
    d = y.shape[1] // 2
    ya, yb = y[:, :d], y[:, d:]
    xb = {{ (yb - t_fn(ya)) * np.exp(-s_fn(ya)) | (yb - t_fn(ya)) / np.exp(s_fn(ya)) | np.exp(-s_fn(ya)) * (yb - t_fn(ya)) ? Undo the shift, then the scale. $y_a = x_a$, so `s_fn` and `t_fn` see the same input as in the forward pass. }}
    return np.concatenate([ya, xb], axis=1)
:::

:::solution{lang=python}
import numpy as np

def coupling_forward(x, s_fn, t_fn):
    d = x.shape[1] // 2
    xa, xb = x[:, :d], x[:, d:]
    s, t = s_fn(xa), t_fn(xa)
    yb = xb * np.exp(s) + t
    return np.concatenate([xa, yb], axis=1), s.sum(axis=1)

def coupling_inverse(y, s_fn, t_fn):
    d = y.shape[1] // 2
    ya, yb = y[:, :d], y[:, d:]
    xb = (yb - t_fn(ya)) * np.exp(-s_fn(ya))
    return np.concatenate([ya, xb], axis=1)
:::

:::explain
[L5-8] **Forward.** `s_fn` and `t_fn` can be any networks: they're never inverted. The scale is $e^{s}$, which is always positive, so the map is invertible for any output of `s_fn`.

[L10-13] **Inverse.** It needs exactly the same `s_fn(ya)` and `t_fn(ya)`. That's why half the dimensions must pass through unchanged. Stack layers that alternate which half is transformed (RealNVP and Glow use permutations or 1×1 convolutions), and every dimension gets updated.

The tests check three things: that inverting recovers the input to machine precision; that `logdet` equals $\log|\det J|$ of a numerically computed Jacobian; and that the inverse also works the other way round.
:::

:::tests{lang=python}
import numpy as np

def _fns(seed=0, d=3):
    rng = np.random.default_rng(seed)
    Ws, Wt = rng.normal(size=(d, d)) * 0.5, rng.normal(size=(d, d))
    return (lambda a: np.tanh(a @ Ws)), (lambda a: a @ Wt + 1.0)

def test_inverse_recovers_input():
    s_fn, t_fn = _fns()
    x = np.random.default_rng(1).normal(size=(8, 6))
    y, _ = coupling_forward(x, s_fn, t_fn)
    assert np.allclose(coupling_inverse(y, s_fn, t_fn), x, atol=1e-12)
    assert np.allclose(coupling_forward(coupling_inverse(x, s_fn, t_fn), s_fn, t_fn)[0], x, atol=1e-12)

def test_logdet_matches_numerical_jacobian():
    s_fn, t_fn = _fns(2)
    x, eps = np.random.default_rng(3).normal(size=(1, 6)), 1e-6
    _, logdet = coupling_forward(x, s_fn, t_fn)
    J = np.zeros((6, 6))
    for j in range(6):
        dx = np.zeros((1, 6))
        dx[0, j] = eps
        J[:, j] = (coupling_forward(x + dx, s_fn, t_fn)[0] - coupling_forward(x - dx, s_fn, t_fn)[0])[0] / (2 * eps)
    assert logdet.shape == (1,) and np.isclose(logdet[0], np.log(abs(np.linalg.det(J))), atol=1e-6)
:::

:::hints
1. The first half passes through untouched, and only the second half changes.
2. In the inverse, undo the operations in reverse order: subtract the shift first, then divide out the scale.
:::

:::interview
- Why is the coupling layer's Jacobian triangular, and what's its log-determinant?
- Why don't `s` and `t` need to be invertible?
- RealNVP versus Glow versus MAF: what does each add?
:::
::::
