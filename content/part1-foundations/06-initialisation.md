---
id: ch06-initialisation
summary: Pick the wrong weight scale and a deep network's signal shrinks to nothing or blows up before training even starts. Predict what happens to 50 layers of activations, derive He initialisation from one variance argument, and implement Xavier, He and orthogonal initialisers.
udl: "Ch. 7 — Gradients and initialisation"
---

## Keep the signal alive

At initialisation, every layer multiplies the variance of its input by some factor $g$. After $L$ layers the variance has been scaled by $g^L$. With $L = 50$, even $g = 0.9$ shrinks it by $200\times$, and $g = 1.1$ grows it by $117\times$. The backward pass multiplies gradients by similar factors, so the same mistake gives **vanishing** or **exploding gradients**.

Initialisation schemes are simply the choices of weight variance that make $g \approx 1$ for a given activation function:

| Scheme | Weight variance | Designed for |
|---|---|---|
| LeCun | $1/D_\text{in}$ | linear / SELU |
| Xavier (Glorot) | $2/(D_\text{in} + D_\text{out})$ | tanh, linear (balances forward and backward) |
| He (Kaiming) | $2/D_\text{in}$ | ReLU (the factor 2 makes up for ReLU zeroing half its inputs) |
| Orthogonal | preserves norms exactly | deep linear networks, RNNs |

::::exercise{id=activation-scale-predict type=predict difficulty=1 time=6 runtime=browser udl="§7.5" tags="initialisation,vanishing-gradients"}
# Fifty ReLU layers, small weights

:::task
A 50-layer ReLU MLP has width $D = 128$. Weights are drawn from $\mathcal N(0, 1/D)$ (LeCun scaling, with no correction for ReLU) and biases are zero. Feed in standard-normal inputs. How does the standard deviation of the activations at layer 50 compare with layer 1?
:::

:::choices
- [ ] About the same: $1/D$ is the "variance-preserving" choice
- [x] Smaller by a factor of about $10^{7}$
- [ ] Larger by a factor of about $10^{7}$
- [ ] It becomes NaN
:::

:::experiment{lang=python}
import numpy as np

def activation_stds(sigma2, depth=50, D=128, B=128, seed=0):
    """Std of the post-ReLU activations after each layer, weights ~ N(0, sigma2)."""
    rng = np.random.default_rng(seed)
    h, stds = rng.normal(size=(B, D)), []
    for _ in range(depth):
        W = rng.normal(0.0, np.sqrt(sigma2), size=(D, D))
        h = np.maximum(h @ W, 0)
        stds.append(h.std())
    return stds

D = 128
for name, s2 in [("1/D (LeCun)", 1 / D), ("2/D (He)   ", 2 / D), ("4/D       ", 4 / D)]:
    s = activation_stds(s2, D=D)
    print(f"{name}  layer 1: {s[0]:.3g}   layer 10: {s[9]:.3g}   layer 50: {s[49]:.3g}   ratio 50/1: {s[49] / s[0]:.2g}")
:::

:::tests{lang=python}
def test_lecun_vanishes():
    s = activation_stds(1 / 128)
    assert s[49] / s[0] < 1e-5

def test_he_is_stable():
    s = activation_stds(2 / 128)
    assert 0.3 < s[49] / s[0] < 3

def test_double_he_explodes():
    s = activation_stds(4 / 128)
    assert s[49] / s[0] > 1e5
:::

:::explain
Each layer computes $z = Wh$, and then $h' = \mathrm{ReLU}(z)$. With $D$ independent terms per output,
$$\mathrm{Var}(z) = D\,\sigma_w^2\,\mathbb E[h^2],\qquad \mathbb E[\mathrm{ReLU}(z)^2] = \tfrac12\mathrm{Var}(z).$$
So the mean square of the activations is multiplied by $g = D\sigma_w^2/2$ per layer.

- **$\sigma_w^2 = 1/D$:** $g = 1/2$. After 49 more layers the mean square has shrunk by $2^{-49}$, so the standard deviation shrinks by $2^{-24.5} \approx 4\times10^{-8}$. That's the $10^{-7}$ factor.
- **$\sigma_w^2 = 2/D$:** $g = 1$, so the scale is preserved. That's He initialisation (next exercise).
- **$\sigma_w^2 = 4/D$:** $g = 2$, and it explodes symmetrically.

$1/D$ *is* variance-preserving for a **linear** layer. The trap is forgetting that ReLU throws away half the signal. Nothing becomes NaN at depth 50 in float64: the numbers are just tiny. In float16 they would underflow to exactly 0.

The same factor $g$ applies to the **gradients** on the way back, so a network initialised like the first row barely trains, even though nothing looks broken.
:::

:::interview
- Why doesn't $1/D$ preserve variance for ReLU networks?
- Do residual connections or normalisation layers change this picture? *(Yes. Both stop the variance compounding layer after layer. See Chapter 10.)*
- How would you detect this in a real run? *(Log activation and gradient norms per layer at step 0.)*
:::
::::

::::exercise{id=he-init-derive type=derive difficulty=2 time=8 runtime=browser udl="§7.5" tags="initialisation,vanishing-gradients"}
# Derive He initialisation

:::prereq
- For independent $w$ and $h$, with $\mathbb E[w]=0$: $\mathrm{Var}(wh) = \mathbb E[w^2]\,\mathbb E[h^2] = \sigma_w^2\,\mathbb E[h^2]$. You need $\mathbb E[h^2]$, **not** $\mathrm{Var}(h)$, because ReLU outputs don't have zero mean.
- If $z$ is symmetric about 0, then $\mathrm{ReLU}(z)^2$ equals $z^2$ half the time and $0$ otherwise.
:::

:::task
For a ReLU layer $z^{(\ell+1)} = W h^{(\ell)}$ with $h^{(\ell)} = \mathrm{ReLU}(z^{(\ell)})$, fan-in $D$, and i.i.d. weights with mean 0 and variance $\sigma_w^2$: derive the $\sigma_w^2$ that keeps $\mathrm{Var}(z)$ constant from layer to layer. Then state Xavier's variance and say what it is for.
:::

:::scaffold
Write `σ_w^2` (or `\sigma_w^2`) for the weight variance and `D` for the fan-in.

**1. ReLU halves the second moment.** For symmetric $z$, $\mathbb E[\mathrm{ReLU}(z)^2] = c\cdot\mathrm{Var}(z)$ with $c =$ {{ 1/2 | 0.5 | \frac{1}{2} | \tfrac{1}{2} | ½ | \frac12 ? $\mathrm{ReLU}(z)^2 = z^2$ when $z > 0$, which happens half the time by symmetry, and $0$ otherwise. }}

**2. Per-layer gain.** $z_j$ sums $D$ independent terms, each with variance $\sigma_w^2\,\mathbb E[h^2]$. So $\mathrm{Var}(z^{(\ell+1)}) = g\cdot\mathrm{Var}(z^{(\ell)})$ with $g =$ {{ Dσ_w^2/2 | D\sigma_w^2/2 | Dσ^2/2 | D\sigma^2/2 | \frac{D\sigma_w^2}{2} | \frac{D\sigma^2}{2} | σ_w^2D/2 | D/2σ_w^2 | 0.5Dσ_w^2 ? Variances of independent terms add, giving a factor of $D$. Then substitute $\mathbb E[h^2] = \frac12\mathrm{Var}(z)$ from step 1. }}

**3. He.** Setting $g = 1$ gives $\sigma_w^2 =$ {{ 2/D | \frac{2}{D} | 2/fan_in | 2/D_{in} | 2/D_in | 2/n_{in} ? Solve $D\sigma_w^2/2 = 1$. }}

**4. Xavier.** For tanh or linear layers, averaging the forward condition ($D_\text{in}\sigma_w^2 = 1$) with the backward one ($D_\text{out}\sigma_w^2 = 1$) gives $\sigma_w^2 =$ {{ 2/(D_in + D_out) | 2/(D_{in} + D_{out}) | 2/(fan_in + fan_out) | 2/(n_in + n_out) | 2/(n_{in} + n_{out}) | re:^\\frac\{2\}\{d_\{?(in|\\text\{in\})\}?\+d_\{?(out|\\text\{out\})\}?\}$ ? The forward pass wants $1/D_\text{in}$ and the backward pass wants $1/D_\text{out}$. Glorot's compromise is the harmonic mean. }}
:::

:::derivation
**One output unit.** $z_j = \sum_{i=1}^{D} w_{ji} h_i$, with the weights independent of the inputs and of each other, and mean zero. Variances add:
$$\mathrm{Var}(z_j) = \sum_i \mathrm{Var}(w_{ji}h_i) = D\,\sigma_w^2\,\mathbb E[h^2].$$
It's $\mathbb E[h^2]$, not $\mathrm{Var}(h)$, because $\mathrm{Var}(wh) = \mathbb E[w^2h^2] - (\mathbb E[w]\mathbb E[h])^2$ and $\mathbb E[w] = 0$.

---step---

**Through the ReLU.** If $z$ is symmetric about zero, half its mass is positive, so $\mathbb E[\mathrm{ReLU}(z)^2] = \tfrac12\mathbb E[z^2] = \tfrac12\mathrm{Var}(z)$.

---step---

**Combine.** $\mathrm{Var}(z^{(\ell+1)}) = \frac{D\sigma_w^2}{2}\,\mathrm{Var}(z^{(\ell)})$. The variance is constant across layers if and only if $\sigma_w^2 = 2/D$, which is **He initialisation**. Drawn uniformly, that's $U(-a, a)$ with $a = \sqrt{6/D}$, because $U(-a,a)$ has variance $a^2/3$.

---step---

**The backward pass.** A similar argument for the gradients gives the condition $D_\text{out}\sigma_w^2/2 = 1$. For tanh-like activations, where the factor is 1 rather than ½, Xavier compromises between the two conditions: $\sigma_w^2 = 2/(D_\text{in} + D_\text{out})$.

---step---

**Say what it doesn't fix.** This is a statement about **initialisation**: once training moves the weights, the argument no longer holds. Normalisation layers and residual connections keep the scale under control *during* training, which is why very deep networks need them as well as a good init.
:::

:::solution{lang=python}
import numpy as np

def he_std(fan_in):
    return np.sqrt(2.0 / fan_in)

def preactivation_stds(depth, D, std, B=512, seed=0):
    rng = np.random.default_rng(seed)
    z, out = rng.normal(size=(B, D)), []
    for _ in range(depth):
        z = np.maximum(z, 0) @ rng.normal(0.0, std, size=(D, D))
        out.append(z.std())
    return out
:::

:::explain
[L3-4] **He's standard deviation.** It's $\sqrt{2/D}$. `torch.nn.init.kaiming_normal_` computes the same thing, with a configurable gain for leaky ReLU.

[L6-12] **Measure it.** This pushes *pre*-activations through ReLU and a random layer repeatedly, which is exactly the quantity the derivation tracks. With `std = he_std(D)`, the printed values stay near the input's scale for any depth.
:::

:::tests{lang=python}
import numpy as np

def test_he_std():
    assert np.isclose(he_std(512), np.sqrt(2 / 512))

def test_he_preserves_preactivation_scale():
    s = preactivation_stds(50, 256, he_std(256))
    assert 0.3 < s[-1] / s[0] < 3

def test_without_the_factor_two_it_vanishes():
    s = preactivation_stds(50, 256, np.sqrt(1 / 256))
    assert s[-1] / s[0] < 1e-5
:::

:::hints
1. Start from one output unit: it's a sum of $D$ products $w_{ji}h_i$.
2. Where does the ReLU enter? Only through $\mathbb E[h^2]$.
:::

:::interview
- Derive He initialisation in under two minutes. Where does the 2 come from?
- What gain would you use for leaky ReLU with slope $a$? *($2/(1 + a^2)$.)*
- Why does PyTorch's default `nn.Linear` init use $U(-1/\sqrt{D}, 1/\sqrt{D})$, and is that He? *(No: its variance is $1/(3D)$, so it relies on normalisation and residuals to train deep networks.)*
:::
::::

::::exercise{id=init-schemes type=implement difficulty=2 time=15 runtime=browser udl="§7.5" tags="initialisation"}
# Xavier, He and orthogonal initialisers

:::prereq
- **Uniform variance.** $U(-a, a)$ has variance $a^2/3$. So Xavier-uniform uses $a = \sqrt{6/(D_\text{in}+D_\text{out})}$.
- **Orthogonal init.** Take the QR decomposition of a Gaussian matrix, $A = QR$. $Q$ has orthonormal columns. Multiply each column by the sign of $R$'s diagonal entry so that $Q$ is uniformly distributed (otherwise QR's sign convention biases it). If you want more rows than columns, $Q$ is already the right shape; otherwise transpose.
- Weights here use this workbook's `(fan_in, fan_out)` convention.
:::

:::task
Implement `xavier_uniform(fan_in, fan_out, rng)`, `he_normal(fan_in, fan_out, rng)` and `orthogonal(n, m, rng)`. For `orthogonal`, the columns must be orthonormal when `n ≥ m` and the rows when `n < m`. Use only the `rng` passed in (a `np.random.Generator`) for randomness.
:::

:::starter{lang=python}
import numpy as np

def xavier_uniform(fan_in, fan_out, rng):
    raise NotImplementedError

def he_normal(fan_in, fan_out, rng):
    raise NotImplementedError

def orthogonal(n, m, rng):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def xavier_uniform(fan_in, fan_out, rng):
    a = {{ np.sqrt(6 / (fan_in + fan_out)) | (6 / (fan_in + fan_out)) ** 0.5 | np.sqrt(6.0 / (fan_in + fan_out)) ? The target variance is $2/(D_\text{in}+D_\text{out})$, and $U(-a,a)$ has variance $a^2/3$. Solve for $a$. }}
    return rng.uniform(-a, a, size=(fan_in, fan_out))

def he_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, {{ np.sqrt(2 / fan_in) | (2 / fan_in) ** 0.5 | np.sqrt(2.0 / fan_in) ? `rng.normal` takes a **standard deviation**, not a variance. He's variance is $2/D_\text{in}$. }}, size=(fan_in, fan_out))

def orthogonal(n, m, rng):
    A = rng.normal(size=(max(n, m), min(n, m)))
    Q, R = {{ np.linalg.qr(A) ? The QR decomposition: $Q$ has orthonormal columns and $R$ is upper triangular. }}
    Q = Q * {{ np.sign(np.diag(R)) | np.sign(R.diagonal()) ? QR fixes signs by convention, which biases the distribution. Flipping each column by the sign of $R_{ii}$ makes $Q$ uniformly (Haar) distributed. }}
    return {{ Q if n >= m else Q.T | Q.T if n < m else Q ? `Q` is tall: `(max, min)`. For a wide `(n, m)` with `n < m`, transpose it, which gives orthonormal *rows*. }}
:::

:::solution{lang=python}
import numpy as np

def xavier_uniform(fan_in, fan_out, rng):
    a = np.sqrt(6 / (fan_in + fan_out))
    return rng.uniform(-a, a, size=(fan_in, fan_out))

def he_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, np.sqrt(2 / fan_in), size=(fan_in, fan_out))

def orthogonal(n, m, rng):
    A = rng.normal(size=(max(n, m), min(n, m)))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    return Q if n >= m else Q.T
:::

:::explain
[L4] **Bound from variance.** $a^2/3 = 2/(D_\text{in}+D_\text{out})$ gives $a = \sqrt{6/(D_\text{in}+D_\text{out})}$. That's where the famous 6 comes from.

[L8] **Standard deviation, not variance.** Passing `2 / fan_in` as the scale is the most common bug: it makes the weights $\sqrt{2/D}$ times too small, which brings back the vanishing activations from the first exercise.

[L11-12] **QR of a Gaussian.** `A` is tall, so `Q` has orthonormal columns.

[L13] **Sign correction.** NumPy's QR makes the choice of signs deterministic, so without this line the distribution over orthogonal matrices isn't uniform. It's a subtle point, and rarely asked, but it's why library implementations include the line.

[L14] **Wide matrices.** Transposing a tall matrix with orthonormal columns gives a wide one with orthonormal rows. Either way, $\|Wx\| = \|x\|$ along the preserved directions, which is why orthogonal init suits RNNs, where the same matrix is applied many times.
:::

:::tests{lang=python}
import numpy as np

def test_xavier_bounds_and_variance():
    W = xavier_uniform(600, 400, np.random.default_rng(0))
    a = np.sqrt(6 / 1000)
    assert W.shape == (600, 400) and np.abs(W).max() <= a
    assert abs(W.var() / (2 / 1000) - 1) < 0.05

def test_he_variance():
    W = he_normal(500, 300, np.random.default_rng(1))
    assert W.shape == (500, 300) and abs(W.var() / (2 / 500) - 1) < 0.05

def test_orthogonal_tall_and_wide():
    Wt = orthogonal(6, 4, np.random.default_rng(2))
    Ww = orthogonal(4, 6, np.random.default_rng(3))
    assert Wt.shape == (6, 4) and np.allclose(Wt.T @ Wt, np.eye(4))
    assert Ww.shape == (4, 6) and np.allclose(Ww @ Ww.T, np.eye(4))

def test_orthogonal_preserves_norms():
    W, x = orthogonal(8, 8, np.random.default_rng(4)), np.random.default_rng(5).normal(size=8)
    assert np.isclose(np.linalg.norm(x @ W), np.linalg.norm(x))

def test_uses_the_given_rng():
    assert np.array_equal(he_normal(4, 3, np.random.default_rng(7)), he_normal(4, 3, np.random.default_rng(7)))
:::

:::hints
1. Convert each target *variance* into the parameter the sampler wants: a bound for `uniform`, a standard deviation for `normal`.
2. For `orthogonal`, always build a tall matrix first, and decide at the end whether to transpose.
:::

:::interview
- Why does orthogonal init help RNNs and very deep linear networks? *(All singular values are 1, so repeated multiplication neither shrinks nor grows the signal.)*
- Should biases be initialised to zero? When might you not? *(For example, a positive forget-gate bias in LSTMs, or a bias that sets the output prior on class imbalance.)*
- What goes wrong if you initialise all weights to the same constant? *(Symmetry: every unit in a layer computes the same thing and gets the same gradient forever.)*
:::
::::
