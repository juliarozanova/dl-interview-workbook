---
id: ch19-diffusion
summary: Diffusion models learn to undo noise one small step at a time. Derive the closed-form forward process, write the ε-prediction loss, and implement the DDPM sampler. It's tested against a data distribution whose optimal denoiser is known exactly. Then see what the sampler's noise term is really for.
udl: "Ch. 18 — Diffusion models"
---

## Noise in, data out

A diffusion model defines a **forward process** that gradually destroys data with Gaussian noise over $T$ steps,
$$q(x_t\mid x_{t-1}) = \mathcal N\!\big(\sqrt{\alpha_t}\,x_{t-1},\; (1-\alpha_t) I\big),\qquad \alpha_t = 1-\beta_t ,$$
until $x_T$ is essentially pure noise. It then **learns the reverse**: a network $\epsilon_\theta(x_t, t)$ predicts the noise in $x_t$, and a sampler uses that prediction to step from $x_T$ back to $x_0$.

Three facts make it work, and each is an exercise here:

1. **Closed-form forward process.** You can jump straight to any $x_t$ without simulating the chain (19.1).
2. **A simple loss.** Regress the noise: $\mathbb E\,\|\epsilon - \epsilon_\theta(x_t, t)\|^2$. That's equivalent to denoising score matching, because $\nabla_{x_t}\log q(x_t\mid x_0) = -\epsilon/\sqrt{1-\bar\alpha_t}$.
3. **An iterative sampler** that adds back a calibrated amount of fresh noise at each step (19.2–19.3).

UDL Chapter 18 treats diffusion as a hierarchical VAE with a *fixed* encoder, which connects it directly to Chapter 17's ELBO.

::::exercise{id=diffusion-forward-derive type=derive difficulty=2 time=8 runtime=browser udl="§18.2" tags="diffusion"}
# The forward process in closed form

:::prereq
- **Reparameterised step.** $x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t$, with $\epsilon_t \sim \mathcal N(0, I)$ independent across steps.
- **Sums of independent Gaussians.** $a\,\epsilon_1 + b\,\epsilon_2 \sim \mathcal N(0, (a^2 + b^2) I)$.
- **Notation.** $\bar\alpha_t = \prod_{s=1}^t \alpha_s$.
:::

:::task
Show that $q(x_t\mid x_0) = \mathcal N\!\big(\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)I\big)$. Then explain why this makes training cheap, and what the network is trained to predict.
:::

:::scaffold
Write `abar_t` or `\bar\alpha_t` for $\bar\alpha_t$, and `sqrt(…)` or `\sqrt{…}` for roots.

**1. Two steps.** Substituting $x_{t-1}$ into $x_t$, the noise terms are $\sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t$. Their combined variance is {{ 1 - α_tα_{t-1} | 1 - \alpha_t\alpha_{t-1} | 1 - α_t α_{t-1} | 1-alpha_t*alpha_{t-1} | re:^1-(α|\\alpha|alpha)_\{?t\}?\*?(α|\\alpha|alpha)_\{?t-1\}?$ ? Add the variances, $\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t)$, and simplify. }}

**2. Unroll to $x_0$.** By induction, $x_t = c\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ with $c =$ {{ sqrt(abar_t) | \sqrt{\bar\alpha_t} | sqrt(\bar\alpha_t) | √abar_t | √ᾱ_t | sqrt(ᾱ_t) ? Each step multiplies the signal by $\sqrt{\alpha_s}$. The product of the square roots is the square root of the product. }}

**3. Variance.** The total noise variance is {{ 1 - abar_t | 1 - \bar\alpha_t | 1 - ᾱ_t | 1-abar_t ? From step 1's pattern: the signal keeps a fraction $\bar\alpha_t$ of the variance, and noise fills the rest, so the total stays 1 for unit-variance data. }}

**4. What the network predicts.** Training samples $t$ and $\epsilon$, forms $x_t$ in one shot, and regresses the network's output onto the {{ noise | ε | \epsilon | epsilon | the noise ε | re:(noise|epsilon|ε|\\epsilon) ? Given $x_t$ and $t$, predicting $\epsilon$ is equivalent to predicting $x_0$, or the score, up to known scalings. Noise prediction works best in practice. }}.
:::

:::derivation
**Two steps.** Substitute $x_{t-1} = \sqrt{\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_{t-1}}\,\epsilon_{t-1}$:
$$x_t = \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t}_{\text{variance } 1-\alpha_t\alpha_{t-1}} .$$

---step---

**Merge the noise.** The two Gaussians are independent, so their sum is a single Gaussian with variance $\alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t\alpha_{t-1}$. The pattern is: the signal coefficient is the root of the product of the $\alpha$s, and the noise variance is one minus that product.

---step---

**Induction.**
$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I) \quad\Longrightarrow\quad q(x_t\mid x_0) = \mathcal N\big(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I\big).$$

---step---

**Why it matters.** Training needs no simulation. Pick a random $t$ and draw $\epsilon$, form $x_t$ in one line, and take a gradient step on $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$. Every timestep is trained in parallel from the same data point.

The schedule ($\beta_t$) is chosen so that $\bar\alpha_T \approx 0$, which makes $x_T$ indistinguishable from $\mathcal N(0, I)$. That's where sampling starts.
:::

:::solution{lang=python}
import numpy as np

def linear_schedule(T, beta_1=1e-4, beta_T=0.02):
    betas = np.linspace(beta_1, beta_T, T)
    return betas, np.cumprod(1 - betas)

def q_sample(x0, t, eps, abar):
    return np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps

def eps_loss(eps_model, x0, t, eps, abar):
    return ((eps - eps_model(q_sample(x0, t, eps, abar), t)) ** 2).mean()
:::

:::explain
[L3-5] **The schedule.** $\bar\alpha$ is a cumulative product. The original DDPM used $T = 1000$ with a linear $\beta$ from $10^{-4}$ to $0.02$. Cosine schedules spend more steps at low noise levels.

[L7-8] **One-shot noising.** This is the derivation's result in one line. The tests check it against actually simulating the chain step by step.

[L10-11] **The training objective.** It's a plain MSE on the noise. The tests also check that the *optimal* predictor for Gaussian data (Exercise 19.2) beats predicting zero.
:::

:::tests{lang=python}
import numpy as np

def test_matches_simulating_the_chain():
    betas, abar = linear_schedule(50)
    rng = np.random.default_rng(0)
    x0 = np.full(200000, 1.5)
    x = x0.copy()
    for b in betas:
        x = np.sqrt(1 - b) * x + np.sqrt(b) * rng.normal(size=x.shape)
    direct = q_sample(x0, 49, rng.normal(size=x0.shape), abar)
    assert abs(x.mean() - direct.mean()) < 0.01 and abs(x.std() - direct.std()) < 0.01
    assert abs(direct.mean() - np.sqrt(abar[49]) * 1.5) < 0.01

def test_final_step_is_nearly_pure_noise():
    _, abar = linear_schedule(1000)
    assert abar[-1] < 1e-4

def test_optimal_predictor_beats_zero():
    _, abar = linear_schedule(100)
    rng = np.random.default_rng(1)
    mu, s, t = 2.0, 0.5, 90  # enough noise that predicting it is informative
    x0, eps = mu + s * rng.normal(size=50000), rng.normal(size=50000)
    oracle = lambda xt, t: (xt - np.sqrt(abar[t]) * mu) * np.sqrt(1 - abar[t]) / (abar[t] * s ** 2 + 1 - abar[t])
    assert eps_loss(oracle, x0, t, eps, abar) < 0.5 * eps_loss(lambda xt, t: 0 * xt, x0, t, eps, abar)
:::

:::hints
1. Do two steps by hand, add the independent noise variances, and spot the pattern.
2. Scalars multiply: $\prod\sqrt{\alpha_s} = \sqrt{\prod\alpha_s}$.
:::

:::interview
- Derive $q(x_t\mid x_0)$. Why does it make training efficient?
- Why predict $\epsilon$ rather than $x_0$? How does ε-prediction relate to the score $\nabla\log p$?
- What does the noise schedule control? Linear versus cosine?
:::
::::

::::exercise{id=ddpm-sampler type=implement difficulty=3 time=20 runtime=browser udl="§18.4" tags="diffusion"}
# The DDPM sampler

:::prereq
**One reverse step** (Ho et al., 2020), given the network's noise estimate $\hat\epsilon = \epsilon_\theta(x_t, t)$:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\hat\epsilon\right) + \sigma_t z,\qquad z\sim\mathcal N(0, I),\ \sigma_t = \sqrt{\beta_t},$$
with **no noise added on the final step** ($t = 0$).

**A test with a known answer.** If the data are $\mathcal N(\mu, s^2)$ (1-D), the *optimal* noise predictor has a closed form:
$$\epsilon^\star(x_t, t) = \frac{(x_t - \sqrt{\bar\alpha_t}\,\mu)\sqrt{1-\bar\alpha_t}}{\bar\alpha_t s^2 + 1 - \bar\alpha_t}.$$
A correct sampler driven by $\epsilon^\star$ must reproduce $\mathcal N(\mu, s^2)$. No neural network is needed to test it.
:::

:::task
Implement `ddpm_sample(eps_fn, n, betas, rng)`, where `eps_fn(x, t)` returns the noise estimate. Start from $x_T \sim \mathcal N(0, I)$ with shape `(n,)`, loop $t = T-1, \dots, 0$ (zero-based), and return $x_0$. Use `rng.normal` for all randomness.
:::

:::starter{lang=python}
import numpy as np

def ddpm_sample(eps_fn, n, betas, rng):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def ddpm_sample(eps_fn, n, betas, rng):
    alphas = 1 - betas
    abar = {{ np.cumprod(alphas) | np.cumprod(1 - betas) ? $\bar\alpha_t$ is the running product of the $\alpha$s. }}
    x = rng.normal(size=n)
    for t in {{ reversed(range(len(betas))) | range(len(betas) - 1, -1, -1) ? Sampling runs the chain **backwards**, from the noisiest step down to 0. }}:
        eps = eps_fn(x, t)
        x = (x - {{ betas[t] / np.sqrt(1 - abar[t]) * eps | betas[t] * eps / np.sqrt(1 - abar[t]) ? Remove the predicted noise, scaled by $\beta_t/\sqrt{1-\bar\alpha_t}$. }}) / np.sqrt(alphas[t])
        if {{ t > 0 | t != 0 | t >= 1 ? The last step returns the mean: adding noise there would only blur the final sample. }}:
            x = x + {{ np.sqrt(betas[t]) * rng.normal(size=n) | np.sqrt(betas[t]) * rng.normal(size=x.shape) | rng.normal(size=n) * np.sqrt(betas[t]) ? Fresh Gaussian noise with variance $\beta_t$: this is the stochastic part of the reverse process. }}
    return x
:::

:::solution{lang=python}
import numpy as np

def ddpm_sample(eps_fn, n, betas, rng):
    alphas = 1 - betas
    abar = np.cumprod(alphas)
    x = rng.normal(size=n)
    for t in reversed(range(len(betas))):
        eps = eps_fn(x, t)
        x = (x - betas[t] / np.sqrt(1 - abar[t]) * eps) / np.sqrt(alphas[t])
        if t > 0:
            x = x + np.sqrt(betas[t]) * rng.normal(size=n)
    return x
:::

:::explain
[L4-6] **Start from pure noise.** This is valid because $\bar\alpha_T \approx 0$: the forward process forgets the data.

[L7-9] **The posterior mean.** Subtracting the scaled noise estimate and rescaling gives the mean of $p_\theta(x_{t-1}\mid x_t)$. This formula is what you get by plugging $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$ into the Gaussian posterior $q(x_{t-1}\mid x_t, x_0)$.

[L10-11] **Add calibrated noise, except at the end.** $\sigma_t^2 = \beta_t$ is one of two standard choices; $\tilde\beta_t$ is the other. The next exercise shows what happens if you leave this out.

**Cost.** Each sample needs $T$ network evaluations, often 1,000. That's why DDIM and other deterministic ODE samplers, distillation and consistency models exist: they cut it to between 1 and 50.
:::

:::tests{lang=python}
import numpy as np

def _oracle(mu, s, abar):
    return lambda x, t: (x - np.sqrt(abar[t]) * mu) * np.sqrt(1 - abar[t]) / (abar[t] * s ** 2 + 1 - abar[t])

def test_recovers_the_data_distribution():
    betas = np.linspace(1e-4, 0.1, 200)
    abar = np.cumprod(1 - betas)
    x = ddpm_sample(_oracle(2.0, 0.5, abar), 20000, betas, np.random.default_rng(0))
    assert x.shape == (20000,) and abs(x.mean() - 2.0) < 0.02 and abs(x.std() - 0.5) < 0.02

def test_a_different_target():
    betas = np.linspace(1e-4, 0.1, 200)
    abar = np.cumprod(1 - betas)
    x = ddpm_sample(_oracle(-1.0, 1.5, abar), 20000, betas, np.random.default_rng(1))
    assert abs(x.mean() + 1.0) < 0.05 and abs(x.std() - 1.5) < 0.05
:::

:::hints
1. Loop from the last index down to 0, and skip the noise on the final iteration.
2. If the mean is right but the spread is far too small, look at the noise term.
:::

:::interview
- Write one DDPM reverse step. Where does each term come from?
- Why does sampling need so many steps, and how do DDIM and distillation reduce them?
- Classifier-free guidance: what does it compute, and what does it trade? *(Fidelity to the prompt against diversity.)*
:::
::::

::::exercise{id=diffusion-noise-predict type=predict difficulty=2 time=6 runtime=browser udl="§18.4" tags="diffusion"}
# What is the sampler's noise for?

:::task
Take the sampler from Exercise 19.2, with the exact optimal denoiser for data $\mathcal N(2, 0.5^2)$, and **delete the noise term**, so every step just applies the posterior mean. What do the samples look like?
:::

:::choices
- [ ] The same distribution: the denoiser is exact, so the noise was only for exploration
- [x] Collapsed onto (almost) a single point near the mean
- [ ] Too wide: they keep the spread of the initial noise
- [ ] They diverge
:::

:::experiment{lang=python}
import numpy as np

betas = np.linspace(1e-4, 0.1, 200)
alphas, abar = 1 - betas, np.cumprod(1 - betas)
mu, s = 2.0, 0.5

def eps_star(x, t):
    return (x - np.sqrt(abar[t]) * mu) * np.sqrt(1 - abar[t]) / (abar[t] * s ** 2 + 1 - abar[t])

def sample(n, add_noise, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    for t in reversed(range(len(betas))):
        x = (x - betas[t] / np.sqrt(1 - abar[t]) * eps_star(x, t)) / np.sqrt(alphas[t])
        if add_noise and t > 0:
            x = x + np.sqrt(betas[t]) * rng.normal(size=n)
    return x

for add_noise in (True, False):
    x = sample(20000, add_noise)
    print(f"noise term {'on ' if add_noise else 'off'}:  mean {x.mean():.3f}   std {x.std():.4f}   (target 2.000, 0.5000)")
:::

:::tests{lang=python}
def test_with_noise_matches_target():
    x = sample(20000, True)
    assert abs(x.std() - 0.5) < 0.02

def test_without_noise_collapses():
    x = sample(20000, False)
    assert x.std() < 0.05 and abs(x.mean() - 2.0) < 0.05
:::

:::explain
Each reverse step's mean pulls $x_t$ towards the *expected* clean signal. Applying only means, 200 times over, behaves like gradient ascent on the density: every sample slides into the mode. Here the standard deviation collapses from 0.5 to about 0.001. The fresh noise $\sigma_t z$ is what keeps the *spread* of the distribution. The reverse process is a **stochastic** process, and its noise isn't optional.

**But deterministic samplers exist?** Yes. **DDIM**, and the probability-flow ODE, are deterministic, but they use a *different* update. It's derived so that the marginal distributions still match, mostly by keeping the predicted noise direction, rather than just dropping the noise term from DDPM. "DDPM minus the noise" isn't DDIM, and this exercise is exactly the mistake that shows why.

**Score-based view.** With $s_\theta \approx \nabla_x\log p_t(x) = -\hat\epsilon/\sqrt{1-\bar\alpha_t}$, DDPM is a form of annealed **Langevin dynamics**: a gradient step on log-density plus noise. Gradient steps alone find modes; adding the noise turns it into sampling.
:::

:::interview
- Why does the reverse process inject noise? What happens without it?
- DDPM versus DDIM: what does DDIM change, and why can it be deterministic?
- How is diffusion related to score matching and Langevin dynamics?
:::
::::
