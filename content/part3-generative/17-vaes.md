---
id: ch17-vaes
summary: The ELBO is the most-asked derivation in generative modelling. Derive it two ways and check the gap numerically on a model where everything is known in closed form. Then implement the reparameterisation trick and the Gaussian KL, and train a VAE.
udl: "Ch. 17 — Variational autoencoders"
---

## A latent-variable model you can train

A **VAE** is a generative model with a latent variable: draw $z \sim p(z) = \mathcal N(0, I)$, then draw $x \sim p_\theta(x\mid z)$ from a decoder network. Maximum likelihood would need
$$\log p_\theta(x) = \log \int p_\theta(x\mid z)\,p(z)\,dz,$$
an integral over every possible $z$, which is intractable. The fix has two parts:

1. **An encoder** $q_\phi(z\mid x)$ approximates the posterior, i.e. which $z$ probably produced this $x$.
2. **A lower bound**, the **ELBO**, replaces $\log p(x)$. It's tractable, and maximising it trains both networks together.

$$\text{ELBO} = \underbrace{\mathbb E_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big]}_{\text{reconstruction}} - \underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{stay close to the prior}} .$$

Compared with Chapter 16's autoencoder, the KL term shapes the code space so that you can **sample**: decoding $z \sim \mathcal N(0, I)$ gives plausible data.

::::exercise{id=elbo-derive type=derive difficulty=2 time=10 runtime=browser udl="§17.3–17.4" tags="vaes,elbo-derivation"}
# Derive the ELBO, two ways

:::prereq
- **Jensen's inequality.** For concave $f$ (like $\log$), $f(\mathbb E[X]) \ge \mathbb E[f(X)]$.
- **Importance weighting.** $\int p(x,z)\,dz = \mathbb E_{q(z)}\!\big[p(x,z)/q(z)\big]$ for any $q$ that is positive wherever $p$ is.
- **KL divergence.** $\mathrm{KL}(q\|p) = \mathbb E_q[\log q - \log p] \ge 0$, with equality if and only if $q = p$.
- **Bayes.** $p(x, z) = p(z\mid x)\,p(x)$.
:::

:::task
Derive $\log p(x) \ge \text{ELBO}$ **twice**: once with Jensen's inequality, and once by showing that $\log p(x) - \text{ELBO}$ is a KL divergence (which also tells you exactly what the gap is). Then split the ELBO into reconstruction and KL terms, and give the closed-form KL for a diagonal Gaussian $q$ against $\mathcal N(0, I)$.
:::

:::scaffold
Write `q(z|x)` for the encoder, `p(z|x)` for the true posterior, and `KL(a||b)` for divergences.

**1. Importance weighting.** $\log p(x) = \log \mathbb E_{q(z\mid x)}\!\left[\,\cdot\,\right]$, where the expectation is over {{ p(x,z)/q(z|x) | p(x, z) / q(z|x) | p(x|z)p(z)/q(z|x) | p(x|z) p(z) / q(z|x) ? Multiply and divide the integrand by $q(z\mid x)$, so the integral becomes an expectation under $q$. }}.

**2. Jensen.** Because $\log$ is concave, $\log \mathbb E[\cdot]$ {{ ≥ | >= | \geq | \ge ? For a concave function, the function of the mean is at least the mean of the function. }} $\mathbb E[\log\,\cdot\,]$, and the right-hand side is the ELBO.

**3. Split the ELBO.** Using $p(x,z) = p(x\mid z)p(z)$: $\text{ELBO} = \mathbb E_q[\log p(x\mid z)] - $ {{ KL(q(z|x)||p(z)) | KL(q||p(z)) | D_KL(q(z|x)||p(z)) | re:^(d_)?\{?kl\}?\(q.*(\|\||\\\|)p\(z\)\)$ ? The $\log p(z) - \log q(z\mid x)$ part, averaged under $q$, is minus the KL from the encoder to the **prior**. }}

**4. The exact gap.** $\log p(x) - \text{ELBO} = \mathrm{KL}\big(q(z\mid x)\,\|\,$ {{ p(z|x) | p(z \mid x) | the posterior | posterior | re:p\(z(\||\\mid)x\) ? Use Bayes, $p(x,z) = p(z\mid x)p(x)$, inside the ELBO. $\log p(x)$ comes out, and what's left is minus the KL to the **true posterior**. }}$\big)$, so the bound is tight exactly when the encoder matches the true posterior.

**5. Closed-form KL.** For $q = \mathcal N(\mu, \sigma^2)$ against $\mathcal N(0, 1)$, per dimension: $\mathrm{KL} = \tfrac12\big(\mu^2 + \sigma^2 - 1 -$ {{ log σ^2 | \log\sigma^2 | log(σ^2) | 2 log σ | 2\log\sigma | logvar | \log \sigma^2 ? $\mathbb E_q[\log q] = -\tfrac12\log(2\pi e\sigma^2)$ and $\mathbb E_q[\log p] = -\tfrac12\log 2\pi - \tfrac12(\mu^2+\sigma^2)$. Subtract them. }}$\big)$.
:::

:::derivation
**Importance-weight the marginal.**
$$\log p(x) = \log\int p(x,z)\,dz = \log \mathbb E_{q(z\mid x)}\!\left[\frac{p(x,z)}{q(z\mid x)}\right].$$

---step---

**Route 1: Jensen.** $\log$ is concave, so moving it inside the expectation can only lower the value:
$$\log p(x) \;\ge\; \mathbb E_{q}\!\left[\log\frac{p(x,z)}{q(z\mid x)}\right] =: \text{ELBO}.$$

---step---

**Route 2: the gap is a KL.** Substitute $p(x,z) = p(z\mid x)\,p(x)$ into the ELBO:
$$\text{ELBO} = \mathbb E_q\!\left[\log p(x) + \log\frac{p(z\mid x)}{q(z\mid x)}\right] = \log p(x) - \mathrm{KL}\big(q(z\mid x)\,\|\,p(z\mid x)\big).$$
KL is at least 0, so the bound holds, and **the gap is exactly the encoder's error**. Maximising the ELBO over $\phi$ therefore pulls $q$ towards the true posterior.

---step---

**The training form.** Instead, write $p(x,z) = p(x\mid z)p(z)$:
$$\text{ELBO} = \mathbb E_q[\log p(x\mid z)] - \mathrm{KL}\big(q(z\mid x)\,\|\,p(z)\big).$$
The first term is reconstruction, estimated with samples of $z$ (one is enough). The second has a closed form for Gaussians:
$$\mathrm{KL}\big(\mathcal N(\mu,\sigma^2)\,\|\,\mathcal N(0,1)\big) = \tfrac12\big(\mu^2+\sigma^2-1-\log\sigma^2\big)\ \text{ per dimension}.$$

---step---

**Why it trains.** Sampling $z \sim q_\phi$ blocks gradients to $\phi$. The **reparameterisation trick** writes $z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon$ with $\epsilon\sim\mathcal N(0, I)$. The randomness now sits in $\epsilon$, so $z$ is a differentiable function of $\phi$ (Exercise 17.2).

Known failure mode: **posterior collapse**. A powerful decoder ignores $z$, the KL term drives $q$ to the prior, and the code carries no information. Common fixes are KL annealing, "free bits", or a weaker decoder.
:::

:::solution{lang=python}
import numpy as np

def log_px(x, s2):
    v = 1 + s2
    return -0.5 * np.log(2 * np.pi * v) - x ** 2 / (2 * v)

def elbo(x, s2, m, v):
    recon = -0.5 * np.log(2 * np.pi * s2) - ((x - m) ** 2 + v) / (2 * s2)
    kl = 0.5 * (m ** 2 + v - 1 - np.log(v))
    return recon - kl

def kl_gauss(m1, v1, m2, v2):
    return 0.5 * (np.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1)

def posterior(x, s2):
    return x / (1 + s2), s2 / (1 + s2)
:::

:::explain
[L3-5] **A model where everything is exact.** With $z\sim\mathcal N(0,1)$ and $x\mid z\sim\mathcal N(z, s^2)$, the marginal is $x \sim \mathcal N(0, 1+s^2)$. So $\log p(x)$ is known.

[L7-10] **The ELBO for any Gaussian $q = \mathcal N(m, v)$.** The reconstruction expectation is analytic, because $\mathbb E_q[(x-z)^2] = (x-m)^2 + v$. The KL to the prior is the closed form above.

[L12-16] **The true posterior** is Gaussian, with mean $x/(1+s^2)$ and variance $s^2/(1+s^2)$. The tests check the derivation's punchline exactly: $\log p(x) - \text{ELBO}(q) = \mathrm{KL}(q\,\|\,\text{posterior})$ for *every* $q$, and the gap is zero at the true posterior.
:::

:::tests{lang=python}
import numpy as np

def test_gap_equals_kl_to_posterior():
    x, s2 = 1.3, 0.5
    pm, pv = posterior(x, s2)
    for m, v in [(0.0, 1.0), (0.4, 0.2), (pm, pv), (-1.0, 3.0)]:
        assert np.isclose(log_px(x, s2) - elbo(x, s2, m, v), kl_gauss(m, v, pm, pv))

def test_bound_is_tight_only_at_the_posterior():
    x, s2 = -0.7, 2.0
    pm, pv = posterior(x, s2)
    assert np.isclose(elbo(x, s2, pm, pv), log_px(x, s2))
    assert elbo(x, s2, pm + 0.3, pv) < log_px(x, s2)
:::

:::hints
1. Start from $\log\int p(x,z)\,dz$ and introduce $q$ by multiplying and dividing by it.
2. For route 2, substitute Bayes' rule inside the ELBO and pull out $\log p(x)$, which doesn't depend on $z$.
:::

:::interview
- Derive the ELBO. What is the gap between it and $\log p(x)$? *(The KL from $q$ to the true posterior.)*
- Why can't you just maximise $\log p(x)$ directly?
- What's posterior collapse, and how do you fight it?
- β-VAE: what does weighting the KL term trade off? *(Disentanglement and compression against reconstruction quality.)*
:::
::::

::::exercise{id=reparam-kl type=implement difficulty=2 time=15 runtime=browser udl="§17.6–17.7" tags="vaes"}
# The reparameterisation trick and the VAE loss

:::prereq
- **Log-variance parameterisation.** The encoder outputs $\mu$ and $\ell = \log\sigma^2$, so $\sigma = e^{\ell/2}$. That keeps $\sigma > 0$ without any constraint, and is numerically gentler than outputting $\sigma$ directly.
- **The reparameterisation trick.** $z = \mu + \sigma\odot\epsilon$ with $\epsilon \sim \mathcal N(0, I)$.
- **A Gaussian decoder with unit variance.** $-\log p(x\mid z) = \tfrac12\|x - \hat x\|^2 + \text{const}$.
- **KL term.** Summed over latent dimensions: $\tfrac12\sum_j(\mu_j^2 + e^{\ell_j} - 1 - \ell_j)$.
:::

:::task
Implement `reparam(mu, logvar, eps)`, `gaussian_kl(mu, logvar)` (per example, summed over the last axis) and `neg_elbo(x, x_hat, mu, logvar)`. The last is the batch mean of $\tfrac12\|x-\hat x\|^2 + \mathrm{KL}$. All inputs are `(B, ·)` arrays.
:::

:::starter{lang=python}
import numpy as np

def reparam(mu, logvar, eps):
    raise NotImplementedError

def gaussian_kl(mu, logvar):
    raise NotImplementedError

def neg_elbo(x, x_hat, mu, logvar):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def reparam(mu, logvar, eps):
    return mu + {{ np.exp(0.5 * logvar) * eps | np.exp(logvar / 2) * eps | eps * np.exp(0.5 * logvar) | np.sqrt(np.exp(logvar)) * eps ? $\sigma = e^{\ell/2}$. Scale the noise by the standard deviation, not the variance. }}

def gaussian_kl(mu, logvar):
    return 0.5 * ({{ mu ** 2 + np.exp(logvar) - 1 - logvar | mu**2 + np.exp(logvar) - 1 - logvar | np.exp(logvar) + mu ** 2 - 1 - logvar ? $\mu^2 + \sigma^2 - 1 - \log\sigma^2$, with $\sigma^2 = e^{\ell}$ and $\log\sigma^2 = \ell$. }}).sum(axis={{ -1 | 1 ? Independent latent dimensions: their KLs **add**. Sum over the latent axis, leaving one value per example. }})

def neg_elbo(x, x_hat, mu, logvar):
    recon = 0.5 * ((x - x_hat) ** 2).sum(axis=-1)
    return {{ (recon + gaussian_kl(mu, logvar)).mean() | np.mean(recon + gaussian_kl(mu, logvar)) | recon.mean() + gaussian_kl(mu, logvar).mean() ? Minus the ELBO is reconstruction **plus** KL, per example, then averaged over the batch. }}
:::

:::solution{lang=python}
import numpy as np

def reparam(mu, logvar, eps):
    return mu + np.exp(0.5 * logvar) * eps

def gaussian_kl(mu, logvar):
    return 0.5 * (mu ** 2 + np.exp(logvar) - 1 - logvar).sum(axis=-1)

def neg_elbo(x, x_hat, mu, logvar):
    recon = 0.5 * ((x - x_hat) ** 2).sum(axis=-1)
    return (recon + gaussian_kl(mu, logvar)).mean()
:::

:::explain
[L4] **Noise outside, parameters inside.** The gradients with respect to `mu` and `logvar` flow through this expression. Sampling `np.random.normal(mu, sigma)` directly would hide them, because the sampler is a black box.

[L7] **KL per example.** Sum over latent dimensions, *not* mean. Averaging silently down-weights the KL by the latent size, which is a common reason a VAE "doesn't regularise".

[L10-11] **Minus the ELBO.** It's averaged over the batch, so the learning rate doesn't depend on batch size. The tests check the KL against a Monte Carlo estimate and against the closed form's defining properties: it's zero at the prior, and positive elsewhere.
:::

:::tests{lang=python}
import numpy as np

def test_reparam_has_the_right_distribution():
    rng = np.random.default_rng(0)
    mu, logvar = np.full((200000, 1), 1.5), np.full((200000, 1), np.log(0.25))
    z = reparam(mu, logvar, rng.normal(size=mu.shape))
    assert abs(z.mean() - 1.5) < 0.01 and abs(z.std() - 0.5) < 0.01

def test_kl_zero_at_prior_and_positive_elsewhere():
    assert np.allclose(gaussian_kl(np.zeros((3, 4)), np.zeros((3, 4))), 0)
    assert (gaussian_kl(np.random.default_rng(1).normal(size=(5, 4)), np.random.default_rng(2).normal(size=(5, 4))) > 0).all()

def test_kl_matches_monte_carlo():
    rng = np.random.default_rng(3)
    mu, logvar = np.array([[0.7, -0.3]]), np.array([[-0.5, 0.4]])
    z = reparam(mu, logvar, rng.normal(size=(400000, 2)))
    log_q = -0.5 * (((z - mu) ** 2) / np.exp(logvar) + logvar + np.log(2 * np.pi)).sum(-1)
    log_p = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(-1)
    assert abs((log_q - log_p).mean() - gaussian_kl(mu, logvar)[0]) < 0.01

def test_neg_elbo_combines_terms():
    x, x_hat = np.ones((2, 3)), np.zeros((2, 3))
    mu, logvar = np.zeros((2, 2)), np.zeros((2, 2))
    assert np.isclose(neg_elbo(x, x_hat, mu, logvar), 1.5)
:::

:::hints
1. If the reparameterisation test gives a standard deviation of 0.25, you used the variance instead of the standard deviation.
2. The KL is a sum over latent dimensions; the final loss is a mean over the batch.
:::

:::interview
- Why does the reparameterisation trick make gradients possible? What estimator would you use for discrete latents instead? *(REINFORCE / score function, or Gumbel-softmax.)*
- Why output $\log\sigma^2$ rather than $\sigma$?
- Why sum the KL over latent dimensions rather than averaging?
:::
::::

::::exercise{id=vae-torch type=implement difficulty=2 time=20 runtime=local-torch udl="§17.5–17.7" tags="vaes,pytorch"}
# Train a VAE

:::prereq
**The encoder** maps $x \to$ hidden $\to (\mu, \log\sigma^2)$, as two output heads or one layer split in two. **The decoder** maps $z \to$ hidden $\to \hat x$. The loss is Exercise 17.2's negative ELBO, written in torch. Use `torch.randn_like(std)` for $\epsilon$.

Sampling new data: draw $z \sim \mathcal N(0, I)$ and decode it.
:::

:::task
Implement `VAE(d_in, d_latent, d_hidden)` with:

- `encode(x) -> (mu, logvar)`,
- `decode(z)`,
- `forward(x) -> (x_hat, mu, logvar)`, using the reparameterisation trick,
- `loss(x)`: the batch-mean negative ELBO, $\tfrac12\|x-\hat x\|^2 + \mathrm{KL}$ per example.
:::

:::starter{lang=python}
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, d_in, d_latent, d_hidden):
        super().__init__()
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def loss(self, x):
        raise NotImplementedError
:::

:::scaffold{lang=python}
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, d_in, d_latent, d_hidden):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hidden), nn.Tanh(), nn.Linear(d_hidden, 2 * d_latent))
        self.dec = nn.Sequential(nn.Linear(d_latent, d_hidden), nn.Tanh(), nn.Linear(d_hidden, d_in))

    def encode(self, x):
        mu, logvar = self.enc(x).{{ chunk(2, dim=-1) | chunk(2, -1) ? One head produces both statistics. Split its output in half along the feature axis. }}
        return mu, logvar

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + {{ torch.exp(0.5 * logvar) * torch.randn_like(mu) | (0.5 * logvar).exp() * torch.randn_like(mu) | torch.randn_like(logvar) * torch.exp(0.5 * logvar) | logvar.mul(0.5).exp() * torch.randn_like(mu) ? Reparameterise: the standard deviation times fresh standard-normal noise, so gradients reach `mu` and `logvar`. }}
        return self.decode(z), mu, logvar

    def loss(self, x):
        x_hat, mu, logvar = self(x)
        recon = 0.5 * ((x - x_hat) ** 2).sum(dim=-1)
        kl = {{ 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(dim=-1) | 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(-1) | 0.5 * (mu ** 2 + torch.exp(logvar) - 1 - logvar).sum(dim=-1) ? The closed-form Gaussian KL, summed over the latent dimensions. }}
        return (recon + kl).mean()
:::

:::solution{lang=python}
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, d_in, d_latent, d_hidden):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hidden), nn.Tanh(), nn.Linear(d_hidden, 2 * d_latent))
        self.dec = nn.Sequential(nn.Linear(d_latent, d_hidden), nn.Tanh(), nn.Linear(d_hidden, d_in))

    def encode(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=-1)
        return mu, logvar

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar

    def loss(self, x):
        x_hat, mu, logvar = self(x)
        recon = 0.5 * ((x - x_hat) ** 2).sum(dim=-1)
        kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(dim=-1)
        return (recon + kl).mean()
:::

:::explain
[L7] **One encoder head, two statistics.** The last layer outputs `2 * d_latent` numbers, split into `mu` and `logvar` in `encode`.

[L18-19] **Reparameterise.** The tests check that gradients of the loss reach the encoder's final layer through *both* halves. That's the property this trick exists for.

[L22-26] **Minus the ELBO.** The KL matches `torch.distributions.kl_divergence` in the tests. On data generated from a 2-D latent linear-Gaussian model, training should roughly halve the loss.

**Sampling.** `vae.decode(torch.randn(n, d_latent))` generates new data. That's the thing a plain autoencoder (Chapter 16) can't do reliably.
:::

:::tests{lang=python}
import torch

def _data(n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 2, generator=g)
    A = torch.randn(2, 8, generator=g)
    return z @ A + 0.1 * torch.randn(n, 8, generator=g)

def test_kl_matches_torch_distributions():
    torch.manual_seed(0)
    vae, x = VAE(8, 2, 32), _data(16)
    mu, logvar = vae.encode(x)
    q = torch.distributions.Normal(mu, (0.5 * logvar).exp())
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
    ref = torch.distributions.kl_divergence(q, p).sum(-1)
    mine = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(-1)
    assert torch.allclose(ref, mine, atol=1e-5)

def test_gradients_reach_both_encoder_heads():
    torch.manual_seed(1)
    vae = VAE(8, 2, 32)
    vae.loss(_data(32)).backward()
    g = vae.enc[-1].weight.grad
    assert g[:2].abs().sum() > 0 and g[2:].abs().sum() > 0

def test_training_reduces_the_loss_and_samples_decode():
    torch.manual_seed(2)
    vae, x = VAE(8, 2, 64), _data()
    opt = torch.optim.Adam(vae.parameters(), lr=1e-2)
    start = vae.loss(x).item()
    for _ in range(800):
        opt.zero_grad()
        loss = vae.loss(x)
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * start
    assert vae.decode(torch.randn(5, 2)).shape == (5, 8)
:::

:::hints
1. Get the shapes right first: `encode` returns two `(B, d_latent)` tensors.
2. If the gradient test fails for the `logvar` half, check that you used `logvar` inside the reparameterisation, not just in the KL.
:::

:::interview
- Walk through a VAE forward pass and loss. Which parts are stochastic?
- Why are VAE samples often blurry? *(A Gaussian likelihood averages modes, and the bound isn't tight.)*
- How does a VAE relate to diffusion models? *(A diffusion model is a hierarchical VAE with a fixed encoder.)*
:::
::::
