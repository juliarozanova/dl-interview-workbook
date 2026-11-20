---
id: ch18-gans
summary: A GAN is a two-player game played with logits. See why the original generator loss stalls exactly when the generator is worst, implement numerically stable GAN losses, and fix the `detach` bug that stops the generator from ever learning.
udl: "Ch. 15 — Generative adversarial networks"
---

## A game, not a likelihood

A GAN never writes down $p(x)$. A **generator** $G(z)$ maps noise to samples, and a **discriminator** $D(x)$ outputs the logit of "this is real". They play
$$\min_G\max_D\;\; \mathbb E_{x\sim\text{data}}[\log \sigma(D(x))] + \mathbb E_{z}[\log(1 - \sigma(D(G(z))))].$$
For the optimal $D$, the generator's objective becomes (up to constants) the **Jensen–Shannon divergence** between the data and model distributions.

In practice, training means alternating steps:

- **D step:** push the logits for real samples up and for fakes down. The fakes are treated as *fixed inputs*.
- **G step:** move $G$ so that $D$ scores its samples as real. Gradients flow *through* $D$ into $G$, but only $G$ is updated.

Getting the gradient plumbing right is half of every GAN interview question. The other half is why training is unstable: mode collapse, vanishing gradients, and the non-saturating trick.

::::exercise{id=gan-saturation-predict type=predict difficulty=2 time=6 runtime=browser udl="§15.2" tags="gans"}
# Why the original generator loss stalls

:::task
Early in training, the discriminator easily spots fakes: $\sigma(D(G(z))) \approx 0.001$. Compare the generator's gradient with respect to the fake logit $f$ under the **original (saturating)** loss, $\log(1 - \sigma(f))$ (minimised by $G$), and the **non-saturating** loss, $-\log\sigma(f)$. Which gives the larger gradient?
:::

:::choices
- [ ] The saturating loss: it's the "true" minimax objective
- [x] The non-saturating loss, by a factor of about 1,000
- [ ] They're equal: both have the same fixed point
- [ ] Both are essentially zero
:::

:::experiment{lang=python}
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def grad_saturating(f):          # d/df log(1 - sigma(f))
    return -sigmoid(f)

def grad_nonsaturating(f):       # d/df -log sigma(f)
    return -(1 - sigmoid(f))

for d_fake in [0.001, 0.01, 0.1, 0.5, 0.9]:
    f = np.log(d_fake / (1 - d_fake))       # logit with sigma(f) = d_fake
    s, n = grad_saturating(f), grad_nonsaturating(f)
    print(f"D(G(z)) = {d_fake:<5}  saturating grad {s:9.4f}   non-saturating grad {n:8.4f}   ratio {n / s:7.1f}")
:::

:::tests{lang=python}
import numpy as np

def test_saturating_vanishes_when_D_is_confident():
    f = np.log(0.001 / 0.999)
    assert abs(grad_saturating(f)) < 0.002 and abs(grad_nonsaturating(f)) > 0.99

def test_same_direction_everywhere():
    for f in np.linspace(-8, 8, 33):
        assert np.sign(grad_saturating(f)) == np.sign(grad_nonsaturating(f))
:::

:::explain
With $p = \sigma(f)$:

- **Saturating:** $\frac{d}{df}\log(1 - p) = -p$. When $D$ is confident the sample is fake ($p \approx 0.001$), the gradient is about $-0.001$. The generator gets almost no signal at exactly the moment it's worst.
- **Non-saturating:** $\frac{d}{df}\,[-\log p] = -(1-p) \approx -1$. That's a strong signal when the generator is bad, fading only as it succeeds.

Both push $f$ the same way, which is why Goodfellow et al. proposed the swap: same fixed point, much better early gradients. It's the default in practice. (Strictly, the non-saturating generator no longer minimises JS divergence, but it trains far better.)

**The bigger picture.** When the real and fake distributions barely overlap, a near-perfect discriminator exists, and JS-based losses give vanishing or uninformative gradients. That motivated the **Wasserstein GAN** (a critic with bounded slope, giving useful gradients even with disjoint supports) and regularisers such as **gradient penalties** and **spectral normalisation**.
:::

:::interview
- Why does the original minimax generator loss saturate? What's the fix?
- What divergence does the GAN objective minimise at the optimal discriminator? *(Jensen–Shannon, up to constants.)*
- Why did WGANs help? What constraint does the critic need, and how is it enforced?
:::
::::

::::exercise{id=gan-losses type=implement difficulty=2 time=12 runtime=browser udl="§15.1–15.2" tags="gans,numerical-stability,custom-losses"}
# Stable GAN losses from logits

:::prereq
**Softplus.** $\mathrm{softplus}(x) = \log(1 + e^x)$ gives
$$-\log\sigma(x) = \mathrm{softplus}(-x),\qquad -\log(1-\sigma(x)) = \mathrm{softplus}(x).$$

Computing $\log(\sigma(x))$ directly underflows for $x \ll 0$, exactly as `log(softmax)` did in Chapter 04. `np.logaddexp(0, x)` is a stable softplus.

The **discriminator loss** is binary cross-entropy with real = 1 and fake = 0. The **non-saturating generator loss** is binary cross-entropy that labels the fakes 1.
:::

:::task
Implement, all taking logits and returning batch means:

- `softplus(x)`, stable for $|x| \sim 10^3$;
- `d_loss(real_logits, fake_logits)` $= \mathbb E[-\log\sigma(r)] + \mathbb E[-\log(1-\sigma(f))]$;
- `g_loss(fake_logits)` $= \mathbb E[-\log\sigma(f)]$, the non-saturating version.
:::

:::starter{lang=python}
import numpy as np

def softplus(x):
    raise NotImplementedError

def d_loss(real_logits, fake_logits):
    raise NotImplementedError

def g_loss(fake_logits):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def softplus(x):
    return {{ np.logaddexp(0, x) | np.logaddexp(0.0, x) | np.logaddexp(x, 0) | np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x))) ? $\log(e^0 + e^x)$: a two-term log-sum-exp, which `np.logaddexp` computes without overflow. }}

def d_loss(real_logits, fake_logits):
    return softplus({{ -real_logits ? $-\log\sigma(r) = \mathrm{softplus}(-r)$: penalise real samples that get a low logit. }}).mean() + softplus({{ fake_logits | +fake_logits ? $-\log(1 - \sigma(f)) = \mathrm{softplus}(f)$: penalise fakes that get a high logit. }}).mean()

def g_loss(fake_logits):
    return {{ softplus(-fake_logits).mean() | np.mean(softplus(-fake_logits)) ? Non-saturating: the generator wants its fakes classified as **real**, so it's the "real" term applied to the fake logits. }}
:::

:::solution{lang=python}
import numpy as np

def softplus(x):
    return np.logaddexp(0, x)

def d_loss(real_logits, fake_logits):
    return softplus(-real_logits).mean() + softplus(fake_logits).mean()

def g_loss(fake_logits):
    return softplus(-fake_logits).mean()
:::

:::explain
[L4] **Stable softplus.** `np.logaddexp(0, x)` computes $\log(e^0 + e^x)$ by factoring out the larger exponent. It's finite for $x = \pm 1000$, where `np.log(1 + np.exp(x))` overflows to `inf`.

[L7] **Discriminator.** This is BCE written as two softplus terms. With PyTorch you'd call `F.binary_cross_entropy_with_logits` for the same reason: never apply a sigmoid and then a log.

[L10] **Generator, non-saturating.** This is the "real" label applied to fakes. Its gradient with respect to $f$ is $-(1 - \sigma(f))$, which stays large while $D$ is winning. That's the previous exercise's point, in code.
:::

:::tests{lang=python}
import numpy as np

def _sig(x):
    return 1 / (1 + np.exp(-x))

def test_matches_naive_formulas_on_moderate_logits():
    rng = np.random.default_rng(0)
    r, f = rng.normal(size=50), rng.normal(size=50)
    assert np.isclose(d_loss(r, f), (-np.log(_sig(r))).mean() + (-np.log(1 - _sig(f))).mean())
    assert np.isclose(g_loss(f), (-np.log(_sig(f))).mean())

def test_extreme_logits_are_finite():
    big = np.array([-1000.0, 1000.0])
    assert np.isfinite(d_loss(big, big)) and np.isfinite(g_loss(big))
    assert np.isclose(softplus(np.array([1000.0]))[0], 1000.0) and softplus(np.array([-1000.0]))[0] >= 0

def test_generator_gradient_does_not_vanish_when_d_wins():
    f, eps = np.array([-7.0]), 1e-5
    grad = (g_loss(f + eps) - g_loss(f - eps)) / (2 * eps)
    assert grad < -0.99
:::

:::hints
1. Write each term as $-\log\sigma(\cdot)$ of something, then use $-\log\sigma(x) = \mathrm{softplus}(-x)$.
2. For the generator, ask: which label does it want the discriminator to assign?
:::

:::interview
- Why work with logits and softplus rather than probabilities and log?
- Write the discriminator and generator losses as BCE calls. Which labels go where?
- Label smoothing for GAN discriminators: what does it do, and why apply it one-sidedly?
:::
::::

::::exercise{id=gan-detach-bug type=debug difficulty=2 time=12 runtime=local-torch udl="§15.1" tags="gans,debugging,training-loop"}
# The generator that never learns

:::prereq
**`.detach()`** returns a tensor that shares the data but is cut out of the autograd graph: no gradients flow back through it.

- **D step:** the fakes should be **detached**. Gradients would otherwise also flow into $G$, which wastes compute, and the leftover gradients can contaminate the next G step.
- **G step:** the fakes must **not** be detached. The gradient has to travel from $D$'s output, back through $D$, into $G$'s parameters. Only `g_opt` steps, so $D$'s weights stay put.
:::

:::task
A colleague's GAN has a discriminator that wins instantly, and the generator's samples never change. These are their two step functions. Find the bug and fix it.
:::

:::starter{lang=python}
import torch
import torch.nn.functional as F

def d_step(G, D, d_opt, real, z):
    d_opt.zero_grad()
    fake = G(z).detach()
    r, f = D(real), D(fake)
    loss = F.binary_cross_entropy_with_logits(r, torch.ones_like(r)) + F.binary_cross_entropy_with_logits(f, torch.zeros_like(f))
    loss.backward()
    d_opt.step()
    return loss.item()

def g_step(G, D, g_opt, z):
    g_opt.zero_grad()
    f = D(G(z).detach())
    loss = F.binary_cross_entropy_with_logits(f, torch.ones_like(f))
    loss.backward()
    g_opt.step()
    return loss.item()
:::

:::scaffold{lang=python}
import torch
import torch.nn.functional as F

def d_step(G, D, d_opt, real, z):
    d_opt.zero_grad()
    fake = G(z).detach()
    r, f = D(real), D(fake)
    loss = F.binary_cross_entropy_with_logits(r, torch.ones_like(r)) + F.binary_cross_entropy_with_logits(f, torch.zeros_like(f))
    loss.backward()
    d_opt.step()
    return loss.item()

def g_step(G, D, g_opt, z):
    g_opt.zero_grad()
    f = D({{ G(z) ? The generator learns only through $D$'s judgement of its samples. Detaching here cuts the only path from the loss back to $G$'s parameters, so they never receive a gradient. }})
    loss = F.binary_cross_entropy_with_logits(f, torch.ones_like(f))
    loss.backward()
    g_opt.step()
    return loss.item()
:::

:::solution{lang=python}
import torch
import torch.nn.functional as F

def d_step(G, D, d_opt, real, z):
    d_opt.zero_grad()
    fake = G(z).detach()
    r, f = D(real), D(fake)
    loss = F.binary_cross_entropy_with_logits(r, torch.ones_like(r)) + F.binary_cross_entropy_with_logits(f, torch.zeros_like(f))
    loss.backward()
    d_opt.step()
    return loss.item()

def g_step(G, D, g_opt, z):
    g_opt.zero_grad()
    f = D(G(z))
    loss = F.binary_cross_entropy_with_logits(f, torch.ones_like(f))
    loss.backward()
    g_opt.step()
    return loss.item()
:::

:::explain
[L6] **Detach in the D step.** It's correct here: the discriminator is learning to classify *these particular* fakes, and nothing should flow into $G$.

[L14] **Don't detach in the G step.** That's the whole fix. With `.detach()`, `loss.backward()` still runs (D's parameters require gradients), but every `p.grad` in $G$ stays `None`, so `g_opt.step()` silently does nothing. That's why there was no error, only a generator that never moved.

**A side effect to know about.** The G step's backward pass *also* fills $D$'s `.grad`. That's harmless, because `d_step` calls `d_opt.zero_grad()` first. Forget that call, and the discriminator would take a step using the generator's gradients. Some codebases set `requires_grad_(False)` on $D$ during the G step to make this explicit and save compute.
:::

:::tests{lang=python}
import torch

def _nets(seed=0):
    torch.manual_seed(seed)
    G = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
    D = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    return G, D

def test_g_step_updates_the_generator():
    G, D = _nets()
    before = [p.clone() for p in G.parameters()]
    g_step(G, D, torch.optim.SGD(G.parameters(), lr=0.1), torch.randn(32, 2))
    assert any(not torch.equal(a, b) for a, b in zip(before, G.parameters()))

def test_g_step_leaves_the_discriminator_alone():
    G, D = _nets(1)
    before = [p.clone() for p in D.parameters()]
    g_step(G, D, torch.optim.SGD(G.parameters(), lr=0.1), torch.randn(32, 2))
    assert all(torch.equal(a, b) for a, b in zip(before, D.parameters()))

def test_d_step_sends_no_gradient_to_the_generator():
    G, D = _nets(2)
    d_step(G, D, torch.optim.SGD(D.parameters(), lr=0.1), torch.randn(32, 2) + 3, torch.randn(32, 2))
    assert all(p.grad is None for p in G.parameters())
:::

:::hints
1. After calling `g_step`, print `[p.grad for p in G.parameters()]`.
2. Which of the two steps should cut the graph, and which must not?
:::

:::interview
- Where do you `.detach()` in a GAN training loop, and why?
- What's mode collapse? Name two mitigations. *(Minibatch discrimination, unrolled GANs, WGAN-GP, or using a diffusion model instead.)*
- Why are GANs hard to evaluate? *(There's no likelihood. FID is used, with caveats.)*
:::
::::
