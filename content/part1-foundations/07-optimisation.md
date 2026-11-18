---
id: ch07-optimisation
summary: Curvature decides what learning rate you can afford. Watch gradient descent blow up just past 2/λ and momentum fix the ill-conditioning; implement warmup + cosine schedules and global-norm clipping; then write Adam and AdamW as a real `torch.optim.Optimizer` that matches PyTorch's to six decimal places.
udl: "Ch. 6 — Fitting models"
---

## Learning rate meets curvature

Near a minimum, any smooth loss looks like a quadratic, $L(\theta) \approx \tfrac12 (\theta-\theta^\star)^\top H (\theta-\theta^\star)$. Gradient descent then acts on each eigendirection of the Hessian $H$ separately. Along an eigenvector with eigenvalue $\lambda$, one step multiplies the error by $(1 - \eta\lambda)$. So:

- Gradient descent converges only if $|1 - \eta\lambda| < 1$ for **every** $\lambda$, which means $\eta < 2/\lambda_{\max}$.
- The flattest direction, $\lambda_{\min}$, then improves by only $1 - \eta\lambda_{\min} \approx 1 - 2/\kappa$ per step, where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the **condition number**.

Momentum, adaptive methods such as Adam, normalisation and good initialisation are all, in part, ways of fighting a large $\kappa$.

::::exercise{id=curvature-lr-predict type=predict difficulty=2 time=8 runtime=browser udl="§6.1–6.3" tags="loss-landscape,optimisers"}
# One learning rate, two curvatures

:::task
Run gradient descent on $f(x) = \tfrac12\,(x_1^2 + 100\,x_2^2)$, starting from $(1, 1)$, with learning rate $\eta = 0.021$. What happens?
:::

:::choices
- [ ] It converges, slowly, because of the flat $x_1$ direction
- [ ] It converges quickly
- [x] It diverges
- [ ] It oscillates forever with a constant amplitude
:::

:::experiment{lang=python}
import numpy as np

H = np.array([1.0, 100.0])          # Hessian eigenvalues (diagonal quadratic)

def gd(lr, steps=1000):
    x = np.array([1.0, 1.0])
    for _ in range(steps):
        x = x - lr * H * x
    return x

def steps_to(tol, lr, beta=0.0, max_steps=5000):
    x, v = np.array([1.0, 1.0]), np.zeros(2)
    for t in range(max_steps):
        if np.abs(x).max() < tol:
            return t
        v = beta * v - lr * H * x            # heavy-ball momentum
        x = x + v
    return max_steps

for lr in [0.019, 0.020, 0.021]:
    print(f"GD lr={lr}:  x after 1000 steps = {gd(lr)}")

kappa = H.max() / H.min()
best_gd = 2 / (H.max() + H.min())
hb_lr = 4 / (np.sqrt(H.max()) + np.sqrt(H.min())) ** 2
hb_beta = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2
print(f"\nsteps to 1e-6: best-lr GD = {steps_to(1e-6, best_gd)},  heavy-ball momentum = {steps_to(1e-6, hb_lr, hb_beta)}")
:::

:::tests{lang=python}
import numpy as np

def test_just_below_threshold_converges():
    assert np.abs(gd(0.019)).max() < 1e-6

def test_just_above_threshold_diverges():
    assert np.abs(gd(0.021)[1]) > 1e6

def test_momentum_beats_best_gd_on_ill_conditioning():
    best = steps_to(1e-6, 2 / 101)
    hb = steps_to(1e-6, 4 / 11 ** 2, (9 / 11) ** 2)
    assert hb * 5 < best
:::

:::explain
Along $x_2$ the curvature is $\lambda = 100$, so each step multiplies $x_2$ by $1 - \eta\lambda = 1 - 2.1 = -1.1$. The error flips sign *and grows* by 10% per step: after 1000 steps that's $1.1^{1000} \approx 10^{41}$. Meanwhile $x_1$ (with $\lambda = 1$) shrinks by $0.979$ per step. **One steep direction is enough to wreck the run.**

The threshold is $\eta < 2/\lambda_{\max} = 0.02$. At $0.019$ the steep direction oscillates but shrinks ($|1-1.9| = 0.9$). At exactly $0.020$ it neither grows nor shrinks, so it oscillates forever, which is the trap in the last option.

**Why momentum helps.** The best fixed step for gradient descent is $2/(\lambda_{\max}+\lambda_{\min})$, which gives a convergence rate of $(\kappa-1)/(\kappa+1) \approx 0.98$ here. Heavy-ball momentum, tuned properly, reaches $(\sqrt\kappa-1)/(\sqrt\kappa+1) \approx 0.82$. It effectively replaces $\kappa$ with $\sqrt\kappa$, so in the experiment it needs about 7× fewer steps (95 against 691).

**In real networks** you never know $\lambda_{\max}$, and it changes during training. That's one reason for **warmup** (next exercise): it starts gently while the curvature is high and uncertain. It's also why loss **spikes** often mean the learning rate has just crossed $2/\lambda_{\max}$ (the "edge of stability").
:::

:::interview
- What's the maximum stable learning rate for gradient descent on a quadratic? Derive it. *($2/\lambda_{\max}$.)*
- What's the condition number, and how does it control convergence speed?
- Saddle points: why are they a bigger obstacle than local minima in high dimensions, and how does noise in SGD help?
:::
::::

## Schedules and clipping

::::exercise{id=schedule-and-clipping type=implement difficulty=2 time=15 runtime=browser udl="§6.3" tags="lr-schedules,gradient-clipping"}
# Warmup + cosine schedule, and global-norm clipping

:::prereq
**Linear warmup, then cosine decay.** For step $t$ (counting from 0):
$$\eta_t = \begin{cases} \eta_\text{base}\,\dfrac{t+1}{W} & t < W\\[6pt] \eta_\text{min} + \tfrac12(\eta_\text{base}-\eta_\text{min})\big(1+\cos(\pi p)\big) & t \ge W,\quad p = \dfrac{t-W}{T-W}\ \text{(capped at 1)}\end{cases}$$

**Global-norm clipping** treats *all* gradients as one long vector $g$. If $\|g\|_2 > c$, it rescales every gradient by $c/\|g\|$. This keeps the update's **direction** and caps its **length**. Clipping each tensor separately would change the direction.
:::

:::task
Implement `lr_at(step, base_lr, warmup, total, min_lr=0.0)` as above, and `clip_grad_norm(grads, max_norm)`. `grads` is a list of NumPy arrays. Return `(clipped_list, total_norm_before)`, and add $10^{-6}$ to the norm in the denominator for safety.
:::

:::starter{lang=python}
import math
import numpy as np

def lr_at(step, base_lr, warmup, total, min_lr=0.0):
    raise NotImplementedError

def clip_grad_norm(grads, max_norm):
    """grads: list of arrays -> (list of clipped arrays, global L2 norm before clipping)."""
    raise NotImplementedError
:::

:::scaffold{lang=python}
import math
import numpy as np

def lr_at(step, base_lr, warmup, total, min_lr=0.0):
    if step < warmup:
        return {{ base_lr * (step + 1) / warmup | base_lr * (step + 1) / float(warmup) | (step + 1) / warmup * base_lr ? Linear ramp: the first step uses $1/W$ of the base rate, and step $W-1$ reaches the full rate. }}
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + {{ math.cos(math.pi * min(1.0, progress)) | math.cos(math.pi * min(progress, 1.0)) | np.cos(np.pi * min(1.0, progress)) | math.cos(math.pi * min(1, progress)) ? Half a cosine period: $\cos(0) = 1$ gives the base rate, and $\cos(\pi) = -1$ gives the minimum. Cap the progress so the rate stays at the minimum after `total`. }})

def clip_grad_norm(grads, max_norm):
    total = math.sqrt({{ sum(float((g ** 2).sum()) for g in grads) | sum(float(np.sum(g ** 2)) for g in grads) | sum((g ** 2).sum() for g in grads) | sum(np.sum(g * g) for g in grads) ? One norm over **all** parameters together: sum every squared entry across every tensor, then take a single square root. }})
    scale = {{ min(1.0, max_norm / (total + 1e-6)) | min(1, max_norm / (total + 1e-6)) ? Shrink only when the norm exceeds the threshold, and never scale up. }}
    return [g * scale for g in grads], total
:::

:::solution{lang=python}
import math
import numpy as np

def lr_at(step, base_lr, warmup, total, min_lr=0.0):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))

def clip_grad_norm(grads, max_norm):
    total = math.sqrt(sum(float((g ** 2).sum()) for g in grads))
    scale = min(1.0, max_norm / (total + 1e-6))
    return [g * scale for g in grads], total
:::

:::explain
[L5-6] **Warmup.** Using `step + 1` means step 0 already has a non-zero rate. Early in training the curvature is high and Adam's second-moment estimates are noisy, so a small learning rate avoids an early blow-up.

[L7-8] **Cosine decay.** It eases into the minimum rate rather than stopping abruptly. The `min(1.0, …)` keeps the schedule flat if you train past `total`. The `max(1, …)` guards against division by zero when `warmup == total`.

[L11] **Global norm.** One number for the whole model, $\sqrt{\sum_\text{tensors}\sum_\text{entries} g^2}$. That's what `torch.nn.utils.clip_grad_norm_` computes.

[L12-13] **Rescale everything by the same factor.** The update direction is unchanged, only its length is capped. Returning the **pre-clip** norm matters: it's the number you log to spot instabilities, because spikes in the gradient norm usually come just before spikes in the loss.
:::

:::tests{lang=python}
import math
import numpy as np

def test_warmup_is_linear():
    assert math.isclose(lr_at(0, 1.0, 10, 100), 0.1)
    assert math.isclose(lr_at(9, 1.0, 10, 100), 1.0)

def test_cosine_endpoints_and_midpoint():
    assert math.isclose(lr_at(10, 1.0, 10, 110), 1.0)
    assert math.isclose(lr_at(60, 1.0, 10, 110, min_lr=0.2), 0.6)
    assert math.isclose(lr_at(110, 1.0, 10, 110, min_lr=0.2), 0.2)
    assert math.isclose(lr_at(500, 1.0, 10, 110, min_lr=0.2), 0.2)

def test_decay_is_monotone():
    lrs = [lr_at(t, 3e-4, 100, 1000) for t in range(100, 1000)]
    assert all(b <= a for a, b in zip(lrs, lrs[1:]))

def test_clipping_uses_the_global_norm():
    g = [np.array([3.0, 0.0]), np.array([[0.0, 4.0]])]
    clipped, norm = clip_grad_norm(g, 1.0)
    assert math.isclose(norm, 5.0)
    assert math.isclose(math.sqrt(sum((c ** 2).sum() for c in clipped)), 1.0, rel_tol=1e-5)
    assert np.allclose(clipped[0] / clipped[1].ravel()[1], g[0] / 4.0)

def test_small_gradients_are_untouched():
    g = [np.array([0.1, -0.2]), np.array([0.05])]
    clipped, _ = clip_grad_norm(g, 10.0)
    assert all(np.array_equal(a, b) for a, b in zip(clipped, g))
:::

:::hints
1. Check the endpoints: what should step 0, step `warmup - 1`, step `warmup` and step `total` give?
2. One norm for everything: don't clip each tensor separately.
:::

:::interview
- Why warm up? Give two reasons. *(High early curvature, and noisy early Adam statistics.)*
- Per-tensor versus global-norm clipping: which preserves the update direction?
- Clipping by value versus by norm? *(Clipping by value changes the direction, but is cheap and robust to single extreme entries.)*
:::
::::

## Adam and AdamW, for real

::::exercise{id=adam-adamw type=implement difficulty=3 time=30 runtime=local-torch udl="§6.4" tags="optimisers,optimiser-impl,weight-decay"}
# Write Adam and AdamW as a `torch.optim.Optimizer`

:::prereq
**Adam** keeps running averages of the gradient and of its square, per parameter:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2.$$

Both start at zero, so early on they're biased towards zero. **Bias correction** fixes this: $\hat m_t = m_t/(1-\beta_1^t)$ and $\hat v_t = v_t/(1-\beta_2^t)$. The update is $\theta \leftarrow \theta - \eta\,\hat m_t / (\sqrt{\hat v_t} + \epsilon)$.

**L2 versus decoupled weight decay.**

- **Adam + L2** adds $\lambda\theta$ to the gradient *before* the moments are computed, so the decay is divided by $\sqrt{\hat v}$ along with everything else.
- **AdamW** applies $\theta \leftarrow \theta(1 - \eta\lambda)$ *separately*, so every weight decays at the same rate (Chapter 08 shows why that matters).

**The `Optimizer` API.** Defaults go into `param_groups`. Per-parameter buffers live in `self.state[p]`. `step()` runs under `torch.no_grad()` and updates the parameters in place.
:::

:::task
Implement `Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False)`:

- `decoupled=False`: L2-style decay. This must match `torch.optim.Adam(weight_decay=…)`.
- `decoupled=True`: AdamW. This must match `torch.optim.AdamW(weight_decay=…)`.

The tests train the same small model with yours and with PyTorch's for 25 steps, and compare every parameter.
:::

:::starter{lang=python}
import torch

class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, decoupled=decoupled))

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError
:::

:::scaffold{lang=python}
import torch

class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, decoupled=decoupled))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd and not group["decoupled"]:
                    g = g + {{ wd * p | p * wd | wd * p.data ? L2 regularisation adds $\nabla\,\tfrac\lambda2\|\theta\|^2 = \lambda\theta$ to the gradient, so it flows through both moment estimates. }}
                state = self.state[p]
                if not state:
                    state["t"], state["m"], state["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                state["t"] += 1
                m, v, t = state["m"], state["v"], state["t"]
                m.mul_(b1).add_(g, alpha={{ 1 - b1 ? An exponential moving average puts weight $1-\beta_1$ on the new gradient. }})
                v.mul_(b2).addcmul_(g, g, value={{ 1 - b2 ? The same idea for the **squared** gradient, with its own decay rate $\beta_2$. }})
                m_hat = m / {{ (1 - b1 ** t) | (1 - b1**t) | (1 - pow(b1, t)) ? Both moments start at 0, so after $t$ steps their weights sum to only $1-\beta^t$. Divide by that to remove the bias. }}
                v_hat = v / (1 - b2 ** t)
                if group["decoupled"]:
                    p.mul_({{ 1 - lr * wd | 1 - wd * lr ? AdamW shrinks the weights directly, by the same factor for every parameter, outside the adaptive scaling. }})
                p.add_(-lr * m_hat / ({{ v_hat.sqrt() + group["eps"] | torch.sqrt(v_hat) + group["eps"] ? Per-parameter step size: divide by the root of the second moment, with $\epsilon$ added **outside** the square root, as PyTorch does. }}))
:::

:::solution{lang=python}
import torch

class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, decoupled=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, decoupled=decoupled))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd and not group["decoupled"]:
                    g = g + wd * p
                state = self.state[p]
                if not state:
                    state["t"], state["m"], state["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                state["t"] += 1
                m, v, t = state["m"], state["v"], state["t"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1 ** t)
                v_hat = v / (1 - b2 ** t)
                if group["decoupled"]:
                    p.mul_(1 - lr * wd)
                p.add_(-lr * m_hat / (v_hat.sqrt() + group["eps"]))
:::

:::explain
[L4-5] **Defaults to `param_groups`.** The base class copies these defaults into each group. That's what lets you give different layers different learning rates, or exclude biases from weight decay, with one optimiser.

[L7-8] **`no_grad`.** These updates must not be recorded by autograd. The `closure` argument keeps the signature compatible with optimisers such as L-BFGS that re-evaluate the loss.

[L15-16] **Coupled (L2) decay.** This modifies the gradient *before* the moments. In flat directions $\sqrt{\hat v}$ is small, so the decay gets amplified; in high-gradient directions it gets suppressed.

[L17-20] **Lazy state.** Buffers are created on the first step, with the parameter's shape, device and dtype. That's why `optimizer.state_dict()` is empty until you call `step()`.

[L22-23] **In-place moment updates.** `addcmul_(g, g, value=…)` adds $(1-\beta_2)\,g\odot g$ without allocating a temporary for $g^2$.

[L24-25] **Bias correction.** At $t=1$, $m_1 = (1-\beta_1)g$, so dividing by $1-\beta_1$ recovers $g$ exactly. Without it, the first steps would be 10× too small for $m$ and the step sizes distorted by $\sqrt{1000}$ for $v$.

[L26-27] **Decoupled decay.** This happens before the gradient step and is independent of $\hat v$. It's the only difference between AdamW and Adam with L2, and the tests check both against PyTorch.

[L28] **The update.** $\epsilon$ is added *outside* the square root, matching PyTorch. Other libraries (some TensorFlow versions) differ, which matters when $\hat v$ is tiny.
:::

:::tests{lang=python}
import torch

def _run(make_opt, steps=25, seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    x, y = torch.randn(32, 5), torch.randn(32, 2)
    opt = make_opt(model.parameters())
    for _ in range(steps):
        opt.zero_grad()
        ((model(x) - y) ** 2).mean().backward()
        opt.step()
    return [p.detach().clone() for p in model.parameters()]

def _close(a, b):
    return all(torch.allclose(p, q, atol=1e-6) for p, q in zip(a, b))

def test_matches_torch_adam():
    assert _close(_run(lambda ps: Adam(ps, lr=1e-2)), _run(lambda ps: torch.optim.Adam(ps, lr=1e-2)))

def test_matches_torch_adam_with_l2():
    assert _close(_run(lambda ps: Adam(ps, lr=1e-2, weight_decay=0.1)),
                  _run(lambda ps: torch.optim.Adam(ps, lr=1e-2, weight_decay=0.1)))

def test_matches_torch_adamw():
    assert _close(_run(lambda ps: Adam(ps, lr=1e-2, weight_decay=0.1, decoupled=True)),
                  _run(lambda ps: torch.optim.AdamW(ps, lr=1e-2, weight_decay=0.1)))

def test_l2_and_decoupled_differ():
    assert not _close(_run(lambda ps: Adam(ps, lr=1e-2, weight_decay=0.1)),
                      _run(lambda ps: Adam(ps, lr=1e-2, weight_decay=0.1, decoupled=True)))
:::

:::hints
1. Everything per-parameter lives in `self.state[p]`, a dict you create on the first step.
2. If only the AdamW test fails, check *when* you apply the decay: before the gradient step, scaled by `lr`.
3. Off by a little everywhere? Check that ε is outside the square root, and that the bias corrections use the step count *after* incrementing.
:::

:::interview
- Why does Adam need bias correction? What would the first update look like without it?
- Adam with L2 versus AdamW: why does the difference matter? *(See Chapter 08, Exercise 8.2.)*
- What memory does Adam use per parameter? *(Two extra buffers, roughly 3× the parameter memory in total. That's why 8-bit and factored optimisers exist.)*
- When might SGD with momentum generalise better than Adam?
:::
::::
