---
id: ch23-scaling
summary: "Loss falls as a power law in model size, data and compute, and that lets you plan training runs before you pay for them. Fit a scaling law with an irreducible-loss floor, derive the compute-optimal split between parameters and tokens, and predict what doubling your compute budget should buy."
udl: "Ch. 20 — Why does deep learning work?"
---

## Predictable returns

Across many orders of magnitude, a language model's loss follows a **power law** plus a floor:
$$L(N) \approx E + \frac{A}{N^{\alpha}}, \qquad L(N, D) \approx E + \frac{A}{N^\alpha} + \frac{B}{D^\beta},$$
where $N$ is the number of parameters and $D$ the number of training tokens. $E$ is the **irreducible loss**, the entropy of the data itself. Training compute is roughly $C \approx 6ND$ FLOPs: 2 for the forward pass and 4 for the backward pass, per parameter per token.

**Why it matters in interviews.** Scaling laws are how labs decide how big a model to train and on how much data. The **Chinchilla** result (Hoffmann et al., 2022) showed that earlier models were *undertrained*: for a fixed budget, parameters and tokens should grow together, at roughly 20 tokens per parameter. UDL Chapter 20 discusses why over-parameterised networks generalise at all, which is the theoretical backdrop to these empirical laws.

::::exercise{id=power-law-fit type=implement difficulty=2 time=15 runtime=browser udl="§20.4" tags="scaling"}
# Fit a scaling law with an irreducible-loss floor

:::prereq
**Without the floor**, a power law is a straight line in log-log space: $\log L = \log A - \alpha\log N$. One `np.polyfit` gives $\alpha$.

**With the floor**, $L - E$ is the power law. But $E$ is unknown, and choosing it wrongly bends the line: too large a guess curves it down, too small curves it up. A simple, robust fit is:

- for each candidate $E$ on a grid below $\min L$, fit a line to $\big(\log N, \log(L - E)\big)$;
- keep the $E$ whose line fits best (smallest squared residual).
:::

:::task
Implement `fit_power_law(N, L, n_grid=400)`, returning `(A, alpha, E)` for $L \approx E + A N^{-\alpha}$. Search $E$ on `np.linspace(0, 0.999 * L.min(), n_grid)`.
:::

:::starter{lang=python}
import numpy as np

def fit_power_law(N, L, n_grid=400):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def fit_power_law(N, L, n_grid=400):
    x = np.log(N)
    best = None
    for E in np.linspace(0, 0.999 * L.min(), n_grid):
        y = {{ np.log(L - E) ? Subtract the candidate floor, then take logs. A correct $E$ makes this a straight line in $\log N$. }}
        slope, intercept = {{ np.polyfit(x, y, 1) | np.polyfit(x, y, deg=1) ? A degree-1 least-squares fit gives the slope and intercept in log-log space. }}
        resid = ((y - (slope * x + intercept)) ** 2).sum()
        if best is None or resid < best[0]:
            best = (resid, {{ np.exp(intercept) ? The intercept is $\log A$. }}, {{ -slope ? The loss **decreases** with $N$, so the fitted slope is $-\alpha$. }}, E)
    return best[1:]
:::

:::solution{lang=python}
import numpy as np

def fit_power_law(N, L, n_grid=400):
    x = np.log(N)
    best = None
    for E in np.linspace(0, 0.999 * L.min(), n_grid):
        y = np.log(L - E)
        slope, intercept = np.polyfit(x, y, 1)
        resid = ((y - (slope * x + intercept)) ** 2).sum()
        if best is None or resid < best[0]:
            best = (resid, np.exp(intercept), -slope, E)
    return best[1:]
:::

:::explain
[L5-7] **Try each floor.** The upper bound stays just below $\min L$, so the log is always defined. Real fits use nonlinear least squares (often with a Huber loss on log-loss) over all three parameters at once. This grid is the same idea, made transparent.

[L8-11] **A line per candidate.** The tests generate data from a known law with 0.2% noise, and check that $\alpha$ comes back within 0.02 and $E$ within 0.05. They also check that ignoring the floor ($E = 0$) gives a visibly wrong exponent. That's the classic mistake when extrapolating scaling curves.
:::

:::tests{lang=python}
import numpy as np

def _data(seed=0):
    N = np.logspace(6, 10, 15)
    L = 1.7 + 400 * N ** -0.34
    return N, L * (1 + 0.002 * np.random.default_rng(seed).normal(size=N.size))

def test_recovers_the_law():
    N, L = _data()
    A, alpha, E = fit_power_law(N, L)
    assert abs(alpha - 0.34) < 0.02 and abs(E - 1.7) < 0.05

def test_ignoring_the_floor_biases_the_exponent():
    N, L = _data(1)
    slope = np.polyfit(np.log(N), np.log(L), 1)[0]
    assert abs(-slope - 0.34) > 0.1
:::

:::hints
1. For a fixed $E$, the fit is linear regression on $\big(\log N, \log(L - E)\big)$.
2. Keep track of the residual, slope and intercept for the best $E$ so far.
:::

:::interview
- Why does a pure power law fail at large scale? *(It ignores the irreducible loss, so it predicts the loss going below the data's entropy.)*
- What are $\alpha$ and $E$ in plain words?
- How would you use small runs to choose hyperparameters for a big one? *(Fit scaling curves, plus µP-style parameterisations so that hyperparameters transfer across sizes.)*
:::
::::

::::exercise{id=compute-optimal type=implement difficulty=3 time=20 runtime=browser udl="§20.4" tags="scaling"}
# The compute-optimal model size

:::prereq
Minimise $L(N, D) = E + A N^{-\alpha} + B D^{-\beta}$ subject to $6ND = C$. Substituting $D = C/(6N)$ and setting $dL/dN = 0$:
$$\alpha A N^{-\alpha} = \beta B D^{-\beta}\quad\Longrightarrow\quad N_\text{opt} = G\left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}},\qquad G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}},$$
and $D_\text{opt} = C/(6N_\text{opt})$. So the optimum balances the *marginal* returns of parameters and data.

Chinchilla's published fit is $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$.
:::

:::task
Implement `chinchilla_optimal(C, A=406.4, B=410.7, alpha=0.34, beta=0.28)`, returning `(N_opt, D_opt)`, and `loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)`.
:::

:::starter{lang=python}
import numpy as np

def loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    raise NotImplementedError

def chinchilla_optimal(C, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    return E + A / N ** alpha + {{ B / D ** beta | B * D ** -beta | B / (D ** beta) ? The data term mirrors the parameter term, with its own coefficient and exponent. }}

def chinchilla_optimal(C, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    G = {{ (alpha * A / (beta * B)) ** (1 / (alpha + beta)) | ((alpha * A) / (beta * B)) ** (1 / (alpha + beta)) ? From $\alpha A N^{-\alpha} = \beta B D^{-\beta}$: collect the constants and take the $1/(\alpha+\beta)$ power. }}
    N = G * (C / 6) ** {{ (beta / (alpha + beta)) ? The optimal model size grows as compute to the power $\beta/(\alpha+\beta)$, about 0.45 for these numbers. Keep the parentheses: `**` binds tighter than `/`. }}
    return N, {{ C / (6 * N) | C / 6 / N ? Whatever compute isn't spent on parameters goes on tokens: $D = C/(6N)$. }}
:::

:::solution{lang=python}
import numpy as np

def loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    return E + A / N ** alpha + B / D ** beta

def chinchilla_optimal(C, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)
:::

:::explain
[L3-4] **The parametric loss.** Three terms: what no model can remove, what a bigger model removes, and what more data removes.

[L6-9] **The closed-form optimum.** The tests check the budget constraint, that the loss along the constraint really is minimised (nudging $N$ either way makes it worse), and the doubling ratio $2^{\beta/(\alpha+\beta)}$ used in the next exercise.

**A sanity check worth memorising.** At Chinchilla's own scale ($C \approx 5.8\times10^{23}$), this gives about 32B parameters and 3.0T tokens. That's about 90 tokens per parameter, not 20. The paper's *other* estimation methods give roughly 70B parameters and 1.4T tokens (20 tokens per parameter), which is the model they trained. The parametric fit (approach 3) is the one most at odds with the others, so quote "about 20 tokens per parameter" as the headline. Modern models often go far beyond it on purpose: they train *past* compute-optimal, because smaller models are cheaper to serve.
:::

:::tests{lang=python}
import numpy as np

def test_budget_is_respected():
    for C in (1e20, 1e22, 5.8e23):
        N, D = chinchilla_optimal(C)
        assert np.isclose(6 * N * D, C)

def test_optimum_beats_neighbours_on_the_constraint():
    C = 1e22
    N, D = chinchilla_optimal(C)
    for f in (0.7, 1.4):
        assert loss(N, D) < loss(N * f, C / (6 * N * f))

def test_doubling_ratio():
    n1, _ = chinchilla_optimal(1e21)
    n2, _ = chinchilla_optimal(2e21)
    assert np.isclose(n2 / n1, 2 ** (0.28 / 0.62))
:::

:::hints
1. Substitute $D = C/(6N)$ and differentiate with respect to $N$. Or start from the balance condition in the prerequisites.
2. Check $6ND = C$ at the end.
:::

:::interview
- Derive the compute-optimal allocation. Why do the parameters and tokens both grow?
- What does $C \approx 6ND$ count? *(2 FLOPs per parameter per token forward, 4 backward.)*
- Why would you deliberately train a model past compute-optimal? *(Inference cost: a smaller model trained longer is cheaper to serve.)*
:::
::::

::::exercise{id=doubling-compute-predict type=predict difficulty=1 time=5 runtime=browser udl="§20.4" tags="scaling"}
# Double the compute budget

:::task
You have a compute-optimal training recipe. Your budget **doubles**. Following Chinchilla-style scaling, by roughly what factor should the **model size** grow?
:::

:::choices
- [ ] 2×: spend all of the new compute on parameters
- [x] About 1.4×, with the data growing by about 1.4× as well
- [ ] 1×: keep the model and train it on twice the data
- [ ] 4×
:::

:::experiment{lang=python}
import numpy as np

A, B, alpha, beta = 406.4, 410.7, 0.34, 0.28

def optimal(C):
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)

for C in (1e21, 2e21, 4e21):
    N, D = optimal(C)
    print(f"C = {C:.0e}:  N = {N / 1e9:6.2f}B params   D = {D / 1e9:7.1f}B tokens   tokens/param = {D / N:5.1f}")
n1, d1 = optimal(1e21)
n2, d2 = optimal(2e21)
print(f"\ndoubling compute: N x{n2 / n1:.3f}, D x{d2 / d1:.3f}   (equal split would be x{np.sqrt(2):.3f} each)")
:::

:::tests{lang=python}
import numpy as np

def test_both_grow_by_about_root_two():
    n1, d1 = optimal(1e21)
    n2, d2 = optimal(2e21)
    assert 1.3 < n2 / n1 < 1.5 and 1.3 < d2 / d1 < 1.6 and np.isclose((n2 / n1) * (d2 / d1), 2)
:::

:::explain
With $C \propto ND$ and an optimum that scales both, $N_\text{opt}\propto C^{a}$ and $D_\text{opt}\propto C^{1-a}$. Chinchilla's headline estimate is $a \approx 0.5$: parameters and tokens grow **equally**, so doubling compute means about $\sqrt2 \approx 1.41\times$ each. The published parametric fit, used in the experiment, gives $a = \beta/(\alpha+\beta) \approx 0.45$. That's 1.37× the parameters and 1.46× the tokens. Either way, you don't put all the new compute into the model.

**History.** Kaplan et al. (2020) estimated $a \approx 0.73$, meaning "mostly grow the model". That led to large, undertrained models (GPT-3 saw about 1.7 tokens per parameter). Chinchilla's correction, mostly from tuning the learning-rate schedule to each run's length, showed that a 4× smaller model trained on 4× more data beat Gopher at the same compute.
:::

:::interview
- Kaplan versus Chinchilla scaling: what changed, and why? *(Learning-rate schedules matched to the run length, plus different model-size ranges and counting conventions.)*
- If compute doubles, how should you split it?
- What limits scaling laws in practice? *(Running out of data, data quality, and inference cost.)*
:::
::::
