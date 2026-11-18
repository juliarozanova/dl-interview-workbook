---
id: ch02-linear-models
summary: Linear regression is where every idea in this workbook first appears in miniature, including the loss, the gradient, the closed form, regularisation and gradient descent. Derive the normal equations, fit least squares and ridge in closed form, then reach the same answer by gradient descent.
udl: "Ch. 2 — Supervised learning"
---

## The supervised-learning recipe

UDL Chapter 2 frames every supervised model the same way:

1. A **model** $f(x;\phi)$ with parameters $\phi$.
2. A **loss** $L(\phi)$ measuring how badly the model fits the training pairs $\{(x_i, y_i)\}$.
3. **Training**: find $\hat\phi = \arg\min_\phi L(\phi)$.
4. **Testing**: check how well $f(x;\hat\phi)$ does on data it hasn't seen.

For a linear model, $f(x) = w^\top x + b$ and the loss is the mean squared error. That combination is special: the minimiser has a **closed form**. Deep networks lose that luxury, which is why the rest of this workbook is about gradients.

::::exercise{id=normal-equations type=derive difficulty=1 time=6 runtime=browser udl="§2.2" tags="linear-models,regularisation"}
# Derive the normal equations, and ridge

:::prereq
- **Matrix form.** Stack inputs as rows of $X\in\mathbb R^{N\times D}$ and targets in $y\in\mathbb R^N$. The residual vector is $Xw - y$, and the sum of squared errors is $\|Xw - y\|^2 = (Xw - y)^\top (Xw-y)$.
- **Two gradient identities.** $\nabla_w\, (a^\top w) = a$, and $\nabla_w\, (w^\top A w) = (A + A^\top)w$, which is $2Aw$ for symmetric $A$.
- **Bias.** Absorb $b$ by appending a column of ones to $X$, or centre the data (Exercise 2.2).
:::

:::task
Derive the least-squares solution $w^\star$ for $L(w) = \|Xw - y\|^2$. Then derive the **ridge** solution for $L(w) = \|Xw - y\|^2 + \lambda\|w\|^2$, and say why ridge is always solvable while plain least squares may not be.
:::

:::scaffold
Write transposes as `X^T` or `X^\top`, and λ as `λ`, `\lambda` or `lambda`.

**1. Gradient.** Expanding $\|Xw-y\|^2 = w^\top X^\top X w - 2y^\top X w + y^\top y$ gives $\nabla_w L =$ {{ 2X^T(Xw - y) | 2X^\top(Xw - y) | 2X^TXw - 2X^Ty | re:^2x\^\{?(t|\\top)\}?\(xw-y\)$ ? Use $\nabla_w(w^\top A w) = 2Aw$ with $A = X^\top X$, and $\nabla_w(c^\top w) = c$ with $c = -2X^\top y$. Then factor out $2X^\top$. }}

**2. Normal equations.** Setting the gradient to zero gives {{ X^TXw = X^Ty | X^\top X w = X^\top y | re:^x\^\{?(t|\\top)\}?xw=x\^\{?(t|\\top)\}?y$ ? Divide by 2 and move $X^\top y$ to the right-hand side. These are called *normal* equations because the residual ends up orthogonal (normal) to the columns of $X$. }}

**3. Solution.** If $X^\top X$ is invertible, $w^\star =$ {{ (X^TX)^{-1}X^Ty | (X^\top X)^{-1} X^\top y | re:^\(x\^\{?(t|\\top)\}?x\)\^\{?-1\}?x\^\{?(t|\\top)\}?y$ ? Multiply both sides of the normal equations by $(X^\top X)^{-1}$ on the left. }}

**4. Ridge.** The penalty adds $2\lambda w$ to the gradient, so $w^\star_{\text{ridge}} =$ {{ (X^TX + λI)^{-1}X^Ty | (X^\top X + \lambda I)^{-1} X^\top y | re:^\(x\^\{?(t|\\top)\}?x\+(λ|\\lambda|lambda|lam)i\)\^\{?-1\}?x\^\{?(t|\\top)\}?y$ ? $\nabla_w \lambda\|w\|^2 = 2\lambda w$. Collect the terms in $w$: $(X^\top X + \lambda I)w = X^\top y$. }}
:::

:::derivation
**Expand the loss.**
$$L(w) = (Xw-y)^\top(Xw-y) = w^\top X^\top X w - 2\,y^\top X w + y^\top y.$$

---step---

**Differentiate.** The quadratic term gives $2X^\top X w$, the linear term gives $-2X^\top y$, and the constant vanishes:
$$\nabla_w L = 2X^\top X w - 2X^\top y = 2X^\top (Xw - y).$$

---step---

**Set it to zero.** This gives the normal equations, $X^\top X\, w = X^\top y$. Geometrically, the residual $y - Xw^\star$ is orthogonal to every column of $X$: the fitted vector $Xw^\star$ is the **projection** of $y$ onto the column space.

When $X^\top X$ is invertible (that is, when $X$ has full column rank), $w^\star = (X^\top X)^{-1}X^\top y$.

---step---

**Ridge.** The extra term $\lambda\|w\|^2$ contributes $2\lambda w$ to the gradient, so
$$(X^\top X + \lambda I)\, w = X^\top y \quad\Longrightarrow\quad w^\star = (X^\top X + \lambda I)^{-1} X^\top y.$$

---step---

**Why ridge is always solvable.** $X^\top X$ is positive *semi*-definite, so its eigenvalues are $\ge 0$, and some are exactly $0$ when $D > N$ or features are collinear. Adding $\lambda I$ shifts every eigenvalue up by $\lambda > 0$, which makes the matrix invertible and better conditioned.

In practice you **never form the inverse**: call `np.linalg.solve`, or `lstsq`, which uses a QR or SVD factorisation.
:::

:::solution{lang=python}
import numpy as np

def normal_equations(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

def ridge_closed_form(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
:::

:::explain
[L4] **`solve`, not `inv`.** `np.linalg.solve(A, b)` factorises $A$ once and back-substitutes. That's faster and more accurate than `np.linalg.inv(A) @ b`.

[L7] **Ridge.** This is the same system with $\lambda$ added to the diagonal. The tests check the *defining* property: the gradient of the ridge loss is zero at the returned $w$.
:::

:::tests{lang=python}
import numpy as np

def test_matches_lstsq():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 3)), rng.normal(size=40)
    assert np.allclose(normal_equations(X, y), np.linalg.lstsq(X, y, rcond=None)[0])

def test_ridge_gradient_is_zero():
    rng = np.random.default_rng(1)
    X, y, lam = rng.normal(size=(30, 4)), rng.normal(size=30), 2.5
    w = ridge_closed_form(X, y, lam)
    assert np.allclose(2 * X.T @ (X @ w - y) + 2 * lam * w, 0, atol=1e-9)

def test_ridge_works_when_d_exceeds_n():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(5, 20)), rng.normal(size=5)
    assert np.all(np.isfinite(ridge_closed_form(X, y, 1e-2)))
:::

:::hints
1. Expand the squared norm into three terms before differentiating.
2. $X^\top X$ is symmetric, so the derivative of $w^\top X^\top X w$ is $2X^\top Xw$.
:::

:::interview
- When is $X^\top X$ singular? *(When $D > N$ or the features are collinear.)* What do you do then? *(Use ridge, or the pseudo-inverse, which gives the minimum-norm solution.)*
- What's the cost of the closed form in $N$ and $D$? *($O(ND^2 + D^3)$.)* When would you prefer gradient descent?
- What's the probabilistic reading of ridge? *(MAP estimation with a Gaussian prior on $w$.)*
:::
::::

::::exercise{id=least-squares-ridge type=implement difficulty=2 time=15 runtime=browser udl="§2.2, §9.1" tags="linear-models,regularisation"}
# Least squares and ridge with an intercept

:::prereq
**Intercept via augmentation.** Append a column of ones, $A = [X \mid \mathbf 1]$, so that $A\,[w; b] = Xw + b$. Plain least squares can then solve for $w$ and $b$ together.

**But don't penalise the intercept.** Ridge shrinks weights towards $0$, and shrinking $b$ would mean the model's predictions depend on where you put the origin of $y$. The standard fix is to **centre** the data:

1. Subtract the means $\bar x$ and $\bar y$.
2. Solve ridge *without* an intercept on the centred data.
3. Recover $b = \bar y - \bar x^\top w$.
:::

:::task
Implement:

- `fit_least_squares(X, y)` → `(w, b)`, by augmenting with a ones column and calling `np.linalg.lstsq`;
- `fit_ridge(X, y, lam)` → `(w, b)`, by centring, solving $(X_c^\top X_c + \lambda I)w = X_c^\top y_c$, and recovering $b$.

`X` has shape `(N, D)` and `y` has shape `(N,)`. Return `w` with shape `(D,)` and `b` as a float.
:::

:::starter{lang=python}
import numpy as np

def fit_least_squares(X, y):
    """X: (N, D), y: (N,) -> (w: (D,), b: float). Unregularised, with intercept."""
    raise NotImplementedError

def fit_ridge(X, y, lam):
    """X: (N, D), y: (N,), lam >= 0 -> (w: (D,), b: float). The intercept is NOT penalised."""
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def fit_least_squares(X, y):
    A = np.hstack([X, {{ np.ones((len(X), 1)) | np.ones((X.shape[0], 1)) | np.ones((len(y), 1)) | np.ones_like(X[:, :1]) ? You need one extra **column** of ones, shape `(N, 1)`, so that $A\,[w; b] = Xw + b$. }}])
    theta = np.linalg.lstsq(A, y, rcond=None)[0]
    return {{ theta[:-1], theta[-1] | theta[:-1], float(theta[-1]) ? The ones column was appended **last**, so its coefficient (the intercept) is the last entry. }}

def fit_ridge(X, y, lam):
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = {{ X - x_mean, y - y_mean ? Centre both inputs and targets, so that the optimal intercept on the centred problem is exactly 0. }}
    w = np.linalg.solve(Xc.T @ Xc + {{ lam * np.eye(X.shape[1]) | lam * np.eye(Xc.shape[1]) | lam * np.identity(X.shape[1]) | np.eye(X.shape[1]) * lam ? The penalty adds $\lambda$ to the diagonal: a $D\times D$ identity, one entry per **weight**. }}, Xc.T @ yc)
    return w, {{ y_mean - x_mean @ w | y_mean - w @ x_mean | float(y_mean - x_mean @ w) | y_mean - np.dot(x_mean, w) ? On the original scale, the fitted line must pass through the point of means $(\bar x, \bar y)$. }}
:::

:::solution{lang=python}
import numpy as np

def fit_least_squares(X, y):
    A = np.hstack([X, np.ones((len(X), 1))])
    theta = np.linalg.lstsq(A, y, rcond=None)[0]
    return theta[:-1], theta[-1]

def fit_ridge(X, y, lam):
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
    return w, y_mean - x_mean @ w
:::

:::explain
[L4-5] **Augment and solve.** `lstsq` uses an SVD, so it copes with rank-deficient `A` by returning the minimum-norm solution. Never write `inv(A.T @ A)`: forming $A^\top A$ *squares* the condition number.

[L9-10] **Centre.** After centring, every feature column has mean zero, so the intercept decouples from the weights.

[L11] **Ridge on the centred problem.** This is a $D\times D$ solve. The intercept is outside the penalty, so `fit_ridge(X, y + 100, lam)` changes only `b`, and the tests check exactly that.

[L12] **Recover the intercept.** The least-squares line passes through $(\bar x, \bar y)$, which gives $b = \bar y - \bar x^\top w$.
:::

:::tests{lang=python}
import numpy as np

def test_recovers_true_parameters_without_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    w_true, b_true = np.array([1.5, -2.0, 0.5]), 0.7
    w, b = fit_least_squares(X, X @ w_true + b_true)
    assert np.allclose(w, w_true) and np.isclose(b, b_true)

def test_ridge_with_zero_lambda_is_least_squares():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(40, 4)), rng.normal(size=40)
    (w1, b1), (w2, b2) = fit_least_squares(X, y), fit_ridge(X, y, 0.0)
    assert np.allclose(w1, w2) and np.isclose(b1, b2)

def test_ridge_shrinks_weights_but_not_intercept():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    y = X @ np.array([3.0, -1.0, 2.0]) + 5.0 + 0.1 * rng.normal(size=60)
    w_small, _ = fit_ridge(X, y, 0.1)
    w_big, b_big = fit_ridge(X, y, 1e6)
    assert np.linalg.norm(w_big) < 0.01 * np.linalg.norm(w_small)
    assert np.isclose(b_big, y.mean(), atol=1e-2)

def test_shifting_y_only_moves_the_intercept():
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(30, 2)), rng.normal(size=30)
    (w1, b1), (w2, b2) = fit_ridge(X, y, 3.0), fit_ridge(X, y + 100.0, 3.0)
    assert np.allclose(w1, w2) and np.isclose(b2 - b1, 100.0)
:::

:::hints
1. `np.hstack` joins arrays along columns. Both pieces need two dimensions.
2. For ridge, centre first. The intercept then falls out at the end from the means.
:::

:::interview
- Why not penalise the intercept? *(The fit would depend on the origin of $y$. Shifting all targets by a constant should only shift $b$.)*
- Should you standardise features before ridge? *(Yes. The penalty treats all weights equally, so the feature scales matter.)*
- What happens to ridge as $\lambda \to \infty$? *($w \to 0$, and predictions tend to $\bar y$.)*
:::
::::

## The same answer, by gradient descent

For models without a closed form, which means every neural network, you follow the negative gradient. Linear regression is the perfect test bed, because you know the exact answer to compare against.

::::exercise{id=gd-linear type=implement difficulty=1 time=12 runtime=browser udl="§6.1" tags="linear-models,optimisers"}
# Gradient descent for linear regression

:::prereq
For the **mean** squared error $L(w,b) = \frac1N\sum_i (w^\top x_i + b - y_i)^2$, with residuals $r = Xw + b - y$:
$$\nabla_w L = \frac{2}{N}X^\top r, \qquad \frac{\partial L}{\partial b} = \frac{2}{N}\sum_i r_i .$$
One gradient-descent step is $w \leftarrow w - \eta\,\nabla_w L$, and the same for $b$. With a small enough learning rate $\eta$, the loss decreases at every step.
:::

:::task
Implement `gd_linear(X, y, lr=0.1, steps=500)`. Start from $w = 0$ and $b = 0$, run full-batch gradient descent on the MSE, and return `(w, b, losses)`, where `losses[t]` is the loss *before* step `t`.
:::

:::starter{lang=python}
import numpy as np

def gd_linear(X, y, lr=0.1, steps=500):
    """X: (N, D), y: (N,) -> (w: (D,), b: float, losses: list of floats, length `steps`)."""
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def gd_linear(X, y, lr=0.1, steps=500):
    N, D = X.shape
    w, b, losses = np.zeros(D), 0.0, []
    for _ in range(steps):
        r = {{ X @ w + b - y | X.dot(w) + b - y | (X @ w + b) - y ? The residual vector: predictions minus targets, shape `(N,)`. }}
        losses.append(float((r ** 2).mean()))
        gw = {{ 2 / N * X.T @ r | 2 * X.T @ r / N | (2 / N) * X.T @ r | 2 * (X.T @ r) / N | 2 / N * (X.T @ r) ? $\frac2N X^\top r$: each weight's gradient correlates its feature with the residuals. }}
        gb = {{ 2 * r.mean() | 2 / N * r.sum() | 2 * np.mean(r) | 2 * r.sum() / N ? The bias acts like a feature that is always 1, so its gradient is $\frac2N\sum_i r_i$. }}
        w, b = {{ w - lr * gw, b - lr * gb ? Step *against* the gradient, scaled by the learning rate. }}
    return w, b, losses
:::

:::solution{lang=python}
import numpy as np

def gd_linear(X, y, lr=0.1, steps=500):
    N, D = X.shape
    w, b, losses = np.zeros(D), 0.0, []
    for _ in range(steps):
        r = X @ w + b - y
        losses.append(float((r ** 2).mean()))
        gw = 2 / N * X.T @ r
        gb = 2 * r.mean()
        w, b = w - lr * gw, b - lr * gb
    return w, b, losses
:::

:::explain
[L7] **Residuals first.** Every gradient below reuses `r`. Computing it once is exactly what backprop does at scale.

[L9-10] **Gradients.** These are the formulas from the prerequisites. A useful check: at the optimum, `X.T @ r` is zero, which is the normal equations from Exercise 2.1.

[L11] **Simultaneous update.** Both gradients come from the *same* `r`. Updating `w` and then recomputing the residual before updating `b` would be a different algorithm (coordinate descent).

**Learning rate.** For this quadratic loss, the step is stable when $\eta < 2/\lambda_{\max}$, where $\lambda_{\max}$ is the largest eigenvalue of the Hessian, $\frac{2}{N}X^\top X$ (with the bias column included). Chapter 07 shows what happens just above that threshold.
:::

:::tests{lang=python}
import numpy as np

def _data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(100, 3))
    return X, X @ np.array([2.0, -1.0, 0.5]) + 1.0 + 0.1 * rng.normal(size=100)

def test_reaches_closed_form():
    X, y = _data()
    w, b, _ = gd_linear(X, y, lr=0.1, steps=2000)
    theta = np.linalg.lstsq(np.hstack([X, np.ones((100, 1))]), y, rcond=None)[0]
    assert np.allclose(w, theta[:-1], atol=1e-4) and np.isclose(b, theta[-1], atol=1e-4)

def test_loss_never_increases_for_small_lr():
    X, y = _data(1)
    _, _, losses = gd_linear(X, y, lr=0.05, steps=200)
    assert len(losses) == 200 and all(b <= a + 1e-12 for a, b in zip(losses, losses[1:]))
:::

:::hints
1. Compute the residual once per step and reuse it for both gradients.
2. Mean versus sum matters: with the mean, the gradient carries a factor of $1/N$.
:::

:::interview
- What learning rate guarantees convergence for least squares? *($\eta < 2/\lambda_{\max}$ of the Hessian.)*
- Why does feature scaling speed up gradient descent? *(It improves the Hessian's condition number.)*
- Full-batch versus stochastic gradient descent: what changes, and why does SGD still work?
:::
::::
