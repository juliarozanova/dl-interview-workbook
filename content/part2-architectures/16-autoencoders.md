---
id: ch16-autoencoders
summary: Representation learning starts with compression. Implement PCA with an SVD, derive why a linear autoencoder can only ever learn the PCA subspace, then train a nonlinear autoencoder that beats PCA on curved data with the same size of code.
udl: "Ch. 14 — Unsupervised learning"
---

## Learning a code

An **autoencoder** maps $x$ to a low-dimensional code $z = \mathrm{enc}(x)$ and back to $\hat x = \mathrm{dec}(z)$, trained to make $\hat x \approx x$. The **bottleneck** forces the code to keep only what matters for reconstruction. That makes it a learned compression, and the simplest instance of representation learning.

- **Linear** encoder and decoder with squared error: this recovers **PCA**, the best rank-$k$ linear subspace (exercise 16.2).
- **Nonlinear** networks can follow **curved** manifolds that no linear subspace fits (exercise 16.3).
- **Embeddings** (word vectors, token embeddings, contrastive encoders in Chapter 21) are the same idea with a different training signal.

UDL Chapter 14 surveys unsupervised learning. Its generative relatives, the VAE (Chapter 17 here) and diffusion (Chapter 19), add probability to the same encode/decode picture.

::::exercise{id=pca-svd type=implement difficulty=1 time=12 runtime=browser udl="§14.1" tags="autoencoders"}
# PCA with an SVD

:::prereq
For centred data $X_c = X - \bar x$ (shape $N\times D$), the SVD $X_c = U S V^\top$ gives everything at once:

- **Principal directions:** the rows of $V^\top$, orthonormal and sorted by importance.
- **Variances along them:** $S^2/(N-1)$, which are the eigenvalues of the sample covariance.
- **Code and reconstruction:** $z = X_cV_k$, and $\hat x = \bar x + zV_k^\top$, keeping the top $k$ directions.

Using the SVD avoids forming $X_c^\top X_c$, which would square the condition number. It's the same lesson as `lstsq` in Chapter 02.
:::

:::task
Implement `pca(X, k)`, returning `(mean, components (k, D), explained_variance (k,))`, and `reconstruct(X, mean, components)`, returning the projection back in the original space.
:::

:::starter{lang=python}
import numpy as np

def pca(X, k):
    raise NotImplementedError

def reconstruct(X, mean, components):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def pca(X, k):
    mean = X.mean(axis=0)
    U, S, Vt = {{ np.linalg.svd(X - mean, full_matrices=False) | np.linalg.svd(X - mean, full_matrices=0) ? Decompose the **centred** data. The thin SVD (`full_matrices=False`) skips the huge unused part of `U`. }}
    return mean, {{ Vt[:k] | Vt[:k, :] ? The principal directions are the rows of $V^\top$, already sorted by singular value. }}, {{ S[:k] ** 2 / (len(X) - 1) | S[:k]**2 / (X.shape[0] - 1) | S[:k] ** 2 / (X.shape[0] - 1) ? The variance along direction $i$ is $s_i^2/(N-1)$, matching `np.cov`'s normalisation. }}

def reconstruct(X, mean, components):
    Z = {{ (X - mean) @ components.T | np.dot(X - mean, components.T) ? Encode: project the centred data onto each direction, giving `(N, D) @ (D, k)`. }}
    return Z @ components + mean
:::

:::solution{lang=python}
import numpy as np

def pca(X, k):
    mean = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return mean, Vt[:k], S[:k] ** 2 / (len(X) - 1)

def reconstruct(X, mean, components):
    Z = (X - mean) @ components.T
    return Z @ components + mean
:::

:::explain
[L4-5] **Centre, then decompose.** Forgetting to centre is the classic PCA bug: the first "component" then points at the mean rather than along the direction of greatest spread.

[L6] **Components and variances.** NumPy returns singular values in descending order, so slicing the first $k$ is enough.

[L9-10] **Encode and decode.** These are two matmuls with the *same* matrix, transposed. The tests check that reconstruction is exact at $k = D$, that its error shrinks as $k$ grows, and that data lying on a 2-D plane in 5-D is recovered with $k = 2$.
:::

:::tests{lang=python}
import numpy as np

def test_components_orthonormal_and_variances_match_cov():
    X = np.random.default_rng(0).normal(size=(200, 6)) @ np.random.default_rng(1).normal(size=(6, 6))
    mean, C, var = pca(X, 4)
    assert C.shape == (4, 6) and np.allclose(C @ C.T, np.eye(4))
    assert np.allclose(var, np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1][:4])

def test_reconstruction_error_falls_with_k_and_is_exact_at_D():
    X = np.random.default_rng(2).normal(size=(100, 5)) * np.array([5, 3, 2, 1, 0.5])
    errs = [((reconstruct(X, *pca(X, k)[:2]) - X) ** 2).mean() for k in range(1, 6)]
    assert all(b < a for a, b in zip(errs, errs[1:])) and errs[-1] < 1e-20

def test_recovers_a_plane():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 2)) @ rng.normal(size=(2, 5)) + 7.0
    mean, C, _ = pca(X, 2)
    assert np.allclose(reconstruct(X, mean, C), X)
:::

:::hints
1. Centre first, and keep the mean: `reconstruct` adds it back.
2. Which factor of the SVD holds the directions in feature space? Its shape must be `(·, D)`.
:::

:::interview
- Why centre before PCA? What happens if you don't?
- Why compute PCA via the SVD rather than the eigendecomposition of $X^\top X$?
- Should you standardise features first? *(Yes, if they're on different scales. PCA is scale-sensitive.)*
- How do you choose $k$? *(Explained-variance ratio, a downstream task, or a knee in the spectrum.)*
:::
::::

::::exercise{id=linear-ae-pca type=derive difficulty=2 time=8 runtime=browser udl="§14.1" tags="autoencoders"}
# A linear autoencoder learns PCA

:::prereq
- **Projection.** For a matrix $U_k \in\mathbb R^{D\times k}$ with orthonormal columns, $U_kU_k^\top$ projects onto their span, and $\|x\|^2 = \|U_kU_k^\top x\|^2 + \|(I - U_kU_k^\top)x\|^2$ (Pythagoras).
- **Covariance.** $\Sigma = \frac1N\sum_i (x_i - \bar x)(x_i-\bar x)^\top$, with eigenvalues $\lambda_1 \ge\dots\ge\lambda_D$.
- **Rayleigh.** $\sum_{j\le k} u_j^\top\Sigma u_j$ over orthonormal $u_j$ is maximised by the top-$k$ eigenvectors (Ky Fan).
:::

:::task
Consider a linear autoencoder with code size $k$: $\hat x = \bar x + U_k U_k^\top(x - \bar x)$, with $U_k^\top U_k = I$. Show that minimising the mean squared reconstruction error picks the **top-$k$ principal subspace**, and that the minimum error is the **sum of the discarded eigenvalues**. Then say what changes if the weights aren't constrained to be orthonormal.
:::

:::scaffold
Write $\tilde x = x - \bar x$.

**1. Pythagoras.** $\hat x - \bar x = U_kU_k^\top\tilde x$ is the orthogonal {{ projection | projection of x onto the span of U_k | re:project ? $U_kU_k^\top$ is idempotent and symmetric: it's the orthogonal projector onto the column space of $U_k$. }} of $\tilde x$, so the error is $\|\tilde x\|^2 - \|U_k^\top\tilde x\|^2$.

**2. The objective.** $\|\tilde x\|^2$ doesn't depend on $U_k$, so minimising the error is the same as maximising the {{ variance | projected variance | captured variance | re:variance ? Averaged over the data, $\mathbb E\|U_k^\top\tilde x\|^2 = \sum_j u_j^\top\Sigma u_j$: the variance captured by the subspace. }} captured.

**3. The optimum.** That's maximised by the top-$k$ {{ eigenvectors | eigenvectors of the covariance | principal components | re:(eigenvector|principal) ? Ky Fan's theorem: the sum of $k$ Rayleigh quotients over orthonormal vectors is largest for the top-$k$ eigenvectors of $\Sigma$. }} of $\Sigma$.

**4. The minimum error.** The mean squared error is then $\sum_{j>k}\lambda_j$: the sum of the {{ discarded eigenvalues | remaining eigenvalues | smallest eigenvalues | re:(discard|remain|smallest|bottom|trailing|lowest|unused|left).*eigen ? Total variance is $\sum_j \lambda_j$, and the subspace keeps $\sum_{j\le k}\lambda_j$. What's left over is the error. }}.

**5. Unconstrained weights.** A linear AE trained by gradient descent, with any encoder and decoder matrices, reaches the same {{ subspace | span | principal subspace | re:(subspace|span) ? Any invertible $k\times k$ mixing $A$ gives the same reconstruction: $(W A)(A^{-1}V)$. Only the span is identified. }}, but not necessarily the same basis: the code is the principal components up to an invertible linear transform.
:::

:::derivation
**Write the error with Pythagoras.** With $P = U_kU_k^\top$ the orthogonal projector,
$$\|\tilde x - P\tilde x\|^2 = \|\tilde x\|^2 - \|P\tilde x\|^2 = \|\tilde x\|^2 - \|U_k^\top \tilde x\|^2 .$$

---step---

**Average over the data.** $\mathbb E\|U_k^\top\tilde x\|^2 = \mathrm{tr}(U_k^\top \Sigma U_k) = \sum_{j=1}^k u_j^\top\Sigma u_j$. So
$$\text{MSE}(U_k) = \mathrm{tr}(\Sigma) - \mathrm{tr}(U_k^\top\Sigma U_k).$$

---step---

**Maximise the captured variance.** By Ky Fan (or by induction on Rayleigh quotients), $\mathrm{tr}(U_k^\top\Sigma U_k) \le \sum_{j\le k}\lambda_j$, with equality when $U_k$ spans the top-$k$ eigenvectors. So the optimal subspace is the **PCA subspace**, and
$$\min_{U_k}\text{MSE} = \sum_{j=1}^D\lambda_j - \sum_{j=1}^k\lambda_j = \sum_{j>k}\lambda_j .$$

---step---

**Drop the orthonormality.** A general linear AE $\hat x = \bar x + W_d W_e\tilde x$ has a rank-$\le k$ product $W_dW_e$. The best rank-$k$ approximation (Eckart–Young) is the same projector. So gradient descent converges to the PCA **subspace**, though the code $W_e\tilde x$ can be any invertible mix of the principal components. That's why linear-autoencoder features aren't "the" PCA axes.

---step---

**Why go nonlinear?** A linear map can only produce flat subspaces. Data on a curved manifold (a circle, a swiss roll, natural images) needs a nonlinear encoder and decoder to compress well. That's the next exercise.
:::

:::solution{lang=python}
import numpy as np

def recon_error(X, k):
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    R = Xc - Xc @ Vt[:k].T @ Vt[:k]
    return (R ** 2).sum(axis=1).mean()

def discarded_eigenvalues(X, k):
    lam = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False, bias=True)))[::-1]
    return lam[k:].sum()
:::

:::explain
[L3-7] **The reconstruction error.** This is the mean over examples of $\|\tilde x - P\tilde x\|^2$, using the top-$k$ right singular vectors.

[L9-11] **The predicted error.** This is the sum of the eigenvalues past $k$. `bias=True` uses $1/N$ to match the mean in `recon_error`. With $1/(N-1)$ the two differ by exactly that factor. The tests check that the two functions agree for every $k$.
:::

:::tests{lang=python}
import numpy as np

def test_error_equals_discarded_eigenvalues():
    X = np.random.default_rng(0).normal(size=(150, 6)) @ np.random.default_rng(1).normal(size=(6, 6))
    for k in range(7):
        assert np.isclose(recon_error(X, k), discarded_eigenvalues(X, k))
:::

:::hints
1. Split $\|\tilde x\|^2$ into the part inside the subspace and the part orthogonal to it.
2. The part inside is a sum of Rayleigh quotients. Which vectors maximise those?
:::

:::interview
- Prove that a linear autoencoder with squared loss recovers the PCA subspace.
- Why isn't its code necessarily equal to the principal components?
- What's the minimum reconstruction error for rank $k$? *(The sum of the discarded eigenvalues.)*
:::
::::

::::exercise{id=nonlinear-autoencoder type=implement difficulty=2 time=20 runtime=local-torch udl="§14.1" tags="autoencoders,pytorch,training-loop"}
# A nonlinear autoencoder that beats PCA

:::prereq
**Architecture.** The encoder is $D \to h \to k$ (with tanh in between) and the decoder is the mirror image, $k \to h \to D$. There's no activation on the code or on the output: those are real-valued.

**Training.** Minimise the mean squared reconstruction error with Adam. This is full-batch here, because the dataset is tiny.

**The data.** A 2-D sheet bent through a tanh into 10-D. It's curved, so no single linear direction captures it well, but a 1-D nonlinear code can trace most of it.
:::

:::task
Implement `AutoEncoder(d_in, d_code, d_hidden)` with `encode`, `decode` and `forward`, plus `train_autoencoder(model, X, steps=1500, lr=1e-2)`, which returns the final training MSE as a float. The tests compare a **1-D code** against **1-component PCA** on the curved data.
:::

:::starter{lang=python}
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, d_in, d_code, d_hidden):
        super().__init__()
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

def train_autoencoder(model, X, steps=1500, lr=1e-2):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, d_in, d_code, d_hidden):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hidden), nn.Tanh(), {{ nn.Linear(d_hidden, d_code) ? The bottleneck: map the hidden layer down to the code size, with no activation, so the code is an unconstrained real vector. }})
        self.dec = nn.Sequential(nn.Linear(d_code, d_hidden), nn.Tanh(), nn.Linear(d_hidden, d_in))

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        return {{ self.decode(self.encode(x)) | self.dec(self.enc(x)) ? Reconstruction is decode ∘ encode. }}

def train_autoencoder(model, X, steps=1500, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = {{ ((model(X) - X) ** 2).mean() | torch.nn.functional.mse_loss(model(X), X) | nn.functional.mse_loss(model(X), X) | ((X - model(X)) ** 2).mean() ? The target is the input itself: mean squared reconstruction error. }}
        loss.backward()
        opt.step()
    return loss.item()
:::

:::solution{lang=python}
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, d_in, d_code, d_hidden):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hidden), nn.Tanh(), nn.Linear(d_hidden, d_code))
        self.dec = nn.Sequential(nn.Linear(d_code, d_hidden), nn.Tanh(), nn.Linear(d_hidden, d_in))

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        return self.decode(self.encode(x))

def train_autoencoder(model, X, steps=1500, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(X) - X) ** 2).mean()
        loss.backward()
        opt.step()
    return loss.item()
:::

:::explain
[L7-8] **Mirror-image encoder and decoder.** The hidden tanh layers are what let the code bend. Remove them, and you're back to the linear autoencoder, which can do no better than PCA (Exercise 16.2).

[L16-17] **Reconstruction.** Keeping `encode` and `decode` as separate methods means the code is easy to read off for downstream use (visualisation, retrieval, or as features).

[L19-26] **The Chapter 09 training loop, unsupervised.** The target is `X` itself, so no labels are involved. The tests check that a **1-D** nonlinear code reconstructs the curved data at least twice as well as **1-component** PCA. That's representation learning paying off on curved structure.

**Caveat.** A plain autoencoder's code space has no particular structure: nearby codes needn't decode to sensible data, and you can't *sample* from it. The VAE (Chapter 17) adds a prior and a probabilistic encoder to fix exactly that.
:::

:::tests{lang=python}
import torch

def _curved_data():
    g = torch.Generator().manual_seed(0)
    z = torch.rand(512, 2, generator=g) * 2 - 1
    A = torch.randn(2, 10, generator=g)
    return torch.tanh(z @ A * 1.5) + 0.02 * torch.randn(512, 10, generator=g)

def test_shapes():
    m = AutoEncoder(10, 3, 16)
    x = torch.randn(4, 10)
    assert m.encode(x).shape == (4, 3) and m(x).shape == (4, 10)

def test_beats_pca_with_the_same_code_size():
    torch.manual_seed(0)
    X = _curved_data()
    Xc = X - X.mean(0)
    _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
    pca_err = ((Xc - Xc @ Vt[:1].T @ Vt[:1]) ** 2).mean().item()
    ae_err = train_autoencoder(AutoEncoder(10, 1, 64), X)
    assert ae_err < 0.5 * pca_err
:::

:::hints
1. The encoder ends at the code size, and the decoder starts there.
2. The loss compares the model's output with its own input.
:::

:::interview
- When does a nonlinear autoencoder beat PCA, and when won't it? *(It wins on curved manifolds; on near-linear data it has no advantage.)*
- Why can't you sample new data from a plain autoencoder? What does a VAE add?
- Denoising, sparse and masked autoencoders: what does each change about the objective, and why do they learn better features?
:::
::::
