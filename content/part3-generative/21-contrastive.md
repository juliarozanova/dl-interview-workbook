---
id: ch21-contrastive
summary: "Contrastive learning turns representation learning into a classification problem: find your partner among the negatives. Implement InfoNCE stably, see what the temperature really controls, and write SimCLR's NT-Xent loss with its 2N-view bookkeeping."
udl: "—"
---

## Learning by telling things apart

There are no labels, so make some. Take two **views** of each example (two augmentations of an image, or an image and its caption). Train an encoder so that each view's embedding is closer to its partner's than to everyone else's in the batch. With a batch of $N$ pairs, each anchor faces an $N$-way classification problem: which of these is my positive?

$$\mathcal L_\text{InfoNCE} = -\frac1N\sum_i \log\frac{\exp(\mathrm{sim}(a_i, b_i)/\tau)}{\sum_j \exp(\mathrm{sim}(a_i, b_j)/\tau)},\qquad \mathrm{sim} = \text{cosine similarity}.$$

- **Theory.** $\log N - \mathcal L$ lower-bounds the mutual information between the views. That's why bigger batches (more negatives) help.
- **Practice.** SimCLR, MoCo and CLIP are all this loss, with different choices of views, negatives and encoders.

> UDL only touches on self-supervised learning in passing, so this chapter is self-contained.

::::exercise{id=info-nce type=implement difficulty=2 time=15 runtime=browser udl="—" tags="contrastive,numerical-stability,custom-losses"}
# InfoNCE

:::prereq
1. **L2-normalise** each embedding, so that dot products are cosine similarities in $[-1, 1]$.
2. **Logits** $= A B^\top/\tau$, shape `(N, N)`. Row $i$ scores anchor $i$ against every candidate. The positive is on the **diagonal**.
3. The loss is **cross-entropy with targets $0, 1, \dots, N-1$**: Chapter 04's stable log-softmax, picking the diagonal entries.
:::

:::task
Implement `l2_normalize(z)` and `info_nce(z1, z2, tau=0.1)`, where `z1[i]` and `z2[i]` are a positive pair. Both are `(N, D)`. Return the mean loss over anchors in `z1`. It must be stable for $\tau = 0.01$.
:::

:::starter{lang=python}
import numpy as np

def l2_normalize(z, eps=1e-8):
    raise NotImplementedError

def info_nce(z1, z2, tau=0.1):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def l2_normalize(z, eps=1e-8):
    return z / ({{ np.linalg.norm(z, axis=1, keepdims=True) + eps | np.linalg.norm(z, axis=-1, keepdims=True) + eps | np.sqrt((z ** 2).sum(axis=1, keepdims=True)) + eps ? Each row's own norm, kept as `(N, 1)` so it divides the row. ε guards against a zero vector. }})

def info_nce(z1, z2, tau=0.1):
    a, b = l2_normalize(z1), l2_normalize(z2)
    logits = {{ a @ b.T / tau | (a @ b.T) / tau | np.dot(a, b.T) / tau ? Every anchor against every candidate, as cosine similarity divided by the temperature: `(N, N)`. }}
    logits = logits - logits.max(axis=1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return {{ -np.mean(np.diag(log_probs)) | -np.diag(log_probs).mean() | -log_probs[np.arange(len(a)), np.arange(len(a))].mean() | -np.trace(log_probs) / len(a) ? Anchor $i$'s positive sits in column $i$: take the diagonal of the log-softmax, negate, and average. }}
:::

:::solution{lang=python}
import numpy as np

def l2_normalize(z, eps=1e-8):
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + eps)

def info_nce(z1, z2, tau=0.1):
    a, b = l2_normalize(z1), l2_normalize(z2)
    logits = a @ b.T / tau
    logits = logits - logits.max(axis=1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return -np.mean(np.diag(log_probs))
:::

:::explain
[L4] **Normalise rows.** Without it the model could cheat by inflating norms: dot products would grow without the directions getting any better. With it, the loss is invariant to rescaling the embeddings, and the tests check that.

[L8] **All pairs in one matmul.** The off-diagonal entries are the **in-batch negatives**. That's why contrastive methods love large batches, and why MoCo keeps a queue of extra negatives.

[L9-11] **Stable cross-entropy.** With $\tau = 0.01$, logits reach 100, and the max-subtraction from Chapter 04 is mandatory. Random embeddings give a loss of about $\log N$ (at $\tau = 1$), and well-aligned pairs drive it towards 0. The tests check both.
:::

:::tests{lang=python}
import numpy as np

def _naive(z1, z2, tau):
    a = z1 / np.linalg.norm(z1, axis=1, keepdims=True)
    b = z2 / np.linalg.norm(z2, axis=1, keepdims=True)
    s = np.exp(a @ b.T / tau)
    return -np.mean(np.log(np.diag(s) / s.sum(axis=1)))

def test_matches_naive():
    rng = np.random.default_rng(0)
    z1, z2 = rng.normal(size=(8, 16)), rng.normal(size=(8, 16))
    assert np.isclose(info_nce(z1, z2, 0.5), _naive(z1, z2, 0.5))

def test_random_is_about_log_n_and_aligned_is_low():
    rng = np.random.default_rng(1)
    z1 = rng.normal(size=(64, 256))
    assert abs(info_nce(z1, rng.normal(size=(64, 256)), 1.0) - np.log(64)) < 0.1
    assert info_nce(z1, z1 + 0.01 * rng.normal(size=z1.shape), 0.1) < 0.05

def test_scale_invariant_and_stable():
    rng = np.random.default_rng(2)
    z1, z2 = rng.normal(size=(16, 8)), rng.normal(size=(16, 8))
    assert np.isclose(info_nce(z1, z2), info_nce(10 * z1, 0.1 * z2))
    assert np.isfinite(info_nce(z1, z2, tau=0.01))
:::

:::hints
1. The positives are on the diagonal of the similarity matrix.
2. It's cross-entropy where the "class" of anchor $i$ is $i$.
:::

:::interview
- Why normalise embeddings in contrastive learning?
- Why do bigger batches help? What does MoCo's queue add? *(More negatives give a tighter mutual-information bound and a harder task.)*
- What is the "collapse" problem, and how do BYOL and SimSiam avoid it *without* negatives? *(A predictor head, stop-gradient, and an EMA target network.)*
:::
::::

::::exercise{id=temperature-predict type=predict difficulty=2 time=6 runtime=browser udl="—" tags="contrastive"}
# What does the temperature do?

:::task
An anchor has 255 negatives, with cosine similarities spread between $-0.3$ and $0.6$. In InfoNCE, the gradient pushes each negative away in proportion to its softmax weight. Lower the temperature from $\tau = 1$ to $\tau = 0.05$. What happens to how that weight is spread over the negatives?
:::

:::choices
- [ ] Nothing changes: $\tau$ only rescales the loss
- [ ] It spreads out more evenly over all negatives
- [x] It concentrates on the hardest (most similar) negatives
- [ ] It moves entirely onto the positive
:::

:::experiment{lang=python}
import numpy as np

rng = np.random.default_rng(0)
sims = np.sort(rng.uniform(-0.3, 0.6, size=255))[::-1]   # negatives, hardest first

def weights(tau):
    w = np.exp(sims / tau)
    return w / w.sum()

for tau in (1.0, 0.5, 0.1, 0.05):
    w = weights(tau)
    print(f"tau={tau:<5} top-10 negatives get {w[:10].sum():5.1%} of the push   effective # of negatives {1 / (w ** 2).sum():6.1f}")
:::

:::tests{lang=python}
def test_low_temperature_focuses_on_hard_negatives():
    w = weights(0.05)
    assert w[:10].sum() > 0.3 and 1 / (w ** 2).sum() < 50

def test_high_temperature_spreads_out():
    w = weights(1.0)
    assert 1 / (w ** 2).sum() > 200
:::

:::explain
The gradient of InfoNCE with respect to negative $j$'s similarity is proportional to its softmax weight, $p_j = e^{s_j/\tau}/\sum_k e^{s_k/\tau}$. Dividing by a small $\tau$ magnifies the differences between similarities before the exponential. The weight then piles onto the **hardest negatives**: the ones that look most like the positive. In the experiment, the effective number of negatives drops from about 240 to about 40.

**The trade-off** (Wang & Isola's alignment and uniformity analysis):

- **Low $\tau$:** strong hard-negative mining and well-spread embeddings. But it's sensitive to **false negatives** (another picture of the same class gets pushed away hard), and it can be unstable.
- **High $\tau$:** smooth and tolerant, but uninformative. Easy negatives dominate and the representations are less discriminative.

Typical values are 0.05–0.2 for SimCLR. CLIP *learns* $\tau$.
:::

:::interview
- What does the temperature control in contrastive losses?
- What are false negatives, and why do they hurt more at low temperature?
- Why might you learn the temperature (as CLIP does)?
:::
::::

::::exercise{id=nt-xent type=implement difficulty=3 time=20 runtime=local-torch udl="—" tags="contrastive,pytorch,masking"}
# SimCLR's NT-Xent loss

:::prereq
**SimCLR** puts **both** views into one pool: $z = [z_1; z_2]$, of shape $(2N, D)$. Every one of the $2N$ embeddings is an anchor:

- its positive is the other view of the same example, at index $i + N$ (or $i - N$);
- its negatives are all the other $2N - 2$ embeddings;
- **itself must be excluded**. Its self-similarity is always 1, and would make the task trivial.

So mask the diagonal of the $(2N, 2N)$ similarity matrix with $-\infty$, then apply cross-entropy with targets $[N, \dots, 2N-1, 0, \dots, N-1]$.
:::

:::task
Implement `nt_xent(z1, z2, tau=0.5)` in PyTorch. Use `F.normalize` and `F.cross_entropy`, and mask self-similarity out-of-place with `masked_fill`.
:::

:::starter{lang=python}
import torch
import torch.nn.functional as F

def nt_xent(z1, z2, tau=0.5):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import torch
import torch.nn.functional as F

def nt_xent(z1, z2, tau=0.5):
    N = z1.shape[0]
    z = F.normalize({{ torch.cat([z1, z2]) | torch.cat([z1, z2], dim=0) | torch.cat((z1, z2)) | torch.vstack([z1, z2]) ? Pool both views into one `(2N, D)` batch: every embedding is an anchor. }}, dim=1)
    sim = z @ z.T / tau
    sim = sim.masked_fill({{ torch.eye(2 * N, dtype=torch.bool) | torch.eye(2 * N).bool() | torch.eye(2*N, dtype=torch.bool) ? Exclude each anchor's similarity with **itself**: it's always 1, and would be an unbeatable "positive". }}, float("-inf"))
    targets = {{ torch.cat([torch.arange(N, 2 * N), torch.arange(N)]) | torch.cat([torch.arange(N) + N, torch.arange(N)]) | (torch.arange(2 * N) + N) % (2 * N) ? Row $i < N$ (a view-1 embedding) has its partner at $i + N$, and rows $\ge N$ point back to $i - N$. }}
    return F.cross_entropy(sim, targets)
:::

:::solution{lang=python}
import torch
import torch.nn.functional as F

def nt_xent(z1, z2, tau=0.5):
    N = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    sim = sim.masked_fill(torch.eye(2 * N, dtype=torch.bool), float("-inf"))
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(N)])
    return F.cross_entropy(sim, targets)
:::

:::explain
[L6] **One pool of $2N$.** Compared with Exercise 21.1, each anchor now has $2N-2$ negatives instead of $N-1$, drawn from both views. The loss is also symmetric: swapping `z1` and `z2` gives the same value, as the tests check.

[L7-8] **Mask the self-pairs out-of-place.** $-\infty$ becomes probability 0 after the softmax. `masked_fill` (not the in-place `masked_fill_`) keeps autograd happy. A bool mask from `torch.eye` is the cleanest way to build it.

[L9-10] **Targets point across the views.** The tests compare against a straightforward per-anchor loop that never builds a mask.
:::

:::tests{lang=python}
import torch

def _loop(z1, z2, tau):
    z = torch.nn.functional.normalize(torch.cat([z1, z2]), dim=1)
    n, total = len(z1), 0.0
    for i in range(2 * n):
        j = (i + n) % (2 * n)
        sims = torch.stack([z[i] @ z[k] / tau for k in range(2 * n) if k != i])
        pos = z[i] @ z[j] / tau
        total += -(pos - torch.logsumexp(sims, 0))
    return total / (2 * n)

def test_matches_loop_reference():
    torch.manual_seed(0)
    z1, z2 = torch.randn(6, 8), torch.randn(6, 8)
    assert torch.allclose(nt_xent(z1, z2, 0.5), _loop(z1, z2, 0.5), atol=1e-5)

def test_symmetric_and_differentiable():
    torch.manual_seed(1)
    z1, z2 = torch.randn(5, 4, requires_grad=True), torch.randn(5, 4)
    loss = nt_xent(z1, z2)
    assert torch.allclose(loss, nt_xent(z2, z1))
    loss.backward()
    assert z1.grad is not None and torch.isfinite(z1.grad).all()

def test_aligned_views_give_low_loss():
    torch.manual_seed(2)
    z1 = torch.randn(32, 64)
    assert nt_xent(z1, z1 + 0.01 * torch.randn(32, 64), tau=0.1) < 0.1
:::

:::hints
1. Build the `(2N, 2N)` similarity matrix first, then mask its diagonal.
2. Write out the targets for N = 2: `[2, 3, 0, 1]`.
:::

:::interview
- How does NT-Xent differ from one-directional InfoNCE? How many negatives does each anchor get?
- Why must self-similarity be masked?
- What's the projection head in SimCLR, and why is the representation taken *before* it?
:::
::::
