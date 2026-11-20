---
id: ch27-no-framework
summary: "Some coding rounds take PyTorch away. Three classics in plain NumPy, each with a time budget: k-means, beam search, and temperature, top-k and nucleus sampling. Turn on interview mode and time yourself."
udl: "—"
---

## Timed drills

These are the "implement it from scratch, no libraries" questions that come up in ML coding rounds. They're short to state and easy to get subtly wrong. For each one:

1. Turn on **interview mode** (top right). It hides the scaffold and solution and adds a clock to each exercise.
2. **Say your plan out loud** before typing: the data structures, the shapes, the edge cases.
3. Write it, run the tests, then compare with the reference.

Earlier chapters already drilled many from-scratch pieces: softmax and cross-entropy (04), backprop (05), Adam (07), conv2d (11), attention (13), LayerNorm (10). This chapter adds three more that appear often.

::::exercise{id=kmeans type=implement difficulty=2 time=15 runtime=browser udl="—" tags="tensor-shapes"}
# k-means (Lloyd's algorithm)

:::prereq
Repeat until the assignments stop changing (or for `n_iter` rounds):

1. **Assign.** Each point goes to its nearest centre. That's a `(N, K)` distance matrix (Chapter 01's pairwise trick), then `argmin`.
2. **Update.** Each centre moves to the mean of its points. A centre with **no** points keeps its old position (or is re-seeded), never `nan`.

The objective (inertia, the sum of squared distances to the assigned centres) never increases. Initialise from $K$ distinct data points. k-means++ is the better initialiser, and a good thing to mention.
:::

:::task
Implement `kmeans(X, k, n_iter=50, seed=0)`, returning `(centers (k, D), labels (N,), inertia_history list)`. Initialise with `k` distinct random rows of `X` (using `rng.choice(len(X), k, replace=False)`), and stop early once the labels stop changing.
:::

:::starter{lang=python}
import numpy as np

def kmeans(X, k, n_iter=50, seed=0):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def kmeans(X, k, n_iter=50, seed=0):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)].copy()
    labels, history = None, []
    for _ in range(n_iter):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new = {{ d.argmin(axis=1) | np.argmin(d, axis=1) | d.argmin(1) ? Each point's nearest centre: the minimum over the centre axis of the `(N, K)` distance matrix. }}
        history.append({{ d[np.arange(len(X)), new].sum() | d.min(axis=1).sum() | np.take_along_axis(d, new[:, None], 1).sum() ? Inertia: each point's squared distance to its assigned centre, summed. }})
        if labels is not None and {{ (new == labels).all() | np.array_equal(new, labels) ? Converged: no point changed cluster. }}:
            break
        labels = new
        for j in range(k):
            if (labels == j).any():
                centers[j] = {{ X[labels == j].mean(axis=0) | X[labels == j].mean(0) ? Move each centre to the mean of its members. An empty cluster keeps its old centre (the `if` above). }}
    return centers, labels, history
:::

:::solution{lang=python}
import numpy as np

def kmeans(X, k, n_iter=50, seed=0):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)].copy()
    labels, history = None, []
    for _ in range(n_iter):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new = d.argmin(axis=1)
        history.append(d[np.arange(len(X)), new].sum())
        if labels is not None and (new == labels).all():
            break
        labels = new
        for j in range(k):
            if (labels == j).any():
                centers[j] = X[labels == j].mean(axis=0)
    return centers, labels, history
:::

:::explain
[L5] **Initialise from the data.** Distinct rows, so no two centres start in the same place. `.copy()` matters: otherwise updating `centers` would write into `X` (Chapter 01's aliasing bug).

[L8-10] **Assignment step.** It's the `(N, K, D)` broadcast and a sum over features. For big $N$ and $D$, use the expansion trick from Exercise 1.2. The inertia is recorded at each assignment, and the tests check that it never increases.

[L11-13] **Convergence.** Stop when no label changes. Lloyd's algorithm always converges in finitely many steps, but to a *local* optimum, which is why people run several restarts and use k-means++ seeding.

[L14-16] **Update with an empty-cluster guard.** The mean of zero points is `nan`, which would then poison every later distance. Keeping the old centre is the simplest safe choice.
:::

:::tests{lang=python}
import numpy as np

def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    true = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    return np.concatenate([c + 0.5 * rng.normal(size=(100, 2)) for c in true]), true

def test_recovers_well_separated_clusters():
    X, true = _blobs()
    best = min((kmeans(X, 3, seed=s) for s in range(5)), key=lambda r: r[2][-1])
    centers = best[0]
    assert all(np.min(np.linalg.norm(centers - t, axis=1)) < 0.3 for t in true)

def test_inertia_never_increases():
    X, _ = _blobs(1)
    _, labels, hist = kmeans(X, 4, seed=2)
    assert labels.shape == (300,) and all(b <= a + 1e-9 for a, b in zip(hist, hist[1:]))

def test_does_not_modify_input_and_survives_empty_clusters():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]])
    before = X.copy()
    centers, _, _ = kmeans(X, 3, seed=0)
    assert np.array_equal(X, before) and np.isfinite(centers).all()
:::

:::hints
1. Two alternating steps: an `argmin` over a distance matrix, then per-cluster means.
2. Guard against empty clusters, and remember to `.copy()` the initial centres.
:::

:::interview
- What does k-means optimise? Is it guaranteed to find the global optimum? *(Inertia. No: it's NP-hard in general, and Lloyd's only finds local optima.)*
- How does k-means++ initialise, and why does it help?
- When is k-means the wrong tool? *(Non-spherical clusters or very different cluster sizes. Use a GMM, DBSCAN or spectral clustering.)*
:::
::::

::::exercise{id=beam-search type=implement difficulty=3 time=20 runtime=browser udl="—" tags="autoregressive"}
# Beam search

:::prereq
**Decoding** picks a high-probability sequence from an autoregressive model, $\log p(y) = \sum_t \log p(y_t\mid y_{<t})$.

- **Greedy:** take the argmax at every step. That's myopic.
- **Beam search:** keep the `B` best **partial** sequences by total log-probability. At each step, expand each with every token, then keep the best `B` of the `B × V` candidates.
- **Finishing:** a sequence that emits `eos` is complete. Set it aside, and keep expanding the others. Stop at `max_len`, then return the best finished sequence (or the best unfinished one, if none finished).

Real systems add **length normalisation**, because summed log-probabilities favour short outputs. Leave it out here.
:::

:::task
Implement `beam_search(next_logprobs, beam=3, max_len=6, eos=0)`. `next_logprobs(prefix)` takes a tuple of token ids and returns a length-`V` array of log-probabilities for the next token. Start from the empty prefix `()`. Return `(best_sequence_tuple, its_total_logprob)`, where finished sequences include their `eos`.
:::

:::starter{lang=python}
import numpy as np

def beam_search(next_logprobs, beam=3, max_len=6, eos=0):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def beam_search(next_logprobs, beam=3, max_len=6, eos=0):
    beams, finished = [((), 0.0)], []
    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            lp = next_logprobs(seq)
            for tok in range(len(lp)):
                candidates.append(({{ seq + (tok,) | (*seq, tok) ? Extend the prefix by one token. Tuples are immutable, so this builds a new one. }}, {{ score + lp[tok] | lp[tok] + score ? Log-probabilities of a sequence **add** across steps. }}))
        candidates.sort(key=lambda c: {{ -c[1] ? Best first: sort by total log-probability, descending. }})
        beams = []
        for seq, score in candidates:
            if seq[-1] == eos:
                finished.append((seq, score))
            else:
                beams.append((seq, score))
            if len(beams) == beam:
                break
        if not beams:
            break
    pool = finished if finished else beams
    return {{ max(pool, key=lambda c: c[1]) | sorted(pool, key=lambda c: -c[1])[0] ? Return the single best complete sequence, or the best partial one if nothing emitted `eos` in time. }}
:::

:::solution{lang=python}
import numpy as np

def beam_search(next_logprobs, beam=3, max_len=6, eos=0):
    beams, finished = [((), 0.0)], []
    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            lp = next_logprobs(seq)
            for tok in range(len(lp)):
                candidates.append((seq + (tok,), score + lp[tok]))
        candidates.sort(key=lambda c: -c[1])
        beams = []
        for seq, score in candidates:
            if seq[-1] == eos:
                finished.append((seq, score))
            else:
                beams.append((seq, score))
            if len(beams) == beam:
                break
        if not beams:
            break
    pool = finished if finished else beams
    return max(pool, key=lambda c: c[1])
:::

:::explain
[L4-9] **Expand every beam with every token.** There are `B × V` candidates, each scored by its summed log-probability. Real implementations vectorise this with a `(B, V)` score matrix and `topk`.

[L10-18] **Keep the best `B` unfinished.** Walking the sorted candidates, sequences ending in `eos` go to `finished`, and the rest fill the beam up to `B`. Note that finished candidates seen *before* the beam fills are also kept.

[L19-21] **Pick the winner.** In the tests, a model built so that greedy decoding takes a tempting first token is beaten by beam search, which matches the brute-force optimum over every sequence. With `beam=1`, the result must equal greedy decoding.

**Why not always beam?** Summed log-probability favours short, generic outputs, and wide beams can make text *worse* ("the beam search curse"). Open-ended generation usually *samples* instead (next exercise).
:::

:::tests{lang=python}
import itertools
import numpy as np

V = 4

def _model(seed=0):
    rng = np.random.default_rng(seed)
    table = {}
    def next_logprobs(prefix):
        if prefix not in table:
            z = rng.normal(size=V) * 2
            table[prefix] = z - np.log(np.exp(z).sum())
        return table[prefix]
    return next_logprobs

def _brute(next_logprobs, max_len, eos=0):
    best = (None, -np.inf)
    for L in range(1, max_len + 1):
        for seq in itertools.product(range(V), repeat=L):
            if eos in seq[:-1] or (L < max_len and seq[-1] != eos):
                continue
            s = sum(next_logprobs(seq[:i])[seq[i]] for i in range(L))
            best = max(best, (seq, s), key=lambda c: c[1])
    return best

def test_wide_beam_finds_the_brute_force_optimum():
    f = _model(3)
    seq, score = beam_search(f, beam=64, max_len=3)
    bseq, bscore = _brute(f, 3)
    assert np.isclose(score, bscore) and seq == bseq

def test_beam_one_is_greedy():
    f, seq = _model(1), ()
    for _ in range(4):
        tok = int(np.argmax(f(seq)))
        seq += (tok,)
        if tok == 0:
            break
    assert beam_search(f, beam=1, max_len=4)[0] == seq

def test_score_is_the_sum_of_logprobs():
    f = _model(2)
    seq, score = beam_search(f, beam=3, max_len=5)
    assert np.isclose(score, sum(f(seq[:i])[seq[i]] for i in range(len(seq))))
:::

:::hints
1. A beam is a list of `(sequence, score)` pairs. Expand them all, sort, and keep the best.
2. Finished sequences (ending in `eos`) must leave the beam, but stay in the running.
:::

:::interview
- Greedy versus beam versus sampling: when would you use each?
- Why do summed log-probabilities favour short outputs? How does length normalisation fix it?
- What's the time complexity of beam search? *($O(T\cdot B\cdot V)$ scoring, plus the cost of selecting the top $B$.)*
:::
::::

::::exercise{id=sampling-filters type=implement difficulty=2 time=15 runtime=browser udl="—" tags="autoregressive,numerical-stability"}
# Temperature, top-k and nucleus sampling

:::prereq
To turn logits into a sampling distribution:

1. **Temperature** $T$: use $\mathrm{softmax}(z/T)$. $T < 1$ sharpens it (towards greedy), and $T > 1$ flattens it.
2. **Top-k:** keep only the $k$ largest logits; set the rest to $-\infty$.
3. **Top-p (nucleus):** sort the probabilities in descending order and keep the **smallest** prefix whose cumulative mass reaches $p$. Always keep at least the top token.
4. **Renormalise** what's left.

Apply them in that order: temperature first, so the nucleus is computed on the tempered distribution.
:::

:::task
Implement `filtered_probs(logits, temperature=1.0, top_k=None, top_p=None)` for a 1-D array of logits, returning a probability vector of the same length.
:::

:::starter{lang=python}
import numpy as np

def filtered_probs(logits, temperature=1.0, top_k=None, top_p=None):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def filtered_probs(logits, temperature=1.0, top_k=None, top_p=None):
    z = logits / temperature
    if top_k is not None:
        kth = {{ np.sort(z)[-top_k] | np.sort(z)[::-1][top_k - 1] | np.partition(z, -top_k)[-top_k] ? The k-th largest logit is the cut-off: anything strictly below it is removed. }}
        z = np.where(z < kth, -np.inf, z)
    p = np.exp(z - z.max())
    p /= p.sum()
    if top_p is not None:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = {{ cum - p[order] < top_p | (cum - p[order]) < top_p | np.concatenate([[True], cum[:-1] < top_p]) ? Keep a token if the mass **before** it is still below $p$. That's the smallest prefix reaching $p$, and it always keeps the top token. }}
        mask = np.zeros_like(p, dtype=bool)
        mask[order[keep]] = True
        p = np.where(mask, p, 0.0)
        p /= {{ p.sum() ? Renormalise over the survivors so it's a distribution again. }}
    return p
:::

:::solution{lang=python}
import numpy as np

def filtered_probs(logits, temperature=1.0, top_k=None, top_p=None):
    z = logits / temperature
    if top_k is not None:
        kth = np.sort(z)[-top_k]
        z = np.where(z < kth, -np.inf, z)
    p = np.exp(z - z.max())
    p /= p.sum()
    if top_p is not None:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = cum - p[order] < top_p
        mask = np.zeros_like(p, dtype=bool)
        mask[order[keep]] = True
        p = np.where(mask, p, 0.0)
        p /= p.sum()
    return p
:::

:::explain
[L4] **Temperature first.** The tests check its limits: a very small $T$ approaches one-hot on the argmax, and a very large $T$ approaches uniform.

[L5-7] **Top-k via a threshold.** Setting the dropped logits to $-\infty$ makes their probability exactly 0 after the stable softmax (Chapter 04). With ties at the cut-off, all tied tokens survive, which is a common and acceptable convention.

[L10-17] **The nucleus.** "The mass *before* this token is below $p$" includes exactly the tokens needed to reach $p$, and always the top one, even if it alone exceeds $p$. Using `cum <= top_p` instead is the classic off-by-one: it can drop the token that crosses the threshold, and with a dominant top token it keeps nothing at all.

**When to use which.** Top-p adapts to the model's confidence: few candidates when it's sure, many when it isn't. Top-k doesn't adapt. Open-ended chat models typically use temperature around 0.7 with top-p around 0.9.
:::

:::tests{lang=python}
import numpy as np

def test_temperature_limits():
    z = np.array([1.0, 2.0, 3.0])
    assert np.allclose(filtered_probs(z, temperature=1e-3), [0, 0, 1], atol=1e-6)
    assert np.allclose(filtered_probs(z, temperature=1e4), 1 / 3, atol=1e-3)

def test_top_k_keeps_exactly_k():
    p = filtered_probs(np.array([0.1, 3.0, 2.0, -1.0, 2.5]), top_k=2)
    assert np.isclose(p.sum(), 1) and (p > 0).sum() == 2 and p[1] > 0 and p[4] > 0

def test_nucleus_is_the_smallest_prefix_reaching_p():
    z = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
    p = filtered_probs(z, top_p=0.75)
    assert np.allclose(p, [0.5 / 0.8, 0.3 / 0.8, 0, 0])
    assert np.allclose(filtered_probs(z, top_p=0.3), [1, 0, 0, 0])

def test_combined_filters_are_a_distribution():
    z = np.random.default_rng(0).normal(size=50)
    p = filtered_probs(z, temperature=0.7, top_k=20, top_p=0.9)
    assert np.isclose(p.sum(), 1) and (p >= 0).all() and (p > 0).sum() <= 20
:::

:::hints
1. Top-k: find the k-th largest value and mask everything below it.
2. Top-p: sort the probabilities descending, and use the cumulative mass **excluding** the current token.
:::

:::interview
- Top-k versus top-p: which adapts to the model's confidence, and why does that matter?
- What does temperature do to the entropy of the distribution?
- Why might greedy decoding repeat itself, and how does sampling help? *(Repetition loops. Sampling, plus repetition penalties, breaks them.)*
:::
::::
