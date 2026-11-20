---
id: ch24-evaluation
summary: "A number without an error bar isn't a result. Put bootstrap confidence intervals on your metrics (paired, when the test set is shared), implement AUROC and calibration error from scratch, and see how fast \"significant\" findings appear when you test enough hypotheses."
udl: "—"
---

## Is the improvement real?

Research interviews probe whether you can tell a real effect from noise. Three habits cover most of it:

1. **Error bars on everything.** Report a confidence interval, not just a point estimate. Run several seeds, and resample the test set (the bootstrap).
2. **Pair your comparisons.** Two models evaluated on the *same* examples share most of their variance. Compare per-example differences, not two independent averages.
3. **Count your comparisons.** Try 20 things at $p < 0.05$ and you should *expect* one false positive.

Metrics also need care. Accuracy hides class imbalance, AUROC ranks without regard to calibration, and a model can be accurate yet badly overconfident (Exercise 24.2).

> UDL Chapter 8 covers measuring performance (train, validation and test splits; generalisation). The statistics here are self-contained.

::::exercise{id=bootstrap-ci type=implement difficulty=2 time=15 runtime=browser udl="—" tags="significance,metrics"}
# Bootstrap confidence intervals, paired

:::prereq
**The percentile bootstrap.** To get an interval for a statistic (here, a mean) of a sample $x_1..x_n$:

1. Resample $n$ points **with replacement**, $B$ times.
2. Compute the statistic on each resample.
3. Take the $\alpha/2$ and $1-\alpha/2$ quantiles of those $B$ values.

It needs no distributional assumptions, just the sample.

**Pairing.** If model A and model B are scored on the same test examples, bootstrap the per-example **differences** $d_i = a_i - b_i$. Example difficulty is shared, so it cancels, and the interval is much tighter than comparing two separately computed intervals.
:::

:::task
Implement `bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=0)`, returning `(mean, lo, hi)`. Draw all resample indices in one `rng.integers` call: no Python loop over resamples. Then implement `paired_diff_ci(a, b, **kw)` using it.
:::

:::starter{lang=python}
import numpy as np

def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=0):
    raise NotImplementedError

def paired_diff_ci(a, b, **kw):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    idx = {{ rng.integers(0, len(x), size=(n_boot, len(x))) | rng.integers(len(x), size=(n_boot, len(x))) | rng.integers(0, len(x), (n_boot, len(x))) ? `n_boot` resamples, each of `n` indices drawn **with replacement**: one `(B, n)` integer array. }}
    means = x[idx].mean(axis={{ 1 | -1 ? Each row is one resample, so average along the row. }})
    lo, hi = {{ np.quantile(means, [alpha / 2, 1 - alpha / 2]) | np.percentile(means, [50 * alpha, 100 - 50 * alpha]) | np.quantile(means, [alpha/2, 1 - alpha/2]) ? The percentile interval: the central $1-\alpha$ of the bootstrap distribution. }}
    return x.mean(), lo, hi

def paired_diff_ci(a, b, **kw):
    return bootstrap_ci({{ a - b | np.asarray(a) - np.asarray(b) ? Pair by example: bootstrap the per-example differences, so that shared difficulty cancels. }}, **kw)
:::

:::solution{lang=python}
import numpy as np

def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return x.mean(), lo, hi

def paired_diff_ci(a, b, **kw):
    return bootstrap_ci(a - b, **kw)
:::

:::explain
[L5-6] **All resamples at once.** Fancy-indexing with a `(B, n)` index array builds every resample in one go: no Python loop. That's Chapter 01's indexing, applied to statistics.

[L7-8] **Percentile interval.** It's simple, and good enough for means at moderate $n$. BCa (bias-corrected and accelerated) intervals fix skewed statistics. The tests check calibration directly: over many simulated datasets, about 95% of the intervals should contain the true mean.

[L10-11] **The paired comparison.** In the tests, two models differ by a small, consistent amount on examples whose difficulty varies a lot. The paired interval excludes 0, while comparing two independent intervals is far too wide to tell. That's the most common way real improvements get dismissed, or noise gets claimed.
:::

:::tests{lang=python}
import numpy as np

def test_coverage_is_about_95_percent():
    rng = np.random.default_rng(0)
    hits = 0
    for k in range(200):
        x = rng.normal(1.0, 2.0, size=60)
        _, lo, hi = bootstrap_ci(x, n_boot=1000, seed=k)
        hits += lo <= 1.0 <= hi
    assert 0.88 <= hits / 200 <= 0.99

def test_width_shrinks_like_root_n():
    rng = np.random.default_rng(1)
    w = lambda n: (lambda r: r[2] - r[1])(bootstrap_ci(rng.normal(size=n)))
    assert 0.35 < w(400) / w(100) < 0.65

def test_pairing_detects_a_small_consistent_gain():
    rng = np.random.default_rng(2)
    difficulty = rng.normal(0, 1, size=300)
    a = difficulty + 0.1 + 0.05 * rng.normal(size=300)
    b = difficulty + 0.05 * rng.normal(size=300)
    _, lo, hi = paired_diff_ci(a, b)
    assert lo > 0
    (_, alo, ahi), (_, blo, bhi) = bootstrap_ci(a), bootstrap_ci(b, seed=1)
    assert alo < bhi
:::

:::hints
1. One `(n_boot, n)` array of indices, then `x[idx]` gives every resample at once.
2. For pairs, what single vector should be resampled?
:::

:::interview
- How would you put an error bar on a test-set accuracy? On the gap between two models?
- Seeds versus bootstrap: what variance does each capture? *(Training randomness versus test-set sampling. Report both when you can.)*
- When does the percentile bootstrap fail? *(Tiny samples, extreme quantiles, and heavy tails.)*
:::
::::

::::exercise{id=auroc-ece type=implement difficulty=2 time=15 runtime=browser udl="—" tags="metrics,experimental-design"}
# AUROC and calibration error from scratch

:::prereq
**AUROC** is the probability that a randomly chosen positive scores higher than a randomly chosen negative, with ties counting ½ (the Mann–Whitney statistic). It depends only on the *ranking*, so it's insensitive to class balance and to any monotonic rescaling of the scores.

**Expected calibration error (ECE).** Bin the predictions by confidence into $M$ equal-width bins. In each bin compare the average confidence with the accuracy:
$$\mathrm{ECE} = \sum_{m} \frac{|B_m|}{n}\,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big| .$$
A model can have a high AUROC and a terrible ECE. Modern networks are typically **overconfident**.
:::

:::task
Implement `auroc(scores, labels)` (labels 0/1; compare every positive–negative pair by broadcasting, with ties counting ½) and `ece(conf, correct, n_bins=10)`, where `conf` holds confidences in $[0, 1]$ and `correct` is 0/1.
:::

:::starter{lang=python}
import numpy as np

def auroc(scores, labels):
    raise NotImplementedError

def ece(conf, correct, n_bins=10):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def auroc(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    wins = {{ (pos[:, None] > neg[None, :]).mean() | (pos[:, None] > neg).mean() | np.mean(pos[:, None] > neg[None, :]) ? Compare **every** positive with **every** negative by broadcasting a column against a row (Chapter 01), then take the fraction of pairs where the positive wins. }}
    ties = (pos[:, None] == neg[None, :]).mean()
    return wins + {{ 0.5 * ties | ties / 2 ? A tie is a coin flip, so it counts half. That's what makes constant scores give exactly 0.5. }}

def ece(conf, correct, n_bins=10):
    bins = np.minimum((conf * n_bins).astype(int), {{ n_bins - 1 ? A confidence of exactly 1.0 would land in bin `n_bins`, which doesn't exist. Clamp it into the last bin. }})
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += {{ m.mean() * abs(correct[m].mean() - conf[m].mean()) | m.mean() * np.abs(correct[m].mean() - conf[m].mean()) | abs(correct[m].mean() - conf[m].mean()) * m.mean() ? Weight each bin's |accuracy − confidence| gap by the fraction of predictions in that bin. }}
    return total
:::

:::solution{lang=python}
import numpy as np

def auroc(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    wins = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return wins + 0.5 * ties

def ece(conf, correct, n_bins=10):
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total
:::

:::explain
[L4-7] **AUROC as pairwise wins.** This is $O(n_+ n_-)$ memory, fine for an interview. At scale, use ranks: sort once, and AUROC $= \frac{R_+ - n_+(n_+ + 1)/2}{n_+ n_-}$, where $R_+$ is the sum of the positives' ranks. That's $O(n\log n)$.

[L10] **Bin by confidence.** Equal-width bins are the standard ECE. Equal-mass bins and the binning-free variants are less biased.

[L12-15] **Weighted gaps.** The tests check that a perfectly calibrated simulator (correct with probability equal to its confidence) gets an ECE near 0, and that an overconfident one (90% confident, 60% accurate) gets about 0.3.

**The fix for overconfidence** is usually **temperature scaling**: divide the logits by a single $T$ fitted on validation data. It changes calibration without changing any ranking or accuracy.
:::

:::tests{lang=python}
import numpy as np

def _brute_auc(s, y):
    pairs = [(p, n) for p in s[y == 1] for n in s[y == 0]]
    return sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in pairs) / len(pairs)

def test_auroc_extremes_and_brute_force():
    y = np.array([0, 0, 1, 1, 0, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7]), y) == 1.0
    assert auroc(-np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7]), y) == 0.0
    assert auroc(np.zeros(6), y) == 0.5
    rng = np.random.default_rng(0)
    s, yy = rng.integers(0, 5, size=40).astype(float), rng.integers(0, 2, size=40)
    assert np.isclose(auroc(s, yy), _brute_auc(s, yy))

def test_ece_calibrated_vs_overconfident():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.5, 1.0, size=20000)
    assert ece(conf, (rng.random(20000) < conf).astype(int)) < 0.02
    over = np.full(20000, 0.9)
    assert abs(ece(over, (rng.random(20000) < 0.6).astype(int)) - 0.3) < 0.02

def test_confidence_of_one_is_binned():
    assert np.isclose(ece(np.array([1.0, 1.0]), np.array([1, 1])), 0.0)
:::

:::hints
1. AUROC: a `(n_pos, n_neg)` boolean matrix of "positive beats negative".
2. ECE: assign each prediction a bin index first, then loop over the bins.
:::

:::interview
- AUROC versus AUPRC under heavy class imbalance: which would you report, and why? *(AUPRC: AUROC can look great while precision at useful thresholds is poor.)*
- What does temperature scaling fix, and what can't it change?
- Your classifier is 95% accurate on a dataset that is 95% one class. What do you say?
:::
::::

::::exercise{id=multiple-comparisons-predict type=predict difficulty=1 time=6 runtime=browser udl="—" tags="significance,experimental-design"}
# Twenty tweaks, one baseline

:::task
You try **20 variants** of a model, comparing each with its own fresh baseline runs. **None of them actually helps**, and each comparison is a fair, independent test at $p < 0.05$. What's the chance that **at least one** variant comes out "statistically significant"?
:::

:::choices
- [ ] 5%: that's what $p < 0.05$ means
- [ ] About 20%
- [x] About 64%
- [ ] 100%: with 20 tests you're guaranteed one
:::

:::experiment{lang=python}
import numpy as np
from math import erfc, sqrt

rng = np.random.default_rng(0)

def p_value(a, b):
    # two-sided z-test on the difference of means (known unit noise, 10 seeds each)
    z = (a.mean() - b.mean()) / sqrt(2 / len(a))
    return erfc(abs(z) / sqrt(2))

def any_significant(n_variants=20, threshold=0.05, shared_baseline=False):
    shared = rng.normal(size=10)
    return any(p_value(rng.normal(size=10), shared if shared_baseline else rng.normal(size=10)) < threshold
               for _ in range(n_variants))

trials = 2000
naive = np.mean([any_significant() for _ in range(trials)])
shared = np.mean([any_significant(shared_baseline=True) for _ in range(trials)])
bonf = np.mean([any_significant(threshold=0.05 / 20) for _ in range(trials)])
print(f"P(at least one 'significant' result, no real effects): {naive:.2f}   (1 - 0.95^20 = {1 - 0.95 ** 20:.2f})")
print(f"...if all 20 reuse ONE baseline run (correlated tests): {shared:.2f}")
print(f"with Bonferroni (p < 0.05/20):                         {bonf:.2f}")
:::

:::tests{lang=python}
def test_family_wise_error_is_large():
    import numpy as np
    rate = np.mean([any_significant() for _ in range(1500)])
    assert 0.5 < rate < 0.75

def test_shared_baseline_is_lower_but_still_far_above_five_percent():
    import numpy as np
    rate = np.mean([any_significant(shared_baseline=True) for _ in range(1500)])
    assert 0.25 < rate < 0.6

def test_bonferroni_controls_it():
    import numpy as np
    rate = np.mean([any_significant(threshold=0.05 / 20) for _ in range(1500)])
    assert rate < 0.1
:::

:::explain
Each null comparison has a 5% chance of a false positive. With 20 independent tests, the chance of *none* is $0.95^{20} \approx 0.36$, so the chance of at least one "win" is about **64%**, and the experiment agrees.

If every variant is compared with the **same** baseline run, the tests are correlated: an unluckily low baseline makes *all* the variants look good at once. The rate drops to about 40%, but that's still 8× the 5% you thought you were accepting.

**The fixes, roughly from strictest to most lenient:**

- **Bonferroni:** test each at $\alpha/m$. It controls the family-wise error rate, but is conservative.
- **Holm–Bonferroni:** a uniformly more powerful step-down version.
- **Benjamini–Hochberg:** controls the *false discovery rate* instead, which is better when you expect several real effects.
- **A held-out confirmation run:** pick the winner on one split or set of seeds, and confirm it on fresh ones. This is what good ML papers do implicitly.

**In ML practice** the "20 tests" are often hidden: hyperparameter sweeps, several datasets, several metrics, early-stopping checkpoints. Report what you tried.
:::

:::interview
- What's the multiple-comparisons problem? Give a fix and its trade-off.
- How many seeds would you run to claim a 0.5-point improvement? *(Estimate the seed standard deviation, then do a power calculation.)*
- What's p-hacking in ML terms? *(Choosing seeds, metrics or checkpoints after looking at the results.)*
:::
::::
