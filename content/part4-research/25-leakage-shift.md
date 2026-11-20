---
id: ch25-leakage-shift
summary: "The two classic ways a model looks better than it is: information leaking from the test set, and shortcuts that stop working when the distribution shifts. Fix a preprocessing leak, predict what a colour shortcut does at test time (Colored MNIST in miniature), then remove the shortcut by reweighting."
udl: "—"
---

## When validation lies

**Leakage** means information about the test data reached training, so the test score is optimistic. Typical routes:

- **Preprocessing fitted on all the data**: normalisation, vocabularies, feature selection, PCA.
- **Duplicates or near-duplicates** across splits: the same patient, user or document, or augmented copies.
- **Temporal leaks**: training on the future to predict the past.
- **Target leaks**: a feature that is only known *because of* the label.

**Distribution shift** means the test distribution differs from training. A model that learned a **spurious shortcut** (a feature that's predictive in training but not causal) collapses when the correlation changes. **Colored MNIST** (Arjovsky et al., 2019) is the canonical demonstration:

- digit shape predicts the label 75% of the time;
- colour predicts it 90% of the time in training;
- in testing, the colour correlation is *reversed*.

Exercises 25.2–25.3 reproduce it in miniature. The full version is in `companion/workbook/data.py` (see "Going further" at the end).

::::exercise{id=preprocessing-leak type=debug difficulty=1 time=8 runtime=browser udl="—" tags="leakage,debugging"}
# The scaler that saw the test set

:::prereq
**Rule:** every statistic used for preprocessing (means, standard deviations, vocabularies, PCA components, imputation values) must be **fitted on the training split only**, then *applied* to validation and test. Otherwise the test rows influence how training data is transformed, and the evaluation is no longer of unseen data.

With a lot of data and plain standardisation the effect is often small. With small data, target encoding, feature selection or oversampling, it can be enormous. The habit matters more than the size.
:::

:::task
`split_and_standardise` shuffles, splits off a test set, and standardises the features. A reviewer says it leaks test information into training. Find the leak and fix it, keeping the function's interface. The training features must still come out exactly standardised.
:::

:::starter{lang=python}
import numpy as np

def split_and_standardise(X, y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test, train = idx[:n_test], idx[n_test:]
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    Z = (X - mu) / sd
    return Z[train], y[train], Z[test], y[test]
:::

:::scaffold{lang=python}
import numpy as np

def split_and_standardise(X, y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test, train = idx[:n_test], idx[n_test:]
    mu, sd = {{ X[train].mean(axis=0), X[train].std(axis=0) + 1e-8 | X[train].mean(0), X[train].std(0) + 1e-8 ? Fit the statistics on the **training rows only**, then apply them to every row. The test rows must not influence how the training data is transformed. }}
    Z = (X - mu) / sd
    return Z[train], y[train], Z[test], y[test]
:::

:::solution{lang=python}
import numpy as np

def split_and_standardise(X, y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test, train = idx[:n_test], idx[n_test:]
    mu, sd = X[train].mean(axis=0), X[train].std(axis=0) + 1e-8
    Z = (X - mu) / sd
    return Z[train], y[train], Z[test], y[test]
:::

:::explain
[L8] **Fit on train, apply to all.** The test features are then transformed with statistics they had no say in, exactly as they would be at deployment time, when tomorrow's data arrives.

[L9] **Apply once, to everything.** The test-split features won't be exactly mean-0 and std-1, and that's correct: it reflects the real shift between the two samples.

**How to make it impossible to get wrong.** Use pipeline objects (scikit-learn's `Pipeline`, or a `fit`/`transform` interface) and fit them inside each cross-validation fold. The tests check the property directly: wildly changing the *test* rows must leave the transformed *training* rows untouched.
:::

:::tests{lang=python}
import numpy as np

def _data(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(5, 3, size=(200, 4)), rng.integers(0, 2, size=200)

def test_train_split_is_exactly_standardised():
    X, y = _data()
    Ztr, _, _, _ = split_and_standardise(X, y)
    assert np.allclose(Ztr.mean(0), 0, atol=1e-10) and np.allclose(Ztr.std(0), 1, atol=1e-6)

def test_test_rows_do_not_influence_training_features():
    X, y = _data(1)
    Ztr, _, _, _ = split_and_standardise(X, y)
    test_idx = np.random.default_rng(0).permutation(len(X))[:40]
    X2 = X.copy()
    X2[test_idx] += 1000.0
    Ztr2, _, _, _ = split_and_standardise(X2, y)
    assert np.allclose(Ztr, Ztr2)
:::

:::hints
1. Which rows does `X.mean(axis=0)` look at?
2. The split indices exist *before* the statistics are computed. Use them.
:::

:::interview
- List four ways test information can leak into training.
- How do you split data that contains several samples per patient or user? *(Group-aware splits, e.g. `GroupKFold`.)*
- How would you detect near-duplicates between train and test? *(Hashing, embedding nearest-neighbour search, n-gram overlap.)*
:::
::::

::::exercise{id=shortcut-shift-predict type=predict difficulty=2 time=8 runtime=browser udl="—" tags="distribution-shift"}
# Colored MNIST in miniature

:::task
Each example has two noisy features. **Shape** agrees with the label 75% of the time. **Colour** agrees with the label 90% of the time in **training**, but only 10% of the time in **testing** (the correlation is reversed). A logistic regression is trained normally (empirical risk minimisation) and reaches about 90% training accuracy. What's its **test** accuracy?
:::

:::choices
- [ ] About 75%: it learns to rely on shape, which is stable
- [ ] About 90%: nothing about the shape-label relationship changed
- [ ] About 50%: the two features cancel out
- [x] Far below 50%: it mostly follows colour, which is now anti-correlated
:::

:::experiment{lang=python}
import numpy as np

def make_env(n, p_color_flip, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    shape = y ^ (rng.random(n) < 0.25)            # agrees with y 75% of the time, everywhere
    color = y ^ (rng.random(n) < p_color_flip)    # agrees 90% in train, 10% in test
    X = np.c_[2 * shape - 1, 2 * color - 1].astype(float) + 0.3 * rng.normal(size=(n, 2))
    return X, y, color

def train_logreg(X, y, w=None, steps=2000, lr=0.5):
    w = np.ones(len(y)) if w is None else w
    w = w / w.sum()
    Xb, theta = np.c_[X, np.ones(len(X))], np.zeros(X.shape[1] + 1)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xb @ theta))
        theta -= lr * Xb.T @ (w * (p - y))
    return theta

def accuracy(theta, X, y):
    return ((np.c_[X, np.ones(len(X))] @ theta > 0) == y).mean()

Xtr, ytr, ctr = make_env(4000, 0.1, seed=0)
Xte, yte, cte = make_env(4000, 0.9, seed=1)
theta = train_logreg(Xtr, ytr)
print(f"weights  shape {theta[0]:+.2f}   colour {theta[1]:+.2f}")
print(f"train accuracy {accuracy(theta, Xtr, ytr):.1%}    test accuracy {accuracy(theta, Xte, yte):.1%}")
:::

:::tests{lang=python}
def test_erm_learns_the_shortcut_and_fails_under_shift():
    theta = train_logreg(Xtr, ytr)
    assert theta[1] > theta[0]
    assert accuracy(theta, Xtr, ytr) > 0.85 and accuracy(theta, Xte, yte) < 0.3
:::

:::explain
In training, colour is the **better** predictor (90% versus 75%), so minimising training loss leans on it. The fitted colour weight is about twice the shape weight. At test time the colour–label relationship flips, the model follows colour, and its accuracy drops to about 13%: **worse than chance**. Training accuracy and an i.i.d. validation split would both have looked excellent.

**The general lesson.** ERM learns *whatever is predictive in the training distribution*, causal or not. The defences are:

- **Know your shifts.** Evaluate on held-out *environments* or time periods, not just random splits.
- **Break the shortcut in the data.** Rebalance or augment so that it's no longer predictive (next exercise).
- **Use invariance-seeking objectives.** IRM, group DRO and others look for features whose relationship to the label holds across environments. They are often hard to make beat well-tuned ERM in practice. Mention that nuance.
:::

:::interview
- What's a spurious correlation? Give a real example. *(Hospital tokens in X-rays, backgrounds in animal photos, length in NLI.)*
- Covariate shift versus label shift versus concept shift: define each.
- How would you build an evaluation that catches shortcut learning?
:::
::::

::::exercise{id=group-reweighting type=implement difficulty=2 time=12 runtime=browser udl="—" tags="distribution-shift,experimental-design"}
# Remove the shortcut by reweighting

:::prereq
If you *know* the spurious attribute (here, colour), you can stop it being predictive in the training objective. Weight each example so that every **(label, colour) group** carries **equal total weight**:
$$w_i = \frac{n}{G\cdot n_{g(i)}},$$
where $G = 4$ groups and $n_g$ is the group's size. Under these weights, colour and label are independent, so colour carries no information about the label, and the model has to use shape.

That's the idea behind group-balanced sampling, and the starting point for group DRO, which upweights the *worst* group adaptively.
:::

:::task
Implement `group_weights(y, color)`, returning per-example weights such that each of the four `(y, color)` groups has the same total weight and the weights average 1. `make_env`, `train_logreg` and `accuracy` from Exercise 25.2 are provided in the tests.
:::

:::starter{lang=python}
import numpy as np

def group_weights(y, color):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def group_weights(y, color):
    g = {{ 2 * y + color | y * 2 + color | 2 * color + y ? Encode the (label, colour) pair as a single group id in {0, 1, 2, 3}. }}
    counts = {{ np.bincount(g, minlength=4) | np.bincount(g, minlength=4).astype(float) ? The size of each group. `minlength` keeps an empty group from shifting the indices. }}
    return {{ len(g) / (4 * counts[g]) | len(y) / (4 * counts[g]) | len(g) / (4.0 * counts[g]) ? Weight inversely to the group size, scaled so that each group's total is $n/4$ and the average weight is 1. }}
:::

:::solution{lang=python}
import numpy as np

def group_weights(y, color):
    g = 2 * y + color
    counts = np.bincount(g, minlength=4)
    return len(g) / (4 * counts[g])
:::

:::explain
[L4-6] **Inverse group frequency.** The rare groups (where colour disagrees with the label, 10% of the data) get about 9× the weight of the common ones. Under the weights, $P(\text{colour}\mid y) = \tfrac12$, so the colour feature is useless to the loss.

**The result** (checked in the tests): the colour weight collapses to about 0, and the model reaches about 75% on **both** environments. That's the accuracy shape alone can give. Test accuracy goes from about 13% to about 75%, at the cost of some training accuracy, which was illusory anyway.

**Going further, with real images.** `companion/workbook/data.py` builds actual Colored MNIST (it downloads MNIST once):

```python
from workbook.data import colored_mnist_envs
envs = colored_mnist_envs()               # train_a (10% flips), train_b (20%), test (90%)
x, y, color = envs["train_a"]             # x: (20000, 2, 28, 28)
```

Train a small CNN with and without these weights. You should see the same story, with a CNN in place of a 2-feature model.
:::

:::tests{lang=python}
import numpy as np

def make_env(n, p_color_flip, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    shape = y ^ (rng.random(n) < 0.25)
    color = y ^ (rng.random(n) < p_color_flip)
    X = np.c_[2 * shape - 1, 2 * color - 1].astype(float) + 0.3 * rng.normal(size=(n, 2))
    return X, y, color

def train_logreg(X, y, w=None, steps=2000, lr=0.5):
    w = np.ones(len(y)) if w is None else w
    w = w / w.sum()
    Xb, theta = np.c_[X, np.ones(len(X))], np.zeros(X.shape[1] + 1)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xb @ theta))
        theta -= lr * Xb.T @ (w * (p - y))
    return theta

def accuracy(theta, X, y):
    return ((np.c_[X, np.ones(len(X))] @ theta > 0) == y).mean()

def test_groups_have_equal_total_weight():
    rng = np.random.default_rng(0)
    y, c = rng.integers(0, 2, 1000), rng.integers(0, 2, 1000)
    w = group_weights(y, c)
    totals = [w[(y == a) & (c == b)].sum() for a in (0, 1) for b in (0, 1)]
    assert np.allclose(totals, totals[0]) and np.isclose(w.mean(), 1.0)

def test_reweighting_removes_the_shortcut():
    Xtr, ytr, ctr = make_env(4000, 0.1, 0)
    Xte, yte, _ = make_env(4000, 0.9, 1)
    theta = train_logreg(Xtr, ytr, group_weights(ytr, ctr))
    assert abs(theta[1]) < 0.2 and accuracy(theta, Xte, yte) > 0.7
:::

:::hints
1. Turn each `(y, color)` pair into one integer group id.
2. A group with $n_g$ members gets weight $\propto 1/n_g$ each.
:::

:::interview
- When can you reweight or resample to remove a shortcut, and when can't you? *(You need to know the attribute, or at least have environment labels.)*
- What does group DRO optimise? *(The worst group's loss.)*
- Why do invariance methods such as IRM often fail to beat tuned ERM on real benchmarks?
:::
::::
