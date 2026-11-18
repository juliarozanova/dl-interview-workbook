---
id: ch09-training-loop
summary: Write the training loop you'll be asked to write in every coding round, with train/eval modes, `zero_grad` and `no_grad` all in the right places. Then fix two bugs that don't crash and only make your model quietly worse.
udl: "Ch. 6–9"
---

## The loop, and how to debug it

Every PyTorch training loop is the same five lines, wrapped in bookkeeping:

```python
model.train()                    # dropout on, batch-norm uses batch statistics
for x, y in loader:
    opt.zero_grad()              # gradients ACCUMULATE by default: clear them
    loss = loss_fn(model(x), y)  # forward
    loss.backward()              # backward: fills p.grad for every parameter
    opt.step()                   # update
```

Evaluation flips two switches: `model.eval()` changes layer *behaviour*, and `torch.no_grad()` stops autograd *recording*. They're independent, and you need both.

**A debugging checklist** (worth reciting in interviews):

1. **Overfit one batch.** A model that can't drive the loss on 8 examples to about 0 has a bug, not a capacity problem.
2. **Check the initial loss.** For $C$ balanced classes it should be about $\log C$ (Chapter 04).
3. **Check gradient norms per layer.** Zero means a detached graph or dead units. Huge means you need a lower learning rate or clipping.
4. **Check the data.** Visualise a batch *after* preprocessing, and check the labels line up.
5. **Switch modes correctly.** Forgetting `eval()` gives noisy, slightly wrong metrics. Forgetting `train()` after evaluating silently disables dropout.

::::exercise{id=train-eval-loop type=implement difficulty=2 time=20 runtime=local-torch udl="§6.2, §9.3" tags="training-loop,pytorch"}
# `train_epoch` and `evaluate`

:::prereq
- **`model.train()` and `model.eval()`** set the `training` flag on every sub-module. Dropout and batch norm read it.
- **`torch.no_grad()`** (as a context manager or decorator) tells autograd not to build a graph. That saves memory and time, and makes outputs have `requires_grad=False`.
- **Averaging correctly.** If the last batch is smaller, a mean of per-batch means is biased. Weight each batch loss by its batch size.
- **`loss.item()`** converts a 0-dim tensor to a Python float. Accumulating the *tensor* instead keeps every batch's graph alive, which is a memory leak.
:::

:::task
Implement:

- `train_epoch(model, loader, opt, loss_fn)`: one pass over `loader` in training mode. Return the mean loss per **example**.
- `evaluate(model, loader, loss_fn)`: evaluation mode, with no autograd. Return `(mean_loss, accuracy)`, where the logits' `argmax` over dim 1 is the prediction.
:::

:::starter{lang=python}
import torch

def train_epoch(model, loader, opt, loss_fn):
    raise NotImplementedError

def evaluate(model, loader, loss_fn):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import torch

def train_epoch(model, loader, opt, loss_fn):
    {{ model.train() | model.train(True) | model.train(mode=True) ? Switch dropout and batch norm into training behaviour. You may have just called `evaluate`, which left the model in eval mode. }}
    total, n = 0.0, 0
    for x, y in loader:
        {{ opt.zero_grad() | opt.zero_grad(set_to_none=True) | model.zero_grad() ? `.backward()` **adds** into `p.grad`. Without clearing it, every step uses the sum of all previous gradients. }}
        loss = loss_fn(model(x), y)
        {{ loss.backward() ? Fill `p.grad` for every parameter by backpropagating from this scalar loss. }}
        opt.step()
        total += loss.item() * len(x)
        n += len(x)
    return total / n

{{ @torch.no_grad() | @torch.inference_mode() ? Evaluation doesn't need gradients. Turning autograd off saves the memory of storing activations, and returns tensors with `requires_grad=False`. }}
def evaluate(model, loader, loss_fn):
    {{ model.eval() | model.train(False) | model.train(mode=False) ? Dropout off, and batch norm uses its running statistics. Without this, the metrics are noisy and biased. }}
    total, correct, n = 0.0, 0, 0
    for x, y in loader:
        logits = model(x)
        total += loss_fn(logits, y).item() * len(x)
        correct += (logits.{{ argmax(dim=1) | argmax(1) | argmax(dim=-1) | argmax(-1) ? The predicted class is the index of the largest logit. You don't need a softmax, because it's monotonic. }} == y).sum().item()
        n += len(x)
    return total / n, correct / n
:::

:::solution{lang=python}
import torch

def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(x)
        n += len(x)
    return total / n

@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total, correct, n = 0.0, 0, 0
    for x, y in loader:
        logits = model(x)
        total += loss_fn(logits, y).item() * len(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += len(x)
    return total / n, correct / n
:::

:::explain
[L4] **Set the mode at the start of every epoch.** The loop can't assume the state the model was left in.

[L7-10] **The core four lines.** Clear, forward, backward, step. `zero_grad(set_to_none=True)`, the default in recent PyTorch, frees the gradient memory rather than filling it with zeros.

[L11-12] **Per-example average.** `loss.item() * len(x)` undoes the batch mean, so a smaller final batch is weighted correctly. `.item()` also detaches, so no graph is kept alive.

[L15-17] **Both switches for evaluation.** `no_grad` controls autograd and `eval()` controls layer behaviour. The tests check both: `evaluate` must be deterministic despite the dropout layer, and must not build a graph.

[L21-22] **Accuracy from logits.** `argmax` over the class dimension. Accumulate counts as Python ints, and divide once at the end.
:::

:::tests{lang=python}
import torch
from torch.utils.data import DataLoader, TensorDataset

def _setup(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.cat([torch.randn(256, 2, generator=g) + 2, torch.randn(256, 2, generator=g) - 2])
    y = torch.cat([torch.zeros(256, dtype=torch.long), torch.ones(256, dtype=torch.long)])
    loader = DataLoader(TensorDataset(x, y), batch_size=60, shuffle=True, generator=g)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.ReLU(), torch.nn.Dropout(0.5), torch.nn.Linear(32, 2))
    return model, loader

def test_learns():
    model, loader = _setup()
    opt, loss_fn = torch.optim.SGD(model.parameters(), lr=0.1), torch.nn.CrossEntropyLoss()
    for _ in range(15):
        train_epoch(model, loader, opt, loss_fn)
    loss, acc = evaluate(model, loader, loss_fn)
    assert acc > 0.95 and loss < 0.2

def test_evaluate_is_deterministic_and_graph_free():
    model, loader = _setup(1)
    seen = []
    def spy(logits, y):
        seen.append(logits.requires_grad)
        return torch.nn.functional.cross_entropy(logits, y)
    before = [p.clone() for p in model.parameters()]
    a, b = evaluate(model, loader, spy), evaluate(model, loader, spy)
    assert abs(a[0] - b[0]) < 1e-6 and a[1] == b[1] and not any(seen)
    assert all(torch.equal(p, q) for p, q in zip(before, model.parameters()))

def test_train_epoch_restores_training_mode():
    model, loader = _setup(2)
    modes = []
    def spy(logits, y):
        modes.append(model.training)
        return torch.nn.functional.cross_entropy(logits, y)
    evaluate(model, loader, spy)
    train_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.01), spy)
    assert modes[-1] is True and modes[0] is False

def test_mean_is_per_example():
    model, loader = _setup(3)
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        model.eval()
        full = sum(loss_fn(model(x), y).item() * len(x) for x, y in loader) / 512
    assert abs(evaluate(model, loader, loss_fn)[0] - full) < 1e-5
:::

:::hints
1. Two switches for evaluation: one for layer behaviour, one for autograd.
2. The batch size is 60 and there are 512 examples, so the last batch is smaller. How should you average?
:::

:::interview
- What's the difference between `model.eval()` and `torch.no_grad()`? When do you need both?
- Why `loss.item()` rather than accumulating `loss`?
- How would you add gradient accumulation over $k$ micro-batches? *(Scale each loss by $1/k$, and call `zero_grad` and `step` only every $k$ batches.)*
- What does `torch.inference_mode()` add over `no_grad`?
:::
::::

::::exercise{id=missing-zero-grad type=debug difficulty=1 time=8 runtime=local-torch udl="§6.2" tags="debugging,training-loop"}
# The loss that wouldn't settle

:::prereq
**PyTorch accumulates gradients.** Each `.backward()` *adds* into `p.grad`. This is deliberate: it lets you sum gradients over micro-batches. But it means every training step must start from cleared gradients.
:::

:::task
This `train_step` trains a linear regression. The loss drops at first, then oscillates wildly and diverges, even with a learning rate that should be safe. Find the bug.
:::

:::starter{lang=python}
import torch

def train_step(model, opt, x, y, loss_fn):
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    return loss.item()
:::

:::scaffold{lang=python}
import torch

def train_step(model, opt, x, y, loss_fn):
    {{ opt.zero_grad() | opt.zero_grad(set_to_none=True) | model.zero_grad() | model.zero_grad(set_to_none=True) ? `.backward()` **adds** into `p.grad`. Without clearing it, step $t$ uses the sum of gradients from steps $1..t$: an ever-growing, stale "momentum" that the learning rate wasn't tuned for. }}
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    return loss.item()
:::

:::solution{lang=python}
import torch

def train_step(model, opt, x, y, loss_fn):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    return loss.item()
:::

:::explain
[L4] **Clear before the backward pass.** Putting it first means the step is correct whatever happened before, for example a gradient-norm probe that called `backward` and left gradients behind.

**Why it diverges, not just slows down.** Without clearing, the update at step $t$ uses $\sum_{s\le t} g_s$. Near the optimum the recent gradients are small, but the accumulated sum isn't, so the model keeps overshooting. It behaves like momentum with $\beta = 1$ (no decay), which is unstable.

**How to catch it.** Set `lr=0` and call the step twice on the same batch. `p.grad` should be identical both times. The test does exactly that.
:::

:::tests{lang=python}
import torch

def _setup(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(64, 3)
    y = x @ torch.tensor([[1.0], [-2.0], [0.5]]) + 0.3
    return torch.nn.Linear(3, 1), x, y

def test_gradients_do_not_accumulate():
    model, x, y = _setup()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    train_step(model, opt, x, y, torch.nn.functional.mse_loss)
    g1 = model.weight.grad.clone()
    train_step(model, opt, x, y, torch.nn.functional.mse_loss)
    assert torch.allclose(model.weight.grad, g1)

def test_converges():
    model, x, y = _setup(1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(300):
        loss = train_step(model, opt, x, y, torch.nn.functional.mse_loss)
    assert loss < 1e-4
:::

:::hints
1. Print `model.weight.grad` after two consecutive calls on the same batch.
2. Which call in the five-line loop is missing?
:::

:::interview
- Why does PyTorch accumulate gradients by default? *(For gradient accumulation across micro-batches, and for summing over multiple losses.)*
- `opt.zero_grad()` versus `model.zero_grad()`: any difference? *(Not if the optimiser holds all the model's parameters.)*
- What does `set_to_none=True` change?
:::
::::

::::exercise{id=double-softmax-bug type=debug difficulty=2 time=10 runtime=local-torch udl="§5.5" tags="debugging,custom-losses"}
# The classifier that plateaus

:::prereq
**`nn.CrossEntropyLoss` expects raw logits.** Internally it computes `log_softmax` followed by `nll_loss` (Chapter 04). If the model already outputs probabilities, the loss applies a second softmax to numbers in $[0, 1]$.
:::

:::task
A 3-class classifier trains on well-separated data. The accuracy gets to 100%, but the training loss gets stuck at about **0.55** and never goes lower, even after hours. The learning-rate sweep didn't help. Find the bug, and explain why the loss can't go below that value.
:::

:::starter{lang=python}
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, n_classes), nn.Softmax(dim=-1))

    def forward(self, x):
        return self.net(x)
:::

:::scaffold{lang=python}
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), {{ nn.Linear(64, n_classes) | nn.Linear(64, n_classes, bias=True) ? Output raw **logits**. `CrossEntropyLoss` applies `log_softmax` itself, so a softmax here gets applied twice and squashes every logit into $[0, 1]$. }})

    def forward(self, x):
        return self.net(x)
:::

:::solution{lang=python}
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.net(x)
:::

:::explain
[L6] **Return logits.** If you need probabilities for reporting, apply `softmax` *outside* the model, at inference time.

**Where the floor comes from.** With the extra softmax, the loss sees "logits" $p \in [0,1]^C$. The best case is $p = (1, 0, 0)$, which gives
$$-\log\frac{e^1}{e^1 + 2e^0} = \log(1 + 2/e) \approx 0.551.$$
That's the plateau. Accuracy can still reach 100%, because the argmax is unaffected. Only the loss and the **gradients** are broken: they shrink as the inner softmax saturates, so training crawls.

**The general lesson.** Check what your loss expects. `CrossEntropyLoss` and `BCEWithLogitsLoss` take logits. `NLLLoss` takes log-probabilities. `BCELoss` takes probabilities, but is less numerically stable.
:::

:::tests{lang=python}
import torch

def _train(model, steps=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[4.0, 0.0], [-4.0, 0.0], [0.0, 4.0]])
    y = torch.arange(3).repeat_interleave(100)
    x = centres[y] + 0.5 * torch.randn(300, 2, generator=g)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    return loss.item()

def test_loss_goes_below_the_double_softmax_floor():
    torch.manual_seed(0)
    assert _train(Classifier(2, 3)) < 0.1

def test_outputs_are_logits():
    torch.manual_seed(1)
    out = Classifier(4, 5)(torch.randn(10, 4))
    assert not torch.allclose(out.sum(dim=-1), torch.ones(10))
:::

:::hints
1. Print `model(x)[:3]`. Do the rows look like logits or like probabilities?
2. What's the smallest cross-entropy you can get if every logit has to lie in $[0, 1]$?
:::

:::interview
- What does `CrossEntropyLoss` expect, and why is that design more stable?
- Why does accuracy look fine while the loss is stuck?
- Name another silent loss/activation mismatch. *(For example, `BCELoss` after a sigmoid computed in float16, or `MSELoss` on class indices.)*
:::
::::
