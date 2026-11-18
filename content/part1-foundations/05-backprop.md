---
id: ch05-backprop
summary: Backprop is the chain rule applied in the right order, with the right bookkeeping. Derive and code the backward pass of an MLP by hand, build a 40-line autograd engine, then teach PyTorch a new operation with a custom `autograd.Function` that passes `gradcheck`.
udl: "Ch. 7 — Gradients and initialisation"
---

## The chain rule, in the right order

A network is a **computational graph**: each node computes a value from its parents. To get $\partial L/\partial\theta$ for *every* parameter at once, run the graph forwards and store the intermediate values. Then sweep **backwards**: each node receives the gradient of the loss with respect to its own output (its *upstream gradient*), multiplies it by its local derivative, and passes the result to its parents.

| Forward op | Backward rule (given upstream `g` = $\partial L/\partial\,\text{out}$) |
|---|---|
| `out = a + b` | `a.grad += g`, `b.grad += g` |
| `out = a * b` | `a.grad += b * g`, `b.grad += a * g` |
| `out = X @ W` | `X.grad += g @ W.T`, `W.grad += X.T @ g` |
| `out = relu(z)` | `z.grad += g * (z > 0)` |
| `out = f(x)` (elementwise) | `x.grad += g * f'(x)` |

Two facts make this efficient. **Reverse mode** costs about one extra forward pass, however many parameters there are. And gradients **accumulate** (`+=`) whenever a value feeds more than one consumer.

**Shape rule for matmul gradients.** The gradient of a tensor always has *that tensor's* shape. For `out = X @ W`, with `X: (B, D)`, `W: (D, H)` and `g: (B, H)`, the only way to get a `(D, H)` result from `X` and `g` is `X.T @ g`. Use this to reconstruct the formula when you forget it.

::::exercise{id=mlp-backprop-numpy type=implement difficulty=3 time=25 runtime=browser udl="§7.2–7.4" tags="backprop"}
# Backprop through a two-layer MLP, by hand

:::prereq
The model and loss, for a batch of $B$ examples with integer labels $y$:
$$z_1 = XW_1 + b_1,\quad h = \mathrm{ReLU}(z_1),\quad z_2 = hW_2 + b_2,\quad L = \tfrac1B\textstyle\sum_i -\log\mathrm{softmax}(z_{2,i})_{y_i}.$$

From Chapter 04, $\partial L/\partial z_2 = (P - Y)/B$, where $P$ is the softmax and $Y$ the one-hot labels. Everything else follows from the table above:

- A **bias** is added to every row, so its gradient **sums over the batch**.
- **ReLU** passes the gradient through where $z_1 > 0$ and blocks it elsewhere.
:::

:::task
Implement `forward_backward(X, y, params)`. `params` is a dict with `W1: (D, H)`, `b1: (H,)`, `W2: (H, C)` and `b2: (C,)`. Return `(loss, grads)`, where `grads` has the same keys and shapes as `params`. Use no autograd library. The tests check every gradient against finite differences.
:::

:::starter{lang=python}
import numpy as np

def forward_backward(X, y, params):
    """X: (B, D), y: (B,) ints -> (loss: float, grads: dict with keys W1, b1, W2, b2)."""
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def forward_backward(X, y, params):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    B = X.shape[0]
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0)
    z2 = h @ W2 + b2
    z = z2 - z2.max(axis=1, keepdims=True)
    P = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    loss = -np.log(P[np.arange(B), y]).mean()

    dz2 = P.copy()
    dz2[np.arange(B), y] -= 1
    dz2 /= B
    dW2 = {{ h.T @ dz2 | np.dot(h.T, dz2) | h.T.dot(dz2) ? `W2` is `(H, C)`. Combine `h: (B, H)` with `dz2: (B, C)` so that the batch axis is summed out. }}
    db2 = {{ dz2.sum(axis=0) | dz2.sum(0) | np.sum(dz2, axis=0) ? The bias was broadcast over the batch, so its gradient sums the upstream gradient over the batch. }}
    dh = {{ dz2 @ W2.T | np.dot(dz2, W2.T) | dz2.dot(W2.T) ? Pass the gradient back *through* `W2`: `(B, C) @ (C, H)` gives the shape of `h`. }}
    dz1 = {{ dh * (z1 > 0) | dh * (h > 0) | np.where(z1 > 0, dh, 0) | dh * (z1 > 0).astype(float) ? ReLU's local derivative is 1 where the unit was active and 0 elsewhere: it *gates* the gradient. }}
    dW1 = {{ X.T @ dz1 | np.dot(X.T, dz1) | X.T.dot(dz1) ? Same pattern as `dW2`: input transposed, times the upstream gradient. }}
    db1 = dz1.sum(axis=0)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
:::

:::solution{lang=python}
import numpy as np

def forward_backward(X, y, params):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    B = X.shape[0]
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0)
    z2 = h @ W2 + b2
    z = z2 - z2.max(axis=1, keepdims=True)
    P = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    loss = -np.log(P[np.arange(B), y]).mean()

    dz2 = P.copy()
    dz2[np.arange(B), y] -= 1
    dz2 /= B
    dW2 = h.T @ dz2
    db2 = dz2.sum(axis=0)
    dh = dz2 @ W2.T
    dz1 = dh * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
:::

:::explain
[L6-11] **Forward, keeping what backward needs.** `z1` is needed for the ReLU mask, and `h` and `X` for the weight gradients. Autograd frameworks store exactly these "saved tensors". That's why training uses far more memory than inference.

[L13-15] **Softmax + cross-entropy, fused.** This is $(P - Y)/B$ without building the one-hot matrix, the result you derived in Chapter 04.

[L16-17] **Second layer.** `h.T @ dz2` sums over the batch inside the matmul. The bias gradient is the same sum without the input.

[L18-19] **Through `W2`, then through the ReLU.** `dz2 @ W2.T` routes the gradient back to each hidden unit. The mask `(z1 > 0)` zeroes it for inactive units. A unit that is inactive for *every* input never gets a gradient: that's a **dead ReLU**.

[L20-21] **First layer.** This is the same pattern again. Every linear layer's backward is these two lines, plus `dX = dz @ W.T` if there's a layer below it.
:::

:::tests{lang=python}
import numpy as np

def _setup(seed=0, B=5, D=4, H=6, C=3):
    rng = np.random.default_rng(seed)
    p = {"W1": rng.normal(size=(D, H)), "b1": rng.normal(size=H) * 0.1,
         "W2": rng.normal(size=(H, C)), "b2": rng.normal(size=C) * 0.1}
    return rng.normal(size=(B, D)), rng.integers(0, C, size=B), p

def test_grad_shapes_match_params():
    X, y, p = _setup()
    _, g = forward_backward(X, y, p)
    assert all(g[k].shape == p[k].shape for k in p)

def test_matches_finite_differences():
    X, y, p = _setup(1)
    _, g = forward_backward(X, y, p)
    eps = 1e-6
    for k in p:
        num = np.zeros_like(p[k])
        for idx in np.ndindex(*p[k].shape):
            old = p[k][idx]
            p[k][idx] = old + eps
            lp, _ = forward_backward(X, y, p)
            p[k][idx] = old - eps
            lm, _ = forward_backward(X, y, p)
            p[k][idx] = old
            num[idx] = (lp - lm) / (2 * eps)
        assert np.allclose(g[k], num, atol=1e-5), k
:::

:::hints
1. Every gradient has the shape of the thing it's the gradient of. Use that to choose where the transposes go.
2. Bias gradients: sum over the batch axis.
3. If only `W1` and `b1` fail, check the ReLU mask: it uses the *pre*-activation.
:::

:::interview
- Why do we need to store activations during the forward pass? What's the memory cost, and how does gradient checkpointing trade it for compute?
- What's a dead ReLU, and how does this backward pass show it?
- Why is reverse mode preferred over forward mode for neural networks? *(Many inputs and one scalar output: reverse mode costs one backward pass, forward mode one pass per input.)*
:::
::::

## Autograd in forty lines

::::exercise{id=micrograd type=implement difficulty=3 time=25 runtime=browser udl="§7.3" tags="autodiff,backprop"}
# A tiny scalar autograd engine

:::prereq
Each `Value` records its `data`, its `grad`, its parents, and a closure `_backward` that pushes `out.grad` into the parents using the **local derivative**.

`backward()` on the final node:

1. Builds a **topological order**: every node appears after all of its parents.
2. Sets its own `grad = 1` (because $\partial L/\partial L = 1$).
3. Calls `_backward` on the nodes in **reverse** topological order. A node's gradient is then complete before it is pushed further back.

**Accumulate, don't assign.** In `y = x*x + x*x`, the node `x` feeds four consumers. Its gradient is the *sum* of their contributions. Writing `=` instead of `+=` is the classic autograd bug.
:::

:::task
Complete `Value`, supporting `+`, `*` (including with plain Python numbers on either side), `tanh()` and `backward()`. The tests compare against hand-derived gradients and finite differences.
:::

:::starter{lang=python}
import math

class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad = float(data), 0.0
        self._parents, self._backward = parents, lambda: None

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def tanh(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    __radd__ = __add__
    __rmul__ = __mul__
:::

:::scaffold{lang=python}
import math

class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad = float(data), 0.0
        self._parents, self._backward = parents, lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def backward():
            self.grad {{ += ? A node can feed several consumers, and its gradient is the **sum** of what each one sends back. Assigning would throw away the earlier contributions. }} out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def backward():
            self.grad += other.data * out.grad
            other.grad += {{ self.data * out.grad | out.grad * self.data ? $\partial(ab)/\partial b = a$, times the upstream gradient. }}
        out._backward = backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def backward():
            self.grad += {{ (1 - t ** 2) * out.grad | (1 - t**2) * out.grad | out.grad * (1 - t ** 2) | (1 - t * t) * out.grad | (1 - out.data ** 2) * out.grad ? $\frac{d}{dx}\tanh x = 1 - \tanh^2 x$. You already have $\tanh x$ stored as `t`. }}
        out._backward = backward
        return out

    def backward(self):
        order, seen = [], set()
        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = {{ 1.0 | 1 | 1. ? The gradient of the output with respect to itself. }}
        for v in {{ reversed(order) | order[::-1] ? `order` lists parents before children. Gradients must flow from the output back towards the inputs, so walk it **backwards**. }}:
            v._backward()

    __radd__ = __add__
    __rmul__ = __mul__
:::

:::solution{lang=python}
import math

class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad = float(data), 0.0
        self._parents, self._backward = parents, lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = backward
        return out

    def backward(self):
        order, seen = [], set()
        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

    __radd__ = __add__
    __rmul__ = __mul__
:::

:::explain
[L9] **Wrap constants.** `x * 2` and `2 * x` both work: `__rmul__` handles the second case, and wrapping makes `2` a leaf `Value` whose gradient nobody reads.

[L11-13] **Addition routes the gradient unchanged** to both inputs. The `+=` is the whole point: in `x*x + x*x`, the node `x` receives four contributions.

[L20-22] **Multiplication swaps the operands.** Each input's local derivative is the *other* input's value. That's why the forward pass must keep `data` around.

[L29-31] **tanh reuses its own output.** The derivative $1 - t^2$ needs only `t`, which is already computed, so nothing extra is saved. The same trick is used for sigmoid ($\sigma(1-\sigma)$) and exp.

[L35-41] **Topological sort by depth-first search.** A node is appended only after all its parents, so `order` ends at the output. The `seen` set means shared sub-expressions are visited once.

[L42-44] **Seed and sweep.** Walking `reversed(order)` guarantees that when `v._backward()` runs, `v.grad` already holds the contributions from *all* of `v`'s consumers.
:::

:::tests{lang=python}
import math

def test_add_mul():
    a, b = Value(2.0), Value(-3.0)
    c = a * b + a
    c.backward()
    assert c.data == -4.0 and a.grad == -2.0 and b.grad == 2.0

def test_reused_node_accumulates():
    x = Value(3.0)
    y = x * x + x * x
    y.backward()
    assert x.grad == 12.0

def test_constants_on_both_sides():
    x = Value(5.0)
    y = 2 * x + 1 + x * 3
    y.backward()
    assert y.data == 26.0 and x.grad == 5.0

def test_tanh_neuron_matches_finite_differences():
    def f(w1, w2, b):
        x1, x2 = 0.5, -1.5
        return (x1 * w1 + x2 * w2 + b).tanh()
    ws = [Value(0.3), Value(-0.8), Value(0.1)]
    out = f(*ws)
    out.backward()
    eps = 1e-6
    for i, w in enumerate(ws):
        hi = [v.data + (eps if j == i else 0) for j, v in enumerate(ws)]
        lo = [v.data - (eps if j == i else 0) for j, v in enumerate(ws)]
        num = (f(*map(Value, hi)).data - f(*map(Value, lo)).data) / (2 * eps)
        assert abs(w.grad - num) < 1e-6
:::

:::hints
1. If `test_reused_node_accumulates` gives 6 instead of 12, an `=` should be a `+=`.
2. If the gradients are sometimes right and sometimes too small, check the direction you iterate over `order`.
:::

:::interview
- Why must gradients accumulate? Give a graph where `=` gives the wrong answer.
- Why a topological sort rather than plain recursion from the output? *(Recursion would push a gradient onward before the node had received all of its contributions.)*
- How does PyTorch free the graph after `.backward()`, and what does `retain_graph=True` change?
:::
::::

## Teaching PyTorch a new operation

::::exercise{id=custom-autograd-function type=implement difficulty=3 time=20 runtime=local-torch udl="§7.3" tags="autodiff,numerical-stability,pytorch"}
# A custom `autograd.Function`: stable log-sum-exp

:::prereq
**`torch.autograd.Function`** lets you supply the forward *and* backward yourself. Reasons to do it: a fused kernel, a numerically better gradient, or an op autograd can't see through.

```python
class F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):            # ctx: a scratchpad for backward
        ctx.save_for_backward(x)    # tensors needed later
        return ...
    @staticmethod
    def backward(ctx, grad_out):    # grad_out = dL/d(output)
        (x,) = ctx.saved_tensors
        return grad_out * ...       # dL/dx, the same shape as x
```

Call it with `F.apply(x)`, never `F()(x)`. **`torch.autograd.gradcheck`** compares your backward against finite differences, and needs `float64` inputs.

**The maths.** $\mathrm{LSE}(x) = \log\sum_j e^{x_j} = m + \log\sum_j e^{x_j - m}$, with $m = \max_j x_j$. Its gradient is the softmax, $e^{x_i - \mathrm{LSE}(x)}$.
:::

:::task
Implement `LogSumExp`, which reduces over the **last** axis: `(…, C) → (…)`. It must stay finite for inputs around $10^4$, and pass `gradcheck`. Expose it as `logsumexp(x)`.
:::

:::starter{lang=python}
import torch

class LogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

def logsumexp(x):
    raise NotImplementedError
:::

:::scaffold{lang=python}
import torch

class LogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        m = {{ x.max(dim=-1, keepdim=True).values | x.amax(dim=-1, keepdim=True) | x.max(-1, keepdim=True).values | x.max(dim=-1, keepdim=True)[0] | torch.amax(x, dim=-1, keepdim=True) ? The row max, kept as `(…, 1)` so it broadcasts. Note that `max(dim=…)` returns `(values, indices)`. }}
        out = m.squeeze(-1) + torch.log(torch.exp(x - m).sum(dim=-1))
        {{ ctx.save_for_backward(x, out) ? Backward needs the input *and* the output: the softmax is $e^{x - \mathrm{LSE}(x)}$, so saving `out` avoids recomputing it. }}
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, out = ctx.saved_tensors
        return {{ grad_out.unsqueeze(-1) | grad_out[..., None] | grad_out.unsqueeze(dim=-1) ? `grad_out` has the *output* shape `(…)`. Give it back the reduced axis so it broadcasts against `(…, C)`. }} * {{ torch.exp(x - out.unsqueeze(-1)) | torch.exp(x - out[..., None]) | torch.softmax(x, dim=-1) | x.softmax(dim=-1) | x.softmax(-1) | torch.softmax(x, -1) ? $\partial\,\mathrm{LSE}/\partial x_i = \mathrm{softmax}(x)_i = e^{x_i - \mathrm{LSE}(x)}$. Subtracting the LSE keeps the exponent at most 0, so it's stable. }}

def logsumexp(x):
    return {{ LogSumExp.apply(x) ? Custom Functions are invoked through the static `apply`, which wires them into the autograd graph. }}
:::

:::solution{lang=python}
import torch

class LogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        m = x.max(dim=-1, keepdim=True).values
        out = m.squeeze(-1) + torch.log(torch.exp(x - m).sum(dim=-1))
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, out = ctx.saved_tensors
        return grad_out.unsqueeze(-1) * torch.exp(x - out.unsqueeze(-1))

def logsumexp(x):
    return LogSumExp.apply(x)
:::

:::explain
[L6-7] **Stable forward.** Shifting by the row max means `exp` never sees a positive argument. `m.squeeze(-1)` drops the kept axis, so the output shape is `(…)`.

[L8] **Save what backward needs.** `ctx.save_for_backward` is the supported way to keep tensors. It lets PyTorch check for in-place modification between the forward and backward passes. Storing tensors as `ctx.x = x` bypasses that check.

[L14] **Backward is upstream × local.** The local gradient is the softmax, written as `exp(x - out)`. Because `out ≥ max(x)`, every exponent is at most 0, so it never overflows. The `unsqueeze` restores the reduced axis for broadcasting. Forgetting it is the most common shape bug in custom Functions.

[L16-17] **`apply`, not the constructor.** `LogSumExp.apply` records a node in the graph whose backward is your method.
:::

:::tests{lang=python}
import torch

def test_matches_torch():
    x = torch.randn(4, 7)
    assert torch.allclose(logsumexp(x), torch.logsumexp(x, dim=-1), atol=1e-6)

def test_huge_inputs_are_finite():
    x = torch.tensor([[1e4, 1e4 - 1.0, -1e4]])
    out = logsumexp(x)
    assert torch.isfinite(out).all() and torch.allclose(out, torch.logsumexp(x, dim=-1))

def test_gradcheck():
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(LogSumExp.apply, (x,))

def test_gradient_is_softmax():
    x = torch.randn(2, 3, 6, requires_grad=True)
    logsumexp(x).sum().backward()
    assert torch.allclose(x.grad, torch.softmax(x, dim=-1), atol=1e-6)
:::

:::hints
1. Forward: shift by the max, exactly as in Chapter 04.
2. Backward: what is $\partial\,\mathrm{LSE}/\partial x_i$? You've seen it before.
3. A shape error in backward usually means `grad_out` needs its reduced axis back.
:::

:::interview
- When would you write a custom `autograd.Function` rather than composing built-in ops?
- What does `gradcheck` do, and why does it need `float64`?
- What does `ctx.save_for_backward` give you over `ctx.x = x`?
:::
::::
