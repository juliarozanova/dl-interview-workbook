---
id: ch03-deep-networks
summary: A ReLU network is a piecewise-linear function. Write the forward pass from scratch, predict how many linear pieces a tiny network can carve out (and why depth beats width), then build the same MLP as a proper `nn.Module` without the parameter-registration bug.
udl: "Ch. 3–4 — Shallow & deep networks"
---

## Networks as piecewise-linear functions

A layer computes $h = a(Wx + b)$: an affine map followed by an elementwise nonlinearity. With $a = \mathrm{ReLU}(z) = \max(0, z)$, each hidden unit is a "hinge" that is flat on one side and linear on the other. Summing hinges gives a **piecewise-linear** function. UDL Chapters 3–4 build the whole intuition for neural networks from this picture:

- A **shallow** network (one hidden layer) with $D$ ReLU units and a scalar input has at most $D + 1$ linear regions. Each unit adds one kink.
- A **deep** network can *fold* the input space, so the number of regions can grow **exponentially with depth** but only polynomially with width.

**Convention.** This workbook stores weights as `W: (in, out)` and computes `h @ W + b` on row-batched inputs `(B, in)`. PyTorch's `nn.Linear` stores `weight: (out, in)` and computes `x @ weight.T + bias`. It's the same maths with a transposed parameter, which is a common source of confusion when porting code.

::::exercise{id=mlp-forward-numpy type=implement difficulty=1 time=10 runtime=browser udl="§3.1, §4.3" tags="mlp,tensor-shapes"}
# An MLP forward pass in NumPy

:::prereq
For layers $\ell = 1..L$ with parameters $(W_\ell, b_\ell)$:
$$h_0 = x,\qquad h_\ell = \mathrm{ReLU}(h_{\ell-1}W_\ell + b_\ell)\ \ (\ell < L),\qquad f(x) = h_{L-1}W_L + b_L.$$
The **last layer has no activation**. Its outputs are raw scores (logits) or regression values, and the loss decides how to interpret them.
:::

:::task
Implement `relu(z)` and `mlp_forward(x, params)`, where `params` is a list of `(W, b)` pairs with `W: (in, out)` and `b: (out,)`, and `x: (B, in)`. Apply ReLU after every layer **except the last**.
:::

:::starter{lang=python}
import numpy as np

def relu(z):
    raise NotImplementedError

def mlp_forward(x, params):
    """x: (B, d_in); params: [(W1, b1), ..., (WL, bL)] with W: (in, out) -> (B, d_out)."""
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def relu(z):
    return {{ np.maximum(z, 0) | np.maximum(0, z) | np.maximum(z, 0.0) | z * (z > 0) | np.where(z > 0, z, 0) | np.clip(z, 0, None) ? Elementwise $\max(0, z)$. Note that Python's built-in `max` is not elementwise. }}

def mlp_forward(x, params):
    h = x
    for i, (W, b) in enumerate(params):
        h = {{ h @ W + b | np.dot(h, W) + b | h.dot(W) + b ? Affine map on row-batched inputs: `(B, in) @ (in, out)` plus a bias that broadcasts over the batch. }}
        if {{ i < len(params) - 1 | i != len(params) - 1 | i + 1 < len(params) ? Every layer gets a ReLU *except* the last one, whose outputs are raw scores. }}:
            h = relu(h)
    return h
:::

:::solution{lang=python}
import numpy as np

def relu(z):
    return np.maximum(z, 0)

def mlp_forward(x, params):
    h = x
    for i, (W, b) in enumerate(params):
        h = h @ W + b
        if i < len(params) - 1:
            h = relu(h)
    return h
:::

:::explain
[L4] **`np.maximum`, not `np.max`.** `np.maximum(z, 0)` compares elementwise against a broadcast scalar. `np.max(z)` reduces the whole array to one number. This mix-up is a classic in timed interviews.

[L9] **The affine map.** `b: (out,)` broadcasts across the batch, because aligning from the right puts `out` against `out`.

[L10-11] **No activation at the output.** A ReLU on the last layer would make negative logits impossible and kill the gradient for any output stuck below zero. One of the tests checks that the network can output negative values.
:::

:::tests{lang=python}
import numpy as np

def _params(sizes, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.normal(size=(a, b)), rng.normal(size=b)) for a, b in zip(sizes[:-1], sizes[1:])]

def test_relu():
    assert np.array_equal(relu(np.array([-2.0, 0.0, 3.0])), [0.0, 0.0, 3.0])

def test_shapes():
    x = np.random.default_rng(1).normal(size=(7, 4))
    assert mlp_forward(x, _params([4, 16, 8, 3])).shape == (7, 3)

def test_matches_manual_two_layer():
    (W1, b1), (W2, b2) = p = _params([3, 5, 2], seed=2)
    x = np.random.default_rng(3).normal(size=(4, 3))
    assert np.allclose(mlp_forward(x, p), np.maximum(x @ W1 + b1, 0) @ W2 + b2)

def test_no_activation_on_output():
    p = _params([2, 2], seed=4)
    x = np.random.default_rng(5).normal(size=(50, 2))
    out = mlp_forward(x, p)
    assert (out < 0).any() and np.allclose(out, x @ p[0][0] + p[0][1])
:::

:::hints
1. Loop over the layers; the only special case is the last one.
2. Use `enumerate` to know which layer you're on.
:::

:::interview
- Why do we need a nonlinearity at all? *(A composition of affine maps is affine, so depth would buy nothing.)*
- How many parameters does an MLP with sizes `[784, 256, 10]` have? *(784·256 + 256 + 256·10 + 10 = 203,530.)*
- ReLU's gradient at exactly 0? *(It's undefined. Frameworks pick 0.)*
:::
::::

## How many pieces?

::::exercise{id=linear-regions-predict type=predict difficulty=2 time=8 runtime=browser udl="§3.1–3.2, §4.4" tags="mlp"}
# Counting linear regions

:::task
A network takes a **scalar** input $x$, has **one hidden layer of 3 ReLU units**, and produces a scalar output. Over the whole real line, what is the **maximum** number of linear regions its output can have?
:::

:::choices
- [ ] 3: one per hidden unit
- [x] 4
- [ ] 6
- [ ] 8: one for each on/off pattern of the 3 units
:::

:::experiment{lang=python}
import numpy as np

def count_regions(layers, lo=-6.0, hi=6.0, n=20001):
    """Count linear pieces of a scalar->scalar ReLU net by tracking hidden on/off patterns along x."""
    x = np.linspace(lo, hi, n)[:, None]
    patterns, h = [], x
    for W, b in layers[:-1]:
        pre = h @ W + b
        patterns.append(pre > 0)
        h = np.maximum(pre, 0)
    pattern = np.concatenate(patterns, axis=1)
    changes = np.any(pattern[1:] != pattern[:-1], axis=1).sum()
    return int(changes) + 1

def random_net(sizes, rng):
    return [(rng.normal(size=(a, b)), rng.normal(size=b)) for a, b in zip(sizes[:-1], sizes[1:])]

rng = np.random.default_rng(0)
shallow = [count_regions(random_net([1, 3, 1], rng)) for _ in range(300)]
deep = [count_regions(random_net([1, 3, 3, 1], rng)) for _ in range(300)]
wide = [count_regions(random_net([1, 6, 1], rng)) for _ in range(300)]
print("1 hidden layer, 3 units   max regions:", max(shallow))
print("1 hidden layer, 6 units   max regions:", max(wide))
print("2 hidden layers, 3 + 3    max regions:", max(deep))
:::

:::tests{lang=python}
import numpy as np

def test_shallow_three_units_max_is_four():
    rng = np.random.default_rng(0)
    counts = [count_regions(random_net([1, 3, 1], rng)) for _ in range(300)]
    assert max(counts) == 4

def test_depth_beats_the_shallow_bound():
    rng = np.random.default_rng(1)
    counts = [count_regions(random_net([1, 3, 3, 1], rng)) for _ in range(300)]
    assert max(counts) > 4
:::

:::explain
Each hidden unit computes $\mathrm{ReLU}(w_i x + b_i)$, which has exactly **one kink**, at $x = -b_i/w_i$. The output is a weighted sum of these hinges, so it can only change slope at those three points. Three kinks split the line into at most $3 + 1 = 4$ pieces. In general, $D$ units on a scalar input give at most $D+1$ regions (UDL §3.2).

**Why not 8?** There are $2^3$ on/off *patterns*, but along a line each unit switches only once, so you only ever visit 4 of them, in order. In higher dimensions the hyperplanes can cut the input space into more regions, but still only polynomially many in the width.

**Depth.** The second layer acts on the first layer's *piecewise-linear* output, and can create new kinks inside every existing piece. In the experiment, the best $3+3$ network reaches 10 regions. That's more than a single layer of 6 units can *ever* reach (7), with the same number of hidden units. On *average*, though, random deep networks have fewer regions than the wide one: capacity is not the same as typical behaviour. With more layers the count can grow exponentially in depth (UDL §4.4). That's the standard argument that depth is more *expressive* per parameter.

**Caveat.** That's the *maximum*. Randomly initialised networks usually have far fewer regions than their capacity allows, and expressivity alone doesn't explain why depth helps in practice.
:::

:::interview
- How many regions can a shallow network with $D$ units and scalar input have? *($D + 1$.)*
- Why might depth be more parameter-efficient than width?
- Do trained networks actually use exponentially many regions? *(Empirically, far fewer than the bound.)*
:::
::::

## The same model as an `nn.Module`

::::exercise{id=mlp-module type=implement difficulty=2 time=15 runtime=local-torch udl="§4.3" tags="mlp,pytorch"}
# A configurable MLP in PyTorch

:::prereq
**Sub-modules must be registered.** `nn.Module` finds parameters by scanning attributes that are themselves `nn.Module`s or `nn.Parameter`s. A **plain Python list** of layers is invisible to it, so `model.parameters()` misses them: the optimiser never updates them, `.to(device)` never moves them, and `state_dict()` never saves them. Use `nn.Sequential` or `nn.ModuleList`.

**`nn.Linear(in, out)`** stores `weight: (out, in)` and computes `x @ weight.T + bias`.

**Activations as modules.** Pass the *class* (e.g. `nn.ReLU`) and instantiate a fresh one per layer with `activation()`.
:::

:::task
Implement `MLP(sizes, activation=nn.ReLU)`, where `sizes = [d_in, h1, ..., d_out]`. Put an activation between consecutive linear layers, but **not after the last** one. Store everything in `self.net`, an `nn.Sequential`.
:::

:::starter{lang=python}
import torch
import torch.nn as nn

class MLP(nn.Module):
    """sizes = [d_in, h1, ..., d_out]; activation between layers, none at the end."""

    def __init__(self, sizes, activation=nn.ReLU):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
:::

:::scaffold{lang=python}
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, sizes, activation=nn.ReLU):
        super().__init__()
        layers = []
        for i, (a, b) in enumerate({{ zip(sizes[:-1], sizes[1:]) | zip(sizes, sizes[1:]) ? Consecutive pairs of sizes: `(d_in, h1), (h1, h2), …`. }}):
            layers.append({{ nn.Linear(a, b) | nn.Linear(in_features=a, out_features=b) ? A fully connected layer from `a` inputs to `b` outputs. }})
            if {{ i < len(sizes) - 2 | i != len(sizes) - 2 | i + 2 < len(sizes) ? There are `len(sizes) - 1` linear layers, and only the last one (index `len(sizes) - 2`) has no activation. }}:
                layers.append(activation())
        self.net = {{ nn.Sequential(*layers) | nn.Sequential(*list(layers)) ? A plain list is invisible to `.parameters()`. `nn.Sequential` registers each layer **and** chains them in `forward`. }}

    def forward(self, x):
        return self.net(x)
:::

:::solution{lang=python}
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, sizes, activation=nn.ReLU):
        super().__init__()
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
:::

:::explain
[L8] **Pairs of sizes.** `zip(sizes[:-1], sizes[1:])` yields `(d_in, h1), (h1, h2), …, (h_k, d_out)`: one pair per linear layer.

[L9-11] **An activation between layers, not after the last.** `activation()` creates a *new* module each time. That matters for modules with state, such as `nn.PReLU`, which has a learnable slope.

[L12] **Registration.** If you write `self.layers = layers` (a plain list), `forward` still works, but `model.parameters()` returns nothing. The optimiser then silently trains nothing, and the loss stays flat. The tests check that every parameter is registered.

[L14-15] **Forward.** `nn.Sequential` chains the modules, so `forward` is one line. Use `nn.ModuleList` instead when you need custom wiring, such as skip connections.
:::

:::tests{lang=python}
import torch
import torch.nn as nn

def test_parameter_count_and_registration():
    m = MLP([4, 16, 16, 3])
    assert sum(p.numel() for p in m.parameters()) == 4 * 16 + 16 + 16 * 16 + 16 + 16 * 3 + 3
    assert len(list(m.parameters())) == 6

def test_output_shape_and_no_final_activation():
    torch.manual_seed(0)
    out = MLP([5, 32, 2])(torch.randn(64, 5))
    assert out.shape == (64, 2) and (out < 0).any()

def test_matches_manual_forward():
    torch.manual_seed(1)
    m, x = MLP([3, 4, 2], activation=nn.Tanh), torch.randn(6, 3)
    l1, l2 = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
    assert torch.allclose(m(x), torch.tanh(x @ l1.weight.T + l1.bias) @ l2.weight.T + l2.bias, atol=1e-6)

def test_trains():
    torch.manual_seed(2)
    m = MLP([2, 16, 1])
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x = torch.randn(64, 2)
    y = x.sum(dim=1, keepdim=True)
    for _ in range(200):
        opt.zero_grad()
        loss = ((m(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
:::

:::hints
1. Count the linear layers: `len(sizes) - 1`. Which index is the last one?
2. If `test_parameter_count_and_registration` reports 0 parameters, you stored a plain Python list.
:::

:::interview
- `nn.Sequential` versus `nn.ModuleList` versus a plain list: when would you use each?
- Why pass the activation *class* rather than an instance?
- What does `model.to("cuda")` move, and what silently stays behind? *(Registered parameters and buffers move; plain tensor attributes stay behind.)*
:::
::::
