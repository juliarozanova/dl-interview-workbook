---
id: ch12-rnns
summary: Recurrent networks reuse one weight matrix across time, so gradients get multiplied by it again and again. Code backprop through time by hand, predict how fast gradients vanish over 50 steps, and build an LSTM cell that matches PyTorch's exactly.
udl: "—"
---

## One matrix, applied T times

A vanilla RNN reads a sequence one step at a time, updating a hidden state:
$$h_t = \tanh(x_t W_x + h_{t-1} W_h + b).$$
The **same** $W_x$, $W_h$ and $b$ are used at every step, which is weight sharing across time, as a CNN shares weights across space. Training unrolls the loop and backpropagates through it: **backprop through time (BPTT)**.

**The problem.** $\partial h_T/\partial h_0$ is a product of $T$ Jacobians, each containing $W_h$. If $W_h$'s singular values sit below 1, the product shrinks exponentially. Above 1, it grows. Either way, learning long-range dependencies is hard. **LSTMs and GRUs** fix this with *gates* and an *additive* cell-state path.

> UDL focuses on transformers rather than RNNs, so this chapter is self-contained. RNNs still come up in interviews: as a contrast to attention, in state-space models, and in streaming or online settings.

::::exercise{id=rnn-bptt type=implement difficulty=3 time=25 runtime=browser udl="—" tags="rnns,backprop"}
# Backprop through time, by hand

:::prereq
**Forward**, for $t = 1..T$: $z_t = x_tW_x + h_{t-1}W_h + b$ and $h_t = \tanh(z_t)$. Keep every $h_t$.

**Backward**, sweeping $t = T..1$. Each $h_t$ receives gradient from two places: the loss directly (`dH[t]`) and the next step's recurrence. So
$$\delta h_t = \texttt{dH}[t] + \underbrace{\delta z_{t+1} W_h^\top}_{\text{from step } t+1},\qquad \delta z_t = \delta h_t \odot (1 - h_t^2).$$
The weights are shared, so their gradients **accumulate across time**: $\nabla W_h \mathrel{+}= h_{t-1}^\top\,\delta z_t$, and the same for $W_x$ and $b$.
:::

:::task
`rnn_forward(xs, h0, Wx, Wh, b)` is given. Implement `rnn_backward(xs, h0, Wx, Wh, b, hs, dH)`, which returns `(dWx, dWh, db, dh0)`. The shapes are `xs: (T, B, I)`, `h0: (B, H)`, `hs: (T, B, H)` (from forward), and `dH: (T, B, H)` (upstream gradients on every hidden state).
:::

:::starter{lang=python}
import numpy as np

def rnn_forward(xs, h0, Wx, Wh, b):
    hs, h = [], h0
    for x in xs:
        h = np.tanh(x @ Wx + h @ Wh + b)
        hs.append(h)
    return np.stack(hs)

def rnn_backward(xs, h0, Wx, Wh, b, hs, dH):
    """-> (dWx, dWh, db, dh0)."""
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def rnn_forward(xs, h0, Wx, Wh, b):
    hs, h = [], h0
    for x in xs:
        h = np.tanh(x @ Wx + h @ Wh + b)
        hs.append(h)
    return np.stack(hs)

def rnn_backward(xs, h0, Wx, Wh, b, hs, dH):
    dWx, dWh, db = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)
    dh_next = np.zeros_like(h0)
    for t in reversed(range(len(xs))):
        h_prev = hs[t - 1] if t > 0 else h0
        dh = {{ dH[t] + dh_next | dh_next + dH[t] ? $h_t$ affects the loss directly **and** through $h_{t+1}$. Add both contributions. }}
        dz = dh * {{ (1 - hs[t] ** 2) | (1 - hs[t]**2) | (1 - hs[t] * hs[t]) ? $\tanh'(z) = 1 - \tanh^2(z)$, and $\tanh(z_t) = h_t$ is already stored. }}
        dWx += xs[t].T @ dz
        dWh += {{ h_prev.T @ dz | np.dot(h_prev.T, dz) ? The recurrent weight multiplied the **previous** state: `(H, B) @ (B, H)`. And `+=`, because the same $W_h$ was used at every step. }}
        db += dz.sum(axis=0)
        dh_next = {{ dz @ Wh.T | np.dot(dz, Wh.T) ? Send the gradient back to $h_{t-1}$ through $W_h$. That's the term that gets multiplied $T$ times, and the source of vanishing gradients. }}
    return dWx, dWh, db, dh_next
:::

:::solution{lang=python}
import numpy as np

def rnn_forward(xs, h0, Wx, Wh, b):
    hs, h = [], h0
    for x in xs:
        h = np.tanh(x @ Wx + h @ Wh + b)
        hs.append(h)
    return np.stack(hs)

def rnn_backward(xs, h0, Wx, Wh, b, hs, dH):
    dWx, dWh, db = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)
    dh_next = np.zeros_like(h0)
    for t in reversed(range(len(xs))):
        h_prev = hs[t - 1] if t > 0 else h0
        dh = dH[t] + dh_next
        dz = dh * (1 - hs[t] ** 2)
        dWx += xs[t].T @ dz
        dWh += h_prev.T @ dz
        db += dz.sum(axis=0)
        dh_next = dz @ Wh.T
    return dWx, dWh, db, dh_next
:::

:::explain
[L12-13] **Walk backwards in time.** `dh_next` carries the gradient flowing into $h_t$ from the future. It starts at zero, because nothing comes after step $T$.

[L14-16] **Two sources of gradient, then through the tanh.** Forgetting `dh_next` is the classic bug: it trains the model with "truncated BPTT of length 1" and nobody notices until long-range tasks fail.

[L17-19] **Shared weights accumulate.** This is the same `+=` story as micrograd in Chapter 05: one parameter, many uses.

[L20] **The recurrence.** Each backward step multiplies by $W_h^\top$ and by the tanh derivative ($\le 1$). Over $T$ steps that product shrinks or explodes, which the next exercise measures. When the loop ends, `dh_next` is $\partial L/\partial h_0$.

**Truncated BPTT** stops this loop after $k$ steps to bound memory and cost, which trades away long-range credit assignment.
:::

:::tests{lang=python}
import numpy as np

def _setup(seed=0, T=5, B=3, I=4, H=6):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(T, B, I)), rng.normal(size=(B, H)) * 0.5,
            rng.normal(size=(I, H)) * 0.5, rng.normal(size=(H, H)) * 0.5, rng.normal(size=H) * 0.1,
            rng.normal(size=(T, B, H)))

def test_matches_finite_differences():
    xs, h0, Wx, Wh, b, R = _setup()
    hs = rnn_forward(xs, h0, Wx, Wh, b)
    grads = rnn_backward(xs, h0, Wx, Wh, b, hs, R)
    params = [Wx, Wh, b, h0]
    loss = lambda: (rnn_forward(xs, h0, Wx, Wh, b) * R).sum()
    eps = 1e-6
    for P, G in zip(params, grads):
        num = np.zeros_like(P)
        for idx in np.ndindex(*P.shape):
            old = P[idx]
            P[idx] = old + eps
            hi = loss()
            P[idx] = old - eps
            lo = loss()
            P[idx] = old
            num[idx] = (hi - lo) / (2 * eps)
        assert np.allclose(G, num, atol=1e-5)

def test_gradient_reaches_through_time():
    xs, h0, Wx, Wh, b, _ = _setup(1)
    hs = rnn_forward(xs, h0, Wx, Wh, b)
    dH = np.zeros_like(hs)
    dH[-1] = 1.0
    _, _, _, dh0 = rnn_backward(xs, h0, Wx, Wh, b, hs, dH)
    assert np.abs(dh0).sum() > 0
:::

:::hints
1. If only `dh0` is wrong, check what you return: the gradient flowing *out* of step 0.
2. If the weight gradients are too small, you probably overwrote (`=`) instead of accumulating (`+=`), or dropped `dh_next`.
:::

:::interview
- Why do the weight gradients accumulate across time steps?
- What's truncated BPTT, and what does it give up?
- How does gradient clipping (Chapter 07) help RNN training specifically? *(It contains the exploding case. The vanishing case needs gating.)*
:::
::::

::::exercise{id=rnn-vanishing-predict type=predict difficulty=2 time=6 runtime=browser udl="—" tags="rnns,vanishing-gradients"}
# Fifty steps through a recurrent matrix

:::task
A **linear** RNN, $h_t = W h_{t-1}$, has $W = 0.9\,Q$ with $Q$ orthogonal, so every singular value of $W$ is exactly $0.9$. What is the spectral norm of the Jacobian $\partial h_{50}/\partial h_0$?
:::

:::choices
- [ ] About $0.9$: each step only shrinks it slightly
- [ ] About $0.45$
- [x] About $5\times 10^{-3}$
- [ ] Exactly $0$
:::

:::experiment{lang=python}
import numpy as np

def jacobian_norm(rho, T=50, H=32, nonlin=False, seed=0):
    """‖∂h_T/∂h_0‖₂ for h_t = f(W h_{t-1}) with W = rho * (orthogonal)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(H, H)))
    W = rho * Q
    h, J = rng.normal(size=H) * 0.5, np.eye(H)
    for _ in range(T):
        h = np.tanh(W @ h) if nonlin else W @ h
        D = np.diag(1 - h ** 2) if nonlin else np.eye(H)
        J = D @ W @ J
    return np.linalg.norm(J, 2)

for rho in (0.9, 1.0, 1.1):
    print(f"rho={rho}:  linear {jacobian_norm(rho):9.3g}   tanh {jacobian_norm(rho, nonlin=True):9.3g}   (rho^50 = {rho ** 50:.3g})")
:::

:::tests{lang=python}
import numpy as np

def test_linear_is_exactly_rho_to_the_T():
    assert np.isclose(jacobian_norm(0.9), 0.9 ** 50, rtol=1e-6)

def test_linear_explodes_above_one():
    assert jacobian_norm(1.1) > 100

def test_tanh_saturation_tames_the_explosion():
    assert jacobian_norm(1.1, nonlin=True) < 10
:::

:::explain
The Jacobian is the product $W^{50}$. With every singular value equal to $0.9$, that's exactly $0.9^{50} \approx 0.0052$. A signal from 50 steps back arrives about 200× weaker, and at 100 steps it's about 40,000× weaker. So the gradient for "what happened at the start" is buried under the gradients from recent steps.

At $\rho = 1.1$ a linear RNN **explodes** ($1.1^{50} \approx 117$). With **tanh**, the derivative $1 - h^2 \le 1$ shrinks each step, and saturation caps the growth. Even at $\rho = 1$ the tanh network *loses* signal. Nonlinear RNNs therefore lean towards vanishing.

**The LSTM fix.** The cell state updates **additively**: $c_t = f_t\odot c_{t-1} + i_t\odot g_t$, so $\partial c_t/\partial c_{t-1} = f_t$, a *learned* gate. If the forget gate stays near 1, the gradient passes through many steps almost untouched. That's the same identity-path idea as residual connections (Chapter 10), discovered 20 years earlier.
:::

:::interview
- Why do vanilla RNNs struggle with long-range dependencies? Give the Jacobian argument.
- How does the LSTM's cell state avoid vanishing gradients? What role does the forget gate play?
- Why initialise the forget-gate bias to a positive value? *(The gates start "open", so gradients flow from the start of training.)*
- How do transformers sidestep this entirely? *(Attention connects any two positions in one step, with path length $O(1)$.)*
:::
::::

::::exercise{id=lstm-cell type=implement difficulty=2 time=15 runtime=local-torch udl="—" tags="rnns,pytorch"}
# An LSTM cell that matches `nn.LSTMCell`

:::prereq
With $a = x W_{ih}^\top + b_{ih} + h W_{hh}^\top + b_{hh}$ split into four equal chunks, $[a_i, a_f, a_g, a_o]$:
$$i = \sigma(a_i),\quad f = \sigma(a_f),\quad g = \tanh(a_g),\quad o = \sigma(a_o)$$
$$c' = f\odot c + i\odot g,\qquad h' = o\odot\tanh(c').$$
- $i$ (the **input gate**) decides how much new content to write.
- $f$ (the **forget gate**) decides how much of the old cell to keep.
- $g$ is the **candidate** content.
- $o$ (the **output gate**) decides how much of the cell to expose.

PyTorch stacks the four gates in exactly this order, **i, f, g, o**, which is what lets the tests copy weights between your cell and `nn.LSTMCell`.
:::

:::task
Implement `LSTMCell(input_size, hidden_size)` with two layers, `self.ih = nn.Linear(input_size, 4*hidden_size)` and `self.hh = nn.Linear(hidden_size, 4*hidden_size)`, and `forward(x, (h, c)) -> (h', c')`.
:::

:::starter{lang=python}
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, state):
        raise NotImplementedError
:::

:::scaffold{lang=python}
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.ih = nn.Linear(input_size, 4 * hidden_size)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = (self.ih(x) + self.hh(h)).{{ chunk(4, dim=-1) | chunk(4, -1) | chunk(4, dim=1) | split(self.hh.in_features, dim=-1) ? One matmul computes all four gates at once. Split the last axis into four equal parts, in PyTorch's order: i, f, g, o. }}
        c = {{ torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g) | f.sigmoid() * c + i.sigmoid() * g.tanh() | torch.sigmoid(i) * torch.tanh(g) + torch.sigmoid(f) * c ? Forget part of the old cell, then write new candidate content through the input gate. It's **additive**, and that's the whole point. }}
        h = {{ torch.sigmoid(o) * torch.tanh(c) | o.sigmoid() * c.tanh() | torch.sigmoid(o) * c.tanh() ? Expose a squashed view of the cell, filtered by the output gate. }}
        return h, c
:::

:::solution{lang=python}
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.ih = nn.Linear(input_size, 4 * hidden_size)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = (self.ih(x) + self.hh(h)).chunk(4, dim=-1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c
:::

:::explain
[L7-8] **Two fused projections.** Each produces all four gates' pre-activations in one matmul, so the cell has $4H(I + H) + 8H$ parameters. PyTorch keeps two separate biases too, which is redundant but conventional.

[L12] **Split in PyTorch's order.** If the order is wrong, the shapes still work and the cell still trains, but the tests that compare against `nn.LSTMCell` fail. It's a silent bug of exactly the kind interviews probe.

[L13] **The additive cell update.** $\partial c'/\partial c = \sigma(a_f)$, elementwise. That's the gated identity path that keeps gradients alive over long sequences.

[L14] **Output.** $h'$ is what the next layer (and the next time step's gates) see. $c'$ is the private memory.

**GRU, for comparison.** It merges the forget and input gates into one *update* gate and drops the separate cell. That gives three gate blocks instead of four, with similar performance.
:::

:::tests{lang=python}
import torch

def _pair(I=5, H=7, seed=0):
    torch.manual_seed(seed)
    mine, ref = LSTMCell(I, H), torch.nn.LSTMCell(I, H)
    with torch.no_grad():
        ref.weight_ih.copy_(mine.ih.weight)
        ref.bias_ih.copy_(mine.ih.bias)
        ref.weight_hh.copy_(mine.hh.weight)
        ref.bias_hh.copy_(mine.hh.bias)
    return mine, ref

def test_matches_torch_over_a_sequence():
    mine, ref = _pair()
    xs = torch.randn(6, 3, 5)
    s1 = s2 = (torch.zeros(3, 7), torch.zeros(3, 7))
    for x in xs:
        s1, s2 = mine(x, s1), ref(x, s2)
    assert torch.allclose(s1[0], s2[0], atol=1e-6) and torch.allclose(s1[1], s2[1], atol=1e-6)

def test_parameter_count():
    assert sum(p.numel() for p in LSTMCell(5, 7).parameters()) == 4 * 7 * (5 + 7) + 8 * 7

def test_gradients_flow_to_initial_cell():
    mine, _ = _pair(seed=1)
    c0 = torch.zeros(2, 7, requires_grad=True)
    h, c = mine(torch.randn(2, 5), (torch.zeros(2, 7), c0))
    (h.sum() + c.sum()).backward()
    assert c0.grad is not None and c0.grad.abs().sum() > 0
:::

:::hints
1. One combined pre-activation, then `.chunk(4, dim=-1)`.
2. Which gates use sigmoid, and which use tanh? Only the candidate $g$ (and the final squash of $c$) use tanh.
:::

:::interview
- Write the LSTM equations from memory. What does each gate do?
- LSTM versus GRU: what's merged, and what's the parameter count difference?
- Why is the cell update additive, and what does that buy?
- Where do RNNs still beat transformers? *(Streaming, constant memory per step, very long sequences, and state-space models such as Mamba.)*
:::
::::
