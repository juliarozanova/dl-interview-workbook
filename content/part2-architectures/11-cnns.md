---
id: ch11-cnns
summary: A convolution is a small dense layer applied at every position with shared weights. Write conv2d without a six-deep loop, compute output sizes and receptive fields in your head, and see translation equivariance (and where striding breaks it) for yourself.
udl: "Ch. 10 — Convolutional networks"
---

## Convolution as weight sharing

A fully connected layer on a $224\times224\times3$ image to just 1,000 units needs 150 million weights, and it would have to learn "edge detector" separately at every location. A convolution builds in two **inductive biases** instead:

- **Locality.** Each output looks at a small $K\times K$ patch.
- **Weight sharing.** The same filter is used at every position. That makes the layer **translation equivariant**: shift the input, and the output shifts with it.

**Shapes** (PyTorch convention): the input is `(N, C, H, W)`, the weights `(F, C, K_h, K_w)`, and the output `(N, F, H', W')`, where
$$H' = \left\lfloor\frac{H + 2p - d(K-1) - 1}{s}\right\rfloor + 1$$
for padding $p$, stride $s$ and dilation $d$.

**Naming trap.** Deep-learning "convolution" is actually **cross-correlation**: the kernel isn't flipped. Because kernels are learned, the difference doesn't matter. But say "cross-correlation" if you're asked.

::::exercise{id=conv2d-im2col type=implement difficulty=3 time=25 runtime=browser udl="§10.2–10.3" tags="cnns,tensor-shapes"}
# conv2d with only two small loops

:::prereq
**The naive version** loops over the batch, output channels, output rows, output columns, input channels and both kernel axes: seven nested loops. That's far too slow, and too easy to get wrong in an interview.

**The im2col idea.** For each kernel offset $(i, j)$, the inputs it touches across *all* output positions form one strided slice: `x[:, :, i : i + s*H' : s, j : j + s*W' : s]`, with shape `(N, C, H', W')`. Stack those slices for every offset into `cols` of shape `(N, C, K_h, K_w, H', W')`. Then the convolution is a single contraction over $C$, $K_h$ and $K_w$:
$$\text{out}[n, f, h, w] = \sum_{c,i,j} \text{cols}[n, c, i, j, h, w]\; w[f, c, i, j] + b[f].$$
:::

:::task
Implement `conv2d(x, w, b, stride=1, padding=0)`, where `x` is `(N, C, H, W)`, `w` is `(F, C, KH, KW)` and `b` is `(F,)`. Zero-pad, then compute the cross-correlation. Loop only over the kernel offsets, and do the contraction with `np.einsum`.
:::

:::starter{lang=python}
import numpy as np

def conv2d(x, w, b, stride=1, padding=0):
    """x: (N, C, H, W), w: (F, C, KH, KW), b: (F,) -> (N, F, OH, OW)."""
    raise NotImplementedError
:::

:::scaffold{lang=python}
import numpy as np

def conv2d(x, w, b, stride=1, padding=0):
    N, C, H, W = x.shape
    F, _, KH, KW = w.shape
    x = np.pad(x, {{ ((0, 0), (0, 0), (padding, padding), (padding, padding)) | [(0, 0), (0, 0), (padding, padding), (padding, padding)] ? Pad only the two **spatial** axes, on both sides. The batch and channel axes get `(0, 0)`. }})
    OH = {{ (H + 2 * padding - KH) // stride + 1 | (H + 2*padding - KH) // stride + 1 | (x.shape[2] - KH) // stride + 1 ? Count how many kernel placements fit along the padded height, stepping by `stride`. }}
    OW = (W + 2 * padding - KW) // stride + 1
    cols = np.empty((N, C, KH, KW, OH, OW))
    for i in range(KH):
        for j in range(KW):
            cols[:, :, i, j] = x[:, :, {{ i:i + stride * OH:stride | i:i + OH * stride:stride | i:i+stride*OH:stride ? Start at offset `i`, take `OH` samples spaced `stride` apart. The end bound is the start plus `stride * OH`. }}, j:j + stride * OW:stride]
    out = np.einsum({{ "ncijhw,fcij->nfhw" | "nckluv,fckl->nfuv" | "ncabhw,fcab->nfhw" ? Contract the input channel and both kernel axes, and keep the batch, output channel and output positions. }}, cols, w)
    return out + {{ b[None, :, None, None] | b[:, None, None] | b.reshape(1, -1, 1, 1) | b.reshape(-1, 1, 1) ? One bias per **output channel**: shape it so it lines up with axis 1 of `(N, F, OH, OW)`. }}
:::

:::solution{lang=python}
import numpy as np

def conv2d(x, w, b, stride=1, padding=0):
    N, C, H, W = x.shape
    F, _, KH, KW = w.shape
    x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    cols = np.empty((N, C, KH, KW, OH, OW))
    for i in range(KH):
        for j in range(KW):
            cols[:, :, i, j] = x[:, :, i:i + stride * OH:stride, j:j + stride * OW:stride]
    out = np.einsum("ncijhw,fcij->nfhw", cols, w)
    return out + b[None, :, None, None]
:::

:::explain
[L6] **Pad the spatial axes only.** `np.pad` takes one `(before, after)` pair per axis.

[L7-8] **The output-size formula** (with dilation 1). It's worth being able to recite: $\lfloor(H + 2p - K)/s\rfloor + 1$.

[L9-12] **Gather the patches by offset.** There are only $K_h K_w$ iterations (9 for a 3×3 kernel), and each copies a strided view covering *every* output position at once. Real im2col reshapes `cols` into a `(C·K_h·K_w, N·H'·W')` matrix and calls one GEMM, which is how cuDNN-era convolutions ran on matrix-multiply hardware.

[L13] **One contraction.** Summing over `c`, `i` and `j` is exactly the dot product between each patch and each filter. The memory cost is the price: `cols` is $K_hK_w$ times bigger than the input.

[L14] **Per-channel bias.** `(F,)` becomes `(1, F, 1, 1)`. Broadcasting from Chapter 01 again.
:::

:::tests{lang=python}
import numpy as np

def _naive(x, w, b, s, p):
    x = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    N, C, H, W = x.shape
    F, _, KH, KW = w.shape
    OH, OW = (H - KH) // s + 1, (W - KW) // s + 1
    out = np.zeros((N, F, OH, OW))
    for n in range(N):
        for f in range(F):
            for h in range(OH):
                for v in range(OW):
                    out[n, f, h, v] = (x[n, :, h * s:h * s + KH, v * s:v * s + KW] * w[f]).sum() + b[f]
    return out

def test_matches_naive_basic():
    rng = np.random.default_rng(0)
    x, w, b = rng.normal(size=(2, 3, 6, 5)), rng.normal(size=(4, 3, 3, 3)), rng.normal(size=4)
    assert np.allclose(conv2d(x, w, b), _naive(x, w, b, 1, 0))

def test_matches_naive_stride_and_padding():
    rng = np.random.default_rng(1)
    x, w, b = rng.normal(size=(1, 2, 7, 8)), rng.normal(size=(3, 2, 3, 2)), rng.normal(size=3)
    out = conv2d(x, w, b, stride=2, padding=1)
    assert out.shape == (1, 3, 4, 5) and np.allclose(out, _naive(x, w, b, 2, 1))

def test_one_by_one_conv_is_a_channel_matmul():
    rng = np.random.default_rng(2)
    x, w = rng.normal(size=(2, 5, 4, 4)), rng.normal(size=(3, 5, 1, 1))
    expected = np.einsum("nchw,fc->nfhw", x, w[:, :, 0, 0])
    assert np.allclose(conv2d(x, w, np.zeros(3)), expected)
:::

:::hints
1. Write the shape of `x[:, :, i::stride, j::stride]` and compare it with `(N, C, OH, OW)`. The slice must stop after exactly `OH` samples.
2. In the einsum string, the letters for `C`, `KH` and `KW` appear in both inputs and not in the output.
:::

:::interview
- How many parameters and FLOPs does a 3×3 conv from 64 to 128 channels on a 56×56 map have? *(73,856 parameters; about 231M multiply-adds.)*
- What's a 1×1 convolution for? *(Mixing channels: a per-pixel linear layer, used for bottlenecks.)*
- Depthwise-separable convolutions: what do they factor, and how much do they save?
- Why "cross-correlation" and not convolution, and does it matter?
:::
::::

::::exercise{id=conv-arithmetic type=implement difficulty=1 time=10 runtime=browser udl="§10.2, §10.5" tags="cnns"}
# Output sizes and receptive fields

:::prereq
**Output size** with dilation $d$: the kernel's effective extent is $d(K-1)+1$, so
$$n_\text{out} = \left\lfloor \frac{n + 2p - d(K-1) - 1}{s}\right\rfloor + 1 .$$

**Receptive field** of a stack of layers: how many input pixels can influence one output pixel. Track two numbers as you go up the stack:

- the **jump** $J$: the distance, in input pixels, between neighbouring units at the current layer. Each stride multiplies it.
- the **receptive field** $R$: each layer with kernel $K$ adds $(K-1)\cdot J_\text{before}$.

Start from $R = 1$ and $J = 1$.
:::

:::task
Implement `conv_out_size(n, k, s=1, p=0, d=1)` and `receptive_field(layers)`, where `layers` is a list of `(kernel, stride)` pairs from input to output.
:::

:::starter{lang=python}
def conv_out_size(n, k, s=1, p=0, d=1):
    raise NotImplementedError

def receptive_field(layers):
    """layers: [(kernel, stride), ...] from input to output -> receptive field in input pixels."""
    raise NotImplementedError
:::

:::scaffold{lang=python}
def conv_out_size(n, k, s=1, p=0, d=1):
    return (n + 2 * p - {{ d * (k - 1) | (k - 1) * d | d*(k-1) ? A dilated kernel spans $d(K-1)+1$ pixels. Subtract its extent minus one. }} - 1) // s + 1

def receptive_field(layers):
    rf, jump = 1, 1
    for k, s in layers:
        rf += {{ (k - 1) * jump | jump * (k - 1) ? Each extra kernel tap reaches one more *unit* sideways, and one unit at this layer is `jump` input pixels. }}
        {{ jump *= s | jump = jump * s ? A stride of `s` puts neighbouring units `s` times further apart, measured in input pixels. }}
    return rf
:::

:::solution{lang=python}
def conv_out_size(n, k, s=1, p=0, d=1):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1

def receptive_field(layers):
    rf, jump = 1, 1
    for k, s in layers:
        rf += (k - 1) * jump
        jump *= s
    return rf
:::

:::explain
[L2] **The general formula.** With $d = 1$ it reduces to $\lfloor(n + 2p - k)/s\rfloor + 1$. "Same" padding for an odd kernel at stride 1 is $p = (k-1)/2$, which is the tests' 32 → 32 case.

[L5-8] **Receptive-field recursion.** Two stacked 3×3 layers see 5×5, and three see 7×7. That's the VGG argument: three 3×3 layers match a 7×7 receptive field with $27C^2$ rather than $49C^2$ weights, and add more nonlinearities. Strides early in the network make later kernels reach much further, which is why ResNet's stem (a 7×7 conv with stride 2, then 3×3 max-pool with stride 2) already sees 11 pixels.
:::

:::tests{lang=python}
def test_output_sizes():
    assert conv_out_size(224, 7, s=2, p=3) == 112
    assert conv_out_size(32, 3, s=1, p=1) == 32
    assert conv_out_size(32, 3, s=1, p=2, d=2) == 32
    assert conv_out_size(28, 5) == 24
    assert conv_out_size(7, 3, s=2) == 3

def test_receptive_fields():
    assert receptive_field([(3, 1)] * 2) == 5
    assert receptive_field([(3, 1)] * 3) == 7
    assert receptive_field([(7, 2), (3, 2)]) == 11
    assert receptive_field([(3, 2), (3, 2), (3, 2)]) == 15
:::

:::hints
1. Check your size formula on `conv_out_size(28, 5) == 24` by hand first.
2. For the receptive field, update `rf` *before* updating `jump`.
:::

:::interview
- What is the receptive field of three stacked 3×3 convolutions? Why prefer them over one 7×7?
- How does dilation grow the receptive field without losing resolution?
- The *effective* receptive field is much smaller than the theoretical one. Why? *(The influence is roughly Gaussian and concentrated in the centre.)*
:::
::::

::::exercise{id=equivariance-predict type=predict difficulty=2 time=8 runtime=browser udl="§10.1" tags="cnns"}
# Shift the input: what happens to the output?

:::task
A 1-D convolution layer (kernel size 5, stride 1, **circular** padding) maps an input signal $x$ of length 32 to an output $y$. You shift $x$ to the right by 3 positions. What happens to $y$?
:::

:::choices
- [x] $y$ shifts right by exactly 3 positions, with no other change
- [ ] $y$ is unchanged: convolutions are translation *invariant*
- [ ] $y$ changes unpredictably: the filter sees different numbers now
- [ ] $y$ shifts by 3 only if the kernel is symmetric
:::

:::experiment{lang=python}
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=32)
kernel = rng.normal(size=5)
W_dense = rng.normal(size=(32, 32)) / np.sqrt(32)

def conv1d_circular(x, kernel, stride=1):
    full = sum(kernel[i] * np.roll(x, -(i - 2)) for i in range(len(kernel)))
    return full[::stride]

def shift(v, k):
    return np.roll(v, k)

y = conv1d_circular(x, kernel)
print("conv, shift by 3 :  conv(shift(x)) == shift(conv(x))?", np.allclose(conv1d_circular(shift(x, 3), kernel), shift(y, 3)))
print("dense layer      :  W(shift(x)) == shift(W x)?       ", np.allclose(W_dense @ shift(x, 3), shift(W_dense @ x, 3)))
print("global avg pool  :  pool(conv(shift(x))) == pool(conv(x))?", np.isclose(conv1d_circular(shift(x, 3), kernel).mean(), y.mean()))
y2 = conv1d_circular(x, kernel, stride=2)
print("stride 2, shift 1:  equivariant?", np.allclose(conv1d_circular(shift(x, 1), kernel, 2), shift(y2, 1)))
print("stride 2, shift 2:  shifts output by 1?", np.allclose(conv1d_circular(shift(x, 2), kernel, 2), shift(y2, 1)))
:::

:::tests{lang=python}
import numpy as np

def test_conv_is_equivariant():
    assert np.allclose(conv1d_circular(shift(x, 3), kernel), shift(conv1d_circular(x, kernel), 3))

def test_dense_is_not():
    assert not np.allclose(W_dense @ shift(x, 3), shift(W_dense @ x, 3))

def test_pooling_makes_it_invariant():
    assert np.isclose(conv1d_circular(shift(x, 3), kernel).mean(), conv1d_circular(x, kernel).mean())

def test_stride_breaks_equivariance_for_odd_shifts():
    assert not np.allclose(conv1d_circular(shift(x, 1), kernel, 2), shift(conv1d_circular(x, kernel, 2), 1))
:::

:::explain
Every output position applies the **same** filter to its own window. Shift the input, and every window's contents move along with it, so the outputs are the same numbers, one shift along. That's **equivariance**: $\mathrm{conv}(\mathrm{shift}(x)) = \mathrm{shift}(\mathrm{conv}(x))$. With circular padding it's exact. With zero padding it holds everywhere except near the borders.

**Equivariance versus invariance.** Classification needs *invariance*, where the label shouldn't depend on position. CNNs get it by applying equivariant layers and then **pooling**: global average pooling of an equivariant feature map is exactly invariant, as the experiment shows.

**Striding breaks it.** With stride 2, a shift of 1 pixel moves the input between the sampling grid's "phases", so the output isn't a shifted copy of anything. Real CNNs are therefore much less shift-robust than the textbook story, which is the motivation for anti-aliased pooling (blur before subsampling).

A **dense layer** has no such structure. It would have to learn the same feature separately at every position, which is why an MLP needs far more data on images.
:::

:::interview
- What's the difference between translation equivariance and invariance? Which layers give which?
- Why are real CNNs not perfectly shift-invariant? *(Striding and pooling alias, and the borders are padded.)*
- How do vision transformers recover some of these inductive biases? *(Patches, position embeddings, and lots of data or augmentation.)*
:::
::::
