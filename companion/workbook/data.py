"""Datasets for local experiments: MNIST and Colored MNIST (Arjovsky et al., 2019, "Invariant Risk Minimization").

    from workbook.data import mnist, colored_mnist, colored_mnist_envs

    images, digits = mnist("train")                      # uint8 (60000, 28, 28), int64 (60000,)
    envs = colored_mnist_envs()                          # {"train_a": ..., "train_b": ..., "test": ...}
    x, y, color = envs["train_a"]                        # float32 (N, 2, 28, 28), int64 (N,), int64 (N,)

Files are downloaded once into ~/.cache/interview-workbook (override with WB_DATA=/path) from the PyTorch
team's public mirror, so nothing needs torchvision. Colored MNIST is built deterministically from MNIST.
"""

from __future__ import annotations

import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def cache_dir() -> Path:
    d = Path(os.environ.get("WB_DATA", Path.home() / ".cache" / "interview-workbook"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch(name: str) -> Path:
    path = cache_dir() / name
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MIRROR + name, tmp)  # noqa: S310 (fixed https URL)
        tmp.rename(path)
    return path


def _read_idx(path: Path) -> np.ndarray:
    """IDX format: 2 zero bytes, a dtype code, the number of dims, big-endian uint32 sizes, then raw data."""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def mnist(split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """-> (images uint8 (N, 28, 28), digit labels int64 (N,)). split is 'train' (60k) or 'test' (10k)."""
    img_file, lbl_file = FILES[split]
    return _read_idx(_fetch(img_file)), _read_idx(_fetch(lbl_file)).astype(np.int64)


def colored_mnist(images: np.ndarray, digits: np.ndarray, p_color_flip: float,
                  label_noise: float = 0.25, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Colored MNIST from Arjovsky et al.: a binary task where colour is a tempting shortcut.

    label  y = [digit < 5], flipped with probability `label_noise`  (so shape predicts y only 75% of the time)
    colour c = y, flipped with probability `p_color_flip`            (colour predicts y (1 - p) of the time)
    The digit is drawn into channel c of a 2-channel image (0 = red, 1 = green).

    With p_color_flip = 0.1 in training, colour is *more* predictive than shape (90% vs 75%), so ERM learns
    colour. Test with p_color_flip = 0.9 and the correlation reverses: that's the distribution shift.
    """
    rng = np.random.default_rng(seed)
    n = len(images)
    y = (digits < 5).astype(np.int64) ^ (rng.random(n) < label_noise)
    color = y ^ (rng.random(n) < p_color_flip)
    x = np.zeros((n, 2, *images.shape[1:]), dtype=np.float32)
    x[np.arange(n), color] = images.astype(np.float32) / 255.0
    return x, y.astype(np.int64), color.astype(np.int64)


def colored_mnist_envs(seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The standard three environments: two training envs (colour flips 10% and 20%) and a test env (90%)."""
    images, digits = mnist("train")
    order = np.random.default_rng(seed).permutation(len(images))
    a, b, t = order[:20000], order[20000:40000], order[40000:]
    return {
        "train_a": colored_mnist(images[a], digits[a], 0.1, seed=seed + 1),
        "train_b": colored_mnist(images[b], digits[b], 0.2, seed=seed + 2),
        "test": colored_mnist(images[t], digits[t], 0.9, seed=seed + 3),
    }
