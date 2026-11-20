"""Offline tests for workbook.data (no download). Set WB_DOWNLOAD=1 to also test the real MNIST download."""

import os

import numpy as np
import pytest

from workbook.data import _read_idx, colored_mnist, mnist


def _fake_mnist(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 256, size=(n, 28, 28), dtype=np.uint8), rng.integers(0, 10, size=n)


def test_shapes_and_one_coloured_channel_per_image():
    images, digits = _fake_mnist(100)
    x, y, color = colored_mnist(images, digits, 0.1)
    assert x.shape == (100, 2, 28, 28) and x.dtype == np.float32 and y.shape == color.shape == (100,)
    assert np.allclose(x[np.arange(100), color], images / 255.0)
    assert (x[np.arange(100), 1 - color] == 0).all()


def test_correlations_match_the_recipe():
    images, digits = _fake_mnist()
    x, y, color = colored_mnist(images, digits, p_color_flip=0.1, label_noise=0.25)
    shape_label = (digits < 5).astype(int)
    assert abs((y == shape_label).mean() - 0.75) < 0.015
    assert abs((color == y).mean() - 0.90) < 0.015


def test_test_environment_reverses_the_colour_shortcut():
    images, digits = _fake_mnist(seed=1)
    _, y, color = colored_mnist(images, digits, p_color_flip=0.9)
    assert abs((color == y).mean() - 0.10) < 0.015


def test_deterministic_given_seed():
    images, digits = _fake_mnist(500)
    a, b = colored_mnist(images, digits, 0.2, seed=7), colored_mnist(images, digits, 0.2, seed=7)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))


def test_idx_reader(tmp_path):
    import gzip
    arr = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    header = bytes([0, 0, 8, 3]) + b"".join(d.to_bytes(4, "big") for d in arr.shape)
    path = tmp_path / "x.gz"
    with gzip.open(path, "wb") as f:
        f.write(header + arr.tobytes())
    assert np.array_equal(_read_idx(path), arr)


@pytest.mark.skipif(not os.environ.get("WB_DOWNLOAD"), reason="set WB_DOWNLOAD=1 to download MNIST")
def test_real_mnist_download():
    images, digits = mnist("test")
    assert images.shape == (10000, 28, 28) and set(np.unique(digits)) == set(range(10))
