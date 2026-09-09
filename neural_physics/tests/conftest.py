import math

import numpy as np
import pytest


@pytest.fixture
def dummy_data() -> np.ndarray:
    """Random (3c x n) position matrix (no temporal structure)."""
    rng = np.random.default_rng(0)
    num_vertices, num_frames = 200, 100
    return rng.random((3 * num_vertices, num_frames))


@pytest.fixture
def oscillator_data():
    """A small deterministic 'subspace' time series with an external input: damped oscillators driven by
    a 2-d Ornstein-Uhlenbeck signal and a weak non-linear coupling. Shapes: Z (u x n), W (2 x n)."""
    rng = np.random.default_rng(0)
    n, u, fps = 3000, 6, 60
    Z, W, w = np.zeros((u, n)), np.zeros((2, n)), np.zeros(2)
    for t in range(2, n):
        w += -0.5 * w / fps + rng.normal(0, 0.15, 2) / math.sqrt(fps)
        W[:, t] = w
        for m in range(u):
            om = 0.5 + 0.4 * m
            Z[m, t] = (2 - om ** 2 / fps - 0.05 / fps) * Z[m, t - 1] - (1 - 0.05 / fps) * Z[m, t - 2] \
                + 0.002 * w[m % 2] + 0.0005 * np.tanh(Z[(m + 1) % u, t - 1])
    return Z.astype(np.float32), W.astype(np.float32), 1.0 / fps
