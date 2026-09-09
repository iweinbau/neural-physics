import numpy as np
import torch

from neural_physics.core_math.pca import PCA
from neural_physics.utils.data_preprocess import (
    initial_model_params,
    initial_model_params_segments,
    init_model_for_frame,
    iterate_window_batches,
    sample_window_batch,
    window_starts,
)


def _ar2(alpha, beta, n=300, seed=0):
    """Exact realisation of z_t = alpha z_{t-1} + beta (z_{t-1} - z_{t-2}) from random initial states."""
    rng = np.random.default_rng(seed)
    z = np.zeros((len(alpha), n))
    z[:, :2] = rng.normal(size=(len(alpha), 2))
    for t in range(2, n):
        z[:, t] = alpha * z[:, t - 1] + beta * (z[:, t - 1] - z[:, t - 2])
    return z


def test_initial_model_params_shapes(dummy_data):
    pca = PCA(8)
    pca.fit(dummy_data)
    subspace_z = pca.encode(dummy_data)
    assert subspace_z.shape == (8, dummy_data.shape[1])
    alphas, betas = initial_model_params(subspace_z)
    assert alphas.shape == (8,) and betas.shape == (8,)
    assert alphas.dtype == torch.float32


def test_initial_model_params_recovers_exact_ar2():
    alpha, beta = np.array([0.999, 0.99, 0.95]), np.array([0.98, 0.9, 0.8])
    z = _ar2(alpha, beta)
    a, b = initial_model_params(z)
    assert np.allclose(a.numpy(), alpha, atol=1e-4) and np.allclose(b.numpy(), beta, atol=1e-4)
    # torch input works too
    a2, b2 = initial_model_params(torch.tensor(z, dtype=torch.float32))
    assert np.allclose(a2.numpy(), alpha, atol=1e-3)


def test_initial_model_params_segments_ignore_boundaries():
    alpha, beta = np.array([0.99, 0.97]), np.array([0.95, 0.85])
    z1, z2 = _ar2(alpha, beta, seed=1), _ar2(alpha, beta, seed=2) * 50.0  # big jump between the two episodes
    z = np.concatenate([z1, z2], axis=1)
    a_naive, b_naive = initial_model_params(z)
    a_seg, b_seg = initial_model_params_segments(z, [(0, z1.shape[1]), (z1.shape[1], z.shape[1])])
    assert np.allclose(a_seg.numpy(), alpha, atol=1e-4) and np.allclose(b_seg.numpy(), beta, atol=1e-4)
    # the naive fit is polluted by the jump
    assert not np.allclose(a_naive.numpy(), alpha, atol=1e-4) or not np.allclose(b_naive.numpy(), beta, atol=1e-4)


def test_init_model_for_frame_batched():
    alphas, betas = torch.tensor([0.5, 1.0]), torch.tensor([1.0, 2.0])
    zp, zpp = torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    z_bar = init_model_for_frame(alphas, betas, zp, zpp)
    assert torch.allclose(z_bar, torch.tensor([[0.5 + 1.0, 2.0 + 2.0], [1.5 + 2.0, 4.0 + 6.0]]))


def test_window_starts_respect_segments():
    starts = window_starts(100, 32, segments=[(0, 40), (40, 100)])
    assert set(starts.tolist()) == set(range(0, 9)) | set(range(40, 69))
    # windows never cross 40
    assert all(not (s < 40 < s + 32) for s in starts)
    starts_all = window_starts(100, 32)
    assert len(starts_all) == 69


def test_sample_window_batch_matches_slices():
    z = torch.arange(5 * 50, dtype=torch.float32).reshape(5, 50)
    w = torch.arange(2 * 50, dtype=torch.float32).reshape(2, 50) * -1
    starts = window_starts(50, 8)
    g = torch.Generator().manual_seed(0)
    zw, ww = sample_window_batch(z, w, starts, batch_size=4, window_size=8, generator=g)
    assert zw.shape == (4, 8, 5) and ww.shape == (4, 8, 2)
    for k in range(4):
        s = int(zw[k, 0, 0].item())            # first component encodes the frame index
        assert torch.equal(zw[k], z[:, s:s + 8].T) and torch.equal(ww[k], w[:, s:s + 8].T)
    n_batches, n_windows = 0, 0
    for zb, wb in iterate_window_batches(z, w, starts, 16, 8, torch.Generator().manual_seed(1)):
        n_batches += 1
        n_windows += zb.shape[0]
    assert n_windows == len(starts) and n_batches == int(np.ceil(len(starts) / 16))
