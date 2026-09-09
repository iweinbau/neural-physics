"""Data preparation for the subspace model: the linear initial model (paper Sec. 5.2) and window
sampling for the training procedure of Sec. 5.4 / Algorithm 1."""
from __future__ import annotations

import random
from typing import Iterator, Optional, Tuple

import numpy as np
import torch

from neural_physics.core_math.alg import least_squares


def initial_model_params(subspace_z) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate Alpha and Beta for the initial model by solving a least square problem for each component
    (paper Eq. 2, t in [2, n)).
    @param subspace_z: encoded data (n_components x n), numpy array or torch tensor
    @return alphas, betas: two (n_components,) float tensors
    """
    z = subspace_z.detach().cpu().numpy() if isinstance(subspace_z, torch.Tensor) else np.asarray(subspace_z)
    z = z.astype(np.float64)
    num_components, num_frames = z.shape
    X = np.zeros((num_components, 2))

    for m in range(num_components):
        A = np.zeros((num_frames - 2, 2))
        prev_frame = num_frames - 1
        A[:, 0] = z[m, 1:prev_frame]                 # z_{t-1}
        A[:, 1] = np.diff(z[m, 0:prev_frame])        # z_{t-1} - z_{t-2}
        b = z[m, 2:num_frames]                       # z_t
        X[m, :] = least_squares(A, b)[0]

    return torch.from_numpy(X[:, 0]).float(), torch.from_numpy(X[:, 1]).float()


def initial_model_params_segments(subspace_z, segments) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same as `initial_model_params` but the regression only uses triples (t-2, t-1, t) that lie inside
    one segment, so episode boundaries do not pollute the fit."""
    z = subspace_z.detach().cpu().numpy() if isinstance(subspace_z, torch.Tensor) else np.asarray(subspace_z)
    z = z.astype(np.float64)
    rows_a, rows_b = [], []
    for s, e in np.asarray(segments).reshape(-1, 2):
        if e - s < 3:
            continue
        seg = z[:, s:e]
        rows_a.append(np.stack([seg[:, 1:-1], seg[:, 1:-1] - seg[:, :-2]], axis=2))  # (u, T-2, 2)
        rows_b.append(seg[:, 2:])                                                     # (u, T-2)
    A, b = np.concatenate(rows_a, axis=1), np.concatenate(rows_b, axis=1)
    X = np.zeros((z.shape[0], 2))
    for m in range(z.shape[0]):
        X[m] = least_squares(A[m], b[m])[0]
    return torch.from_numpy(X[:, 0]).float(), torch.from_numpy(X[:, 1]).float()


def init_model_for_frame(
    alphas: torch.Tensor,
    betas: torch.Tensor,
    z_star_prev: torch.Tensor,
    z_star_prev_prev: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate initial model z_bar = alpha ⊙ z_t−1 + beta ⊙ (z_t−1 − z_t−2)   (paper Eq. 1)
    Works for single frames (n_components,) and batches (batch, n_components).
    """
    return alphas * z_star_prev + betas * (z_star_prev - z_star_prev_prev)


def window_starts(num_frames: int, window_size: int, segments=None) -> np.ndarray:
    """All start indices of length-`window_size` windows that do not cross a segment boundary."""
    if segments is None:
        segments = [(0, num_frames)]
    starts = [np.arange(s, e - window_size + 1) for s, e in np.asarray(segments).reshape(-1, 2) if e - s >= window_size]
    if not starts:
        raise ValueError(f"no segment is at least {window_size} frames long")
    return np.concatenate(starts)


def sample_window_batch(subspace_z: torch.Tensor, subspace_w: torch.Tensor, starts: np.ndarray, batch_size: int,
                        window_size: int, generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Draw `batch_size` random windows (with replacement) -- one mini-batch of Algorithm 1.
    @param subspace_z: (n_components x num_frames)
    @param subspace_w: (n_external x num_frames)
    @param starts: allowed window start indices (see `window_starts`)
    @return z_windows (batch x window_size x n_components), w_windows (batch x window_size x n_external)
    """
    starts_t = torch.as_tensor(starts, device=subspace_z.device)
    pick = torch.randint(0, len(starts_t), (batch_size,), generator=generator, device=subspace_z.device)
    idx = starts_t[pick][:, None] + torch.arange(window_size, device=subspace_z.device)[None]   # (B, s)
    return subspace_z.T[idx], subspace_w.T[idx]


def iterate_window_batches(subspace_z: torch.Tensor, subspace_w: torch.Tensor, starts: np.ndarray, batch_size: int,
                           window_size: int, generator: Optional[torch.Generator] = None) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """One epoch: every allowed window exactly once, in random order, grouped in mini-batches."""
    perm = torch.randperm(len(starts), generator=generator).numpy()
    starts_t = torch.as_tensor(starts, device=subspace_z.device)
    ar = torch.arange(window_size, device=subspace_z.device)[None]
    for k in range(0, len(perm), batch_size):
        idx = starts_t[torch.as_tensor(perm[k:k + batch_size], device=subspace_z.device)][:, None] + ar
        yield subspace_z.T[idx], subspace_w.T[idx]


def get_windows(subspace_z: torch.Tensor, subspace_w: torch.Tensor, window_size: int = 32):
    """Sequential stride-1 windows of (n_components x window_size). Prefer `iterate_window_batches`."""
    num_components_x, num_frames = subspace_z.shape
    num_components_y, _ = subspace_w.shape
    assert num_frames == subspace_w.shape[1]
    if num_frames < window_size:
        yield subspace_z, subspace_w
        return
    for i in range(num_frames - window_size + 1):
        yield subspace_z[:, i:i + window_size], subspace_w[:, i:i + window_size]


def get_windows_random(subspace_z: torch.Tensor, subspace_w: torch.Tensor, window_size: int = 32):
    """Stride-1 windows in random order. Prefer `iterate_window_batches`."""
    num_frames = subspace_z.shape[1]
    assert num_frames == subspace_w.shape[1]
    if num_frames < window_size:
        yield subspace_z, subspace_w
        return
    for i in random.sample(range(num_frames - window_size + 1), num_frames - window_size + 1):
        yield subspace_z[:, i:i + window_size], subspace_w[:, i:i + window_size]


def get_windows_(subspace_z: torch.Tensor, window_size: int = 32):
    """Sequential windows of a single series."""
    num_frames = subspace_z.shape[1]
    if num_frames < window_size:
        yield subspace_z
        return
    for i in range(num_frames - window_size + 1):
        yield subspace_z[:, i:i + window_size]
