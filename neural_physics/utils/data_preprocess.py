import random
from typing import Tuple

import numpy as np
import torch
from neural_physics.core_math.alg import least_squares

def get_windows_random(subspace_z: torch.Tensor, subspace_w: torch.Tensor, window_size: int = 32) -> torch.Tensor:
    """
    Split the subspace_z into windows of size window_size and stide 1
    @param subspace_z: (n_components x num_frames) matrix with the subspace_z
    @param window_size: size of the window
    @return iterator yielding windows of shape (n_components x window_size)
    """
    num_components_x, num_frames = subspace_z.shape
    num_components_y, num_frames = subspace_w.shape
    assert num_frames == subspace_w.shape[1]

    if num_frames < window_size:
        yield subspace_z, subspace_w

    for i in random.sample(range(num_frames - window_size + 1), num_frames - window_size + 1):
        subspace_z_window = subspace_z[:, i:i+window_size]
        subspace_w_window = subspace_w[:, i:i+window_size]
        assert subspace_z_window.shape == (num_components_x, window_size)
        assert subspace_w_window.shape == (num_components_y, window_size)
        yield subspace_z_window, subspace_w_window

def get_windows(subspace_z: torch.Tensor, subspace_w: torch.Tensor, window_size: int = 32) -> torch.Tensor:
    """
    Split the subspace_z into windows of size window_size and stide 1
    @param subspace_z: (n_components x num_frames) matrix with the subspace_z
    @param window_size: size of the window
    @return iterator yielding windows of shape (n_components x window_size)
    """
    num_components_x, num_frames = subspace_z.shape
    num_components_y, num_frames = subspace_w.shape
    assert num_frames == subspace_w.shape[1]

    if num_frames < window_size:
        yield subspace_z, subspace_w

    for i in range(num_frames- window_size + 1):
        subspace_z_window = subspace_z[:, i:i+window_size]
        subspace_w_window = subspace_w[:, i:i+window_size]
        assert subspace_z_window.shape == (num_components_x, window_size)
        assert subspace_w_window.shape == (num_components_y, window_size)
        yield subspace_z_window, subspace_w_window

def get_windows_(subspace_z: torch.Tensor, window_size: int = 32) -> torch.Tensor:
    """
    Split the subspace_z into windows of size window_size and stide 1
    @param subspace_z: (n_components x num_frames) matrix with the subspace_z
    @param window_size: size of the window
    @return iterator yielding windows of shape (n_components x window_size)
    """
    num_components_x, num_frames = subspace_z.shape

    if num_frames < window_size:
        yield subspace_z
        
    for i in range(num_frames- window_size + 1):
        subspace_z_window = subspace_z[:, i:i+window_size]
        assert subspace_z_window.shape == (num_components_x, window_size)
        yield subspace_z_window


def initial_model_params(subspace_z: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate Alpha and Beta for the initial model by solving a least square problem for each component
    @param z: encode data vector (n_components x n)
    @return X: [alpha_m, beta_m] with m in [1, n_components]
    """

    num_components, num_frames = subspace_z.shape
    X = np.zeros((num_components, 2))

    for m in range(0, num_components):
        A = np.zeros((num_frames - 2, 2))
        prev_frame = num_frames - 1
        A[:, 0] = subspace_z[m, 1:prev_frame]
        A[:, 1] = np.diff(subspace_z[m, 0:prev_frame])
        b = subspace_z[m, 2:num_frames]

        # x = [alpha_m, beta_m]
        X[m, :] = least_squares(A, b)[0]

    alphas = X[:, 0]
    betas = X[:, 1]

    return torch.from_numpy(alphas).float(), torch.from_numpy(betas).float()


def init_model_for_frame(
    alphas: torch.Tensor,
    betas: torch.Tensor,
    z_star_prev: torch.Tensor,
    z_star_prev_prev: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate initial model z_bar = alpha ⊙ z_t−1 + beta ⊙ (zt−1 − zt−2)
    @param alpha: (n_components x 1) vector with the alpha factors
    @param beta: (n_components x 1) vector with the beta factors
    @return z_bar: (n_components x 1) initial model vector for current frame
    """
    z_bar = alphas * z_star_prev + betas * (z_star_prev - z_star_prev_prev)
    return z_bar

