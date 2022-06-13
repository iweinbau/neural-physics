from typing import Tuple

import numpy as np
from neural_physics.core_math.alg import least_squares


def get_windows(subspace_z: np.ndarray, window_size: int = 32) -> np.ndarray:
    """
    Split the subspace_z into windows of size window_size
    @param subspace_z: (n_components x num_frames) matrix with the subspace_z
    @param window_size: size of the window
    @return iterator yielding windows of shape (n_components x window_size)
    """
    num_components, num_frames = subspace_z.shape

    if num_frames < window_size:
        yield subspace_z

    # Window size must be a divisor of the number of frames.
    if num_frames % window_size != 0:
        # drop remainder of frames
        remainder = num_frames % window_size
        subspace_z = subspace_z[:, :-remainder]

    for subspace_z_window in np.array_split(
        subspace_z, num_frames // window_size, axis=1
    ):
        assert subspace_z_window.shape == (num_components, window_size)
        yield subspace_z_window


def initial_model_params(subspace_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    return alphas, betas


def init_model_for_frame(
    alphas: np.ndarray,
    betas: np.ndarray,
    z_star_prev: np.ndarray,
    z_star_prev_prev: np.ndarray,
) -> np.ndarray:
    """
    Calculate initial model z_bar = alpha ⊙ z_t−1 + beta ⊙ (zt−1 − zt−2)
    @param alpha: (n_components x 1) vector with the alpha factors
    @param beta: (n_components x 1) vector with the beta factors
    @return z_bar: (n_components x 1) initial model vector for current frame
    """
    z_bar = alphas * z_star_prev + betas * (z_star_prev - z_star_prev_prev)
    return z_bar
