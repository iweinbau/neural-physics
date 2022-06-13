from typing import Tuple

import numpy as np
from neural_physics.core_math.alg import least_squares


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
