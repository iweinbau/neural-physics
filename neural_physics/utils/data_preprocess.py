import numpy as np
from core_math.alg import least_squares


def initial_model_param(z):
    """
    Calculate Alpha and Beta for the initial model by solving a least square problem for each component
    @param z: encode data vector (n_components x n)
    @return X: [alpha_m, beta_m] with m in [1, n_components]
    """

    n_components, n = z.shape
    X = np.zeros((n_components, 2))

    for i in range(0, n_components):
        A = np.zeros((n - 2, 2))
        A[:, 0] = z[i, 1:n-1]
        A[:, 1] = np.diff(z[i, 0:n-1])
        b = z[i, 2:n]

        # x = [alpha_m, beta_m]
        X[i, :] = least_squares(A, b)[0]

    return X


def init_model(z, alpha, beta):
    """
    Calculate initial model z_bar = alpha ⊙ z_t−1 + beta ⊙ (zt−1 − zt−2)
    @param alpha: (n_components x 1) vector with the alpha factors
    @param beta: (n_components x 1) vector with the beta factors
    @param z: (n_components x n) input data. Where n_components are pca components and n the number of data samples
    @return z_bar: (n_components x n-2) initial model vector
    """
    n_components, n = z.shape
    z_bar = np.zeros((n_components, n-2))
    for i in range(2, n):
        z_bar[:, i-2] = alpha * z[:, i-1] + beta * (z[:, i-1] - z[:, i-2])
    return z_bar
