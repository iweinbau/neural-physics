import numpy as np
from core_math.alg import least_squares


def initial_model(z):
    """
    Calculate Alpha and Beta for the initial model by solving a least square problem for each component
    @param z: encode data vector (n_components x n)
    @return X: [alpha_m, beta_m] with m in [1, n_components]
    """

    n_components, n = z.shape
    X = np.zeros((n_components, 2))
    beta = np.zeros(n_components)

    for i in range(0, n_components):
        A = np.zeros((n - 2, 2))
        A[:, 0] = z[i, 1:n-1]
        A[:, 1] = np.diff(z[i, 0:n-1])
        b = z[i, 2:n]

        # x = [alpha_m, beta_m]
        X[i, :] = least_squares(A, b)[0]

    return X

