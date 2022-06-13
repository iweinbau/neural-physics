import numpy as np


class PCA:
    def __init__(self, n):
        """
        Create new PCA object
        @param n: number of components
        """
        self.n_components = n
        self.U = np.array([])
        self.mean = np.array([])

    def fit(self, x, mean_zero=True):
        """
        Calculate PCA from data x
        @param mean_zero: true if you want to do pca with a zero mean.
        @param x: m x n data array, where n is the number of data points and m DOF
        """
        self.mean = np.mean(x, axis=1, keepdims=True)
        if mean_zero:
            data = x - self.mean
        else:
            data = x

        # calculate eigen values and vectors of the covariance matrix
        cov = np.cov(data, rowvar=True)
        eigen_val, eigen_vec = np.linalg.eigh(cov)

        # sort descending
        sorted_index = np.argsort(eigen_val)[: -self.n_components - 1 : -1]
        sorted_eigen_vec = eigen_vec[:, sorted_index]
        self.U = sorted_eigen_vec.T

    def encode(self, x, mean_zero=True):
        """
        Compose data in to its PCA components
        @param x: m x n data vector, where n are the number of data points and m the number of features
        @param mean_zero: if the data should be standardised
        @return: return an n_components x n data vector
        """
        if mean_zero:
            return np.matmul(self.U, (x - self.mean))
        else:
            return np.matmul(self.U, x)

    def decode(self, x, mean_zero=True):
        """
        Decompose a PCA feature vector in to its full feature vector
        @param x: n_components x n array
        @return: m x n array with n the number of data points and m the number of features
        """
        if mean_zero:
            return np.matmul(self.U.T, x) + self.mean
        else:
            return np.matmul(self.U.T, x)
