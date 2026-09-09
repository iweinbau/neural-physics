"""Linear subspace (PCA) used to compress the object state (paper Sec. 5.1).

    Z = U (X - x_mu),   U in R^{u x 3c}
    X ~= U^T Z + x_mu

Data layout follows the paper: columns are frames, rows are degrees of freedom, i.e. `x` is
(m x n) with m = 3c and n = number of frames.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class PCA:
    def __init__(self, n):
        """
        Create new PCA object
        @param n: number of components
        """
        self.n_components = int(n)
        self.U = np.array([])                    # (n_components x m)
        self.mean = np.array([])                 # (m x 1)
        self.explained_variance = np.array([])   # (n_components,)
        self.total_variance = 0.0

    def fit(self, x, mean_zero=True, max_frames: Optional[int] = 10000, seed: int = 0):
        """
        Calculate PCA from data x
        @param x: m x n data array, where n is the number of data points (frames) and m DOF
        @param mean_zero: true if you want to do pca with a zero mean (subtract the mean first; the paper does)
        @param max_frames: at most this many (randomly chosen) frames are used to build the basis. The paper
                           subsamples too ("If the memory usage is too large to perform PCA we subsample the data").
                           None = use every frame.
        """
        x = np.asarray(x, dtype=np.float64)
        m, n = x.shape
        if self.n_components > min(m, n):
            raise ValueError(f"n_components={self.n_components} exceeds min(DOF, frames)={min(m, n)}")
        self.mean = np.mean(x, axis=1, keepdims=True) if mean_zero else np.zeros((m, 1))
        if max_frames is not None and n > max_frames:
            cols = np.random.default_rng(seed).choice(n, size=max_frames, replace=False)
            data = x[:, np.sort(cols)] - self.mean
        else:
            data = x - self.mean
        n_used = data.shape[1]

        if n_used <= m:
            # thin SVD of the (m x n) data matrix: cheaper than the m x m covariance when frames < DOF
            u_full, s, _ = np.linalg.svd(data, full_matrices=False)
            var = s ** 2 / max(n_used - 1, 1)
            self.U = u_full[:, : self.n_components].T
            self.explained_variance = var[: self.n_components]
            self.total_variance = float(var.sum())
        else:
            # eigen-decomposition of the covariance matrix (m x m) when there are many frames
            cov = (data @ data.T) / max(n_used - 1, 1)
            eigen_val, eigen_vec = np.linalg.eigh(cov)
            order = np.argsort(eigen_val)[::-1]
            self.U = eigen_vec[:, order[: self.n_components]].T
            self.explained_variance = np.clip(eigen_val[order[: self.n_components]], 0, None)
            self.total_variance = float(np.clip(eigen_val, 0, None).sum())
        # deterministic sign convention: largest-magnitude entry of every basis vector is positive
        sign = np.sign(self.U[np.arange(self.n_components), np.abs(self.U).argmax(axis=1)])
        sign[sign == 0] = 1.0
        self.U = self.U * sign[:, None]
        return self

    def encode(self, x, mean_zero=True):
        """
        Compose data in to its PCA components
        @param x: m x n data vector, where n are the number of data points and m the number of features
        @param mean_zero: subtract the mean (must match `fit`)
        @return: n_components x n array
        """
        if mean_zero:
            return np.matmul(self.U, (x - self.mean))
        return np.matmul(self.U, x)

    def decode(self, z, mean_zero=True):
        """
        Decompose a PCA feature vector in to its full feature vector
        @param z: n_components x n array
        @return: m x n array with n the number of data points and m the number of features
        """
        if mean_zero:
            return np.matmul(self.U.T, z) + self.mean
        return np.matmul(self.U.T, z)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.explained_variance / self.total_variance if self.total_variance > 0 else self.explained_variance

    def reconstruction_error(self, x) -> np.ndarray:
        """Per-frame RMS error over all vertices of decode(encode(x)) vs x, in the units of x.
        @param x: m x n array with m = 3c
        @return: (n,) array
        """
        r = self.decode(self.encode(x)) - x
        return np.sqrt((r.reshape(-1, 3, r.shape[1]) ** 2).sum(1).mean(0))

    def save(self, path):
        np.savez(path, U=self.U, mean=self.mean, explained_variance=self.explained_variance,
                 total_variance=self.total_variance, n_components=self.n_components)

    @classmethod
    def load(cls, path) -> "PCA":
        with np.load(path) as f:
            pca = cls(int(f["n_components"]))
            pca.U, pca.mean = f["U"], f["mean"]
            pca.explained_variance, pca.total_variance = f["explained_variance"], float(f["total_variance"])
        return pca
