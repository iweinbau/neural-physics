import numpy as np

from neural_physics.core_math.pca import PCA


def test_pca_roundtrip_without_reduction():
    dim, num_frames = 5, 20
    inputs = np.random.default_rng(0).random((dim, num_frames))
    pca = PCA(dim)
    pca.fit(inputs)
    reduced = pca.encode(inputs)
    assert reduced.shape == (dim, num_frames)
    assert np.allclose(inputs, pca.decode(reduced))
    assert np.allclose(pca.explained_variance_ratio.sum(), 1.0)


def test_pca_svd_and_covariance_paths_agree():
    """Few frames -> SVD path; many frames -> covariance path. Both must span the same subspace."""
    rng = np.random.default_rng(1)
    m, k = 30, 4
    latent = rng.normal(size=(k, 500))
    mixing = rng.normal(size=(m, k))
    x = mixing @ latent + 0.01 * rng.normal(size=(m, 500)) + 3.0
    pca_svd = PCA(k).fit(x[:, :25])          # n < m  -> svd branch
    pca_cov = PCA(k).fit(x, max_frames=None)  # n > m  -> covariance branch
    assert pca_svd.U.shape == pca_cov.U.shape == (k, m)
    # orthonormal rows
    assert np.allclose(pca_cov.U @ pca_cov.U.T, np.eye(k), atol=1e-8)
    # both bases reconstruct the (low-rank) data well
    for pca in (pca_svd, pca_cov):
        err = pca.reconstruction_error(x)
        assert err.shape == (500,)
        assert err.mean() < 0.1
    assert np.all(np.diff(pca_cov.explained_variance) <= 1e-9)   # descending


def test_pca_subsampling_and_io(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(12, 400)) * np.linspace(1, 5, 12)[:, None]
    pca = PCA(3).fit(x, max_frames=100, seed=0)
    path = tmp_path / "pca.npz"
    pca.save(path)
    loaded = PCA.load(path)
    assert loaded.n_components == 3
    assert np.allclose(loaded.U, pca.U) and np.allclose(loaded.mean, pca.mean)
    assert np.allclose(loaded.encode(x), pca.encode(x))


def test_pca_rejects_too_many_components():
    x = np.random.default_rng(0).random((4, 10))
    try:
        PCA(5).fit(x)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
