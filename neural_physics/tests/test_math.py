import numpy as np
from neural_physics.core_math.pca import PCA


def test_pca():
    dim = 5
    num_frames = 20
    inputs = np.random.rand(dim, num_frames)

    # no reduction, to make sure reconstruction is correct
    pca = PCA(dim)
    pca.fit(inputs)
    reduced = pca.encode(inputs)

    inputs_reconstructed = pca.decode(reduced)

    # check if the reconstruction is good
    assert np.allclose(inputs, inputs_reconstructed)
