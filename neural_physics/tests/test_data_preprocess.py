from neural_physics.core_math.pca import PCA
from neural_physics.utils.data_preprocess import initial_model_params


def test_train(dummy_data):
    X = dummy_data

    num_components = 8
    pca = PCA(num_components)
    pca.fit(X)
    subspace_z = pca.encode(X)

    assert subspace_z.shape == (num_components, X.shape[1])

    # pca = PCA(n_components)
    # pca.fit(Y)
    # W = pca.encode(Y)

    alphas, betas = initial_model_params(subspace_z)

    assert alphas.shape == (num_components,)
    assert betas.shape == (num_components,)
