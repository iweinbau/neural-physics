import numpy as np
import torch
from neural_physics.core_math.pca import PCA
from neural_physics.train.subspace_neural_physics import SubSpaceNeuralNetwork
from neural_physics.utils.data_preprocess import (
    init_model_for_frame,
    initial_model_params,
)


def test_model():
    batch_size = 32
    n_components = 256

    inputs = torch.rand(batch_size, n_components)
    model = SubSpaceNeuralNetwork(n_hidden_layers=2, num_components=n_components)
    outputs = model(inputs)

    assert outputs.shape == (batch_size, n_components)


def test_train(dummy_data):
    X = dummy_data

    num_components = 8
    pca = PCA(num_components)
    pca.fit(X)
    subspace_z = pca.encode(X)
    # pca = PCA(n_components)
    # pca.fit(Y)
    # W = pca.encode(Y)

    alphas, betas = initial_model_params(subspace_z)

    network_correction = SubSpaceNeuralNetwork(
        n_hidden_layers=2, num_components=num_components
    )

    window_size = 10

    # Window size must be a divisor of the number of frames.
    if subspace_z.shape[1] % window_size != 0:
        # drop remainder of frames
        subspace_z = subspace_z[:, : -subspace_z.shape[1] % window_size]

    for subspace_z_window in np.array_split(
        subspace_z, subspace_z.shape[1] // window_size
    ):
        z_star = np.zeros((num_components, window_size))
        z_star[:, 0] = subspace_z_window[:, 0]
        z_star[:, 1] = subspace_z_window[:, 1]

        for frame in range(2, window_size):
            z_star_prev = z_star[:, frame - 1]
            z_star_prev_prev = z_star[:, frame - 2]
            z_bar = init_model_for_frame(alphas, betas, z_star_prev, z_star_prev_prev)
            model_inputs = np.concatenate((z_bar, z_star_prev)).T
            model_inputs = torch.from_numpy(model_inputs).float().unsqueeze(0)
            z_star[:, frame] = z_bar + network_correction(model_inputs)  # , w_i)
