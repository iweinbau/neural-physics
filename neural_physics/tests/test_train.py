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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    dummy_z_bar = torch.rand([batch_size, n_components], device=device)
    dummy_z_star_prev = torch.rand([batch_size, n_components], device=device)
    inputs = torch.cat((dummy_z_bar, dummy_z_star_prev), dim=1)

    model = SubSpaceNeuralNetwork(num_components=n_components, n_hidden_layers=2)
    model = model.to(device)

    outputs = model(inputs)

    assert outputs.shape == (batch_size, n_components)


def test_train(dummy_data):
    X = dummy_data

    num_components = 8
    pca = PCA(num_components)
    pca.fit(X)
    subspace_z = pca.encode(X)

    assert subspace_z.shape == (num_components, X.shape[1])

    # TODO: add external object
    # pca = PCA(n_components)
    # pca.fit(Y)
    # W = pca.encode(Y)

    alphas, betas = initial_model_params(subspace_z)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    network_correction = SubSpaceNeuralNetwork(
        n_hidden_layers=2, num_components=num_components
    )
    network_correction = network_correction.to(device)

    window_size = 10

    # Window size must be a divisor of the number of frames.
    if subspace_z.shape[1] % window_size != 0:
        # drop remainder of frames
        subspace_z = subspace_z[:, : -subspace_z.shape[1] % window_size]

    for subspace_z_window in np.array_split(
        subspace_z, subspace_z.shape[1] // window_size, axis=1
    ):
        assert subspace_z_window.shape == (num_components, window_size)

        z_star = np.zeros((num_components, window_size))
        z_star[:, 0] = subspace_z_window[:, 0]
        z_star[:, 1] = subspace_z_window[:, 1]

        for frame in range(2, window_size):
            z_star_prev = z_star[:, frame - 1]
            z_star_prev_prev = z_star[:, frame - 2]

            z_bar = init_model_for_frame(alphas, betas, z_star_prev, z_star_prev_prev)

            model_inputs = np.concatenate((z_bar, z_star_prev)).T
            model_inputs = torch.from_numpy(model_inputs).float().unsqueeze(0)
            model_inputs = model_inputs.to(device)
            residual_effects = network_correction(model_inputs).detach().numpy()

            z_star[:, frame] = z_bar + residual_effects

        # z_star all non-zero
        assert np.all(z_star)

    # TODO: compute loss
