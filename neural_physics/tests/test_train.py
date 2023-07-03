import torch
from neural_physics.core_math.pca import PCA
from neural_physics.train.subspace_neural_physics import SubSpaceNeuralNetwork, loss_fn
from neural_physics.utils.data_preprocess import (
    get_windows_,
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

    model = SubSpaceNeuralNetwork(num_components_X=n_components, n_hidden_layers=2)
    model = model.to(device)

    outputs = model(inputs)

    assert outputs.shape == (batch_size, n_components)


def test_train(dummy_data):
    X = dummy_data

    num_components = 8
    pca = PCA(num_components)
    pca.fit(X)
    subspace_z = pca.encode(X)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    subspace_z = torch.from_numpy(subspace_z).float().to(device)

    assert subspace_z.shape == (num_components, X.shape[1])

    # TODO: add external object
    # pca = PCA(n_components)
    # pca.fit(Y)
    # W = pca.encode(Y)

    alphas, betas = initial_model_params(subspace_z)

    network_correction = SubSpaceNeuralNetwork(
        n_hidden_layers=2, num_components_X=num_components, num_components_Y=0
    )
    network_correction = network_correction.to(device)

    optimizer = torch.optim.Adam(network_correction.parameters(), lr=0.01, amsgrad=True)

    for subspace_z_window in get_windows_(subspace_z, window_size=32):
        num_components, window_size = subspace_z_window.shape

        z_star = torch.zeros(
            (num_components, window_size), dtype=torch.float32, device=device
        )
        # TODO: add noise
        z_star[:, 0] = subspace_z_window[:, 0]
        z_star[:, 1] = subspace_z_window[:, 1]

        for frame in range(2, window_size):
            z_star_prev = z_star[:, frame - 1]
            z_star_prev_prev = z_star[:, frame - 2]

            z_bar = init_model_for_frame(alphas, betas, z_star_prev, z_star_prev_prev)

            model_inputs = torch.cat((z_bar, z_star_prev)).view(1, -1)
            model_inputs = model_inputs.to(device)
            residual_effects = network_correction(model_inputs)

            z_star[:, frame] = z_bar + residual_effects

        # z_star all non-zero
        assert torch.all(z_star)

        loss = loss_fn(
            z_star=z_star[:, 2:],
            z=subspace_z_window[:, 2:],
            z_star_prev=z_star[:, 1:-1],
            z_prev=subspace_z_window[:, 1:-1],
        )
        assert loss.shape == torch.Size([])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
