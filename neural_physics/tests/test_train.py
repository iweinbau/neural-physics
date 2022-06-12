import torch
from neural_physics.train.subspace_neural_physics import SubSpaceNeuralNetwork


def test_model():
    batch_size = 32
    n_components = 256

    inputs = torch.rand(batch_size, n_components)
    model = SubSpaceNeuralNetwork(n_hidden_layers=2, n_components=n_components)
    outputs = model(inputs)

    assert outputs.shape == (batch_size, n_components)
