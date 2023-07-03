import numpy as np
import torch
from torch import nn

# 60 fps
delta_t = 1 / 60

class SubSpaceNeuralNetwork(nn.Module):
    def __init__(self, num_components_X: int = 256, num_components_Y: int=4, n_hidden_layers: int = 10):
        super().__init__()

        self.input_size = num_components_X * 2 + num_components_Y  # (z_bar, z_star_prev) concatenated

        hidden_size = round(1.5 * num_components_X)

        self.encode = nn.Sequential(nn.Linear(self.input_size, hidden_size), nn.ReLU())

        self.feed_forward = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
                for _ in range(n_hidden_layers)
            ],
        )

        self.decode = nn.Linear(hidden_size, num_components_X)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input and then decodes it.

        @param inputs: The input to the network. Shape: (batch_size, n_components)
        @return: The output of the network. Shape: (batch_size, n_components)
        """
        assert (
            inputs.shape[1] == self.input_size
        ), f"input_size: {inputs.shape[1]} does not match expected input_size: {self.input_size}"

        hidden = self.encode(inputs)  # (batch_size, hidden_units)
        hidden = self.feed_forward(hidden)  # (batch_size, hidden_units)
        logits = self.decode(hidden)  # (batch_size, n_components)
        return logits

def loss_position(z_star, z):
    """
    Calculate loss originating from errors in the positions
    @param z_star: predicted (s x n_components) feature vector
    @param z: actual (s x n_components) feature vector
    @return: 1 x n_component loss vector
    """
    return torch.abs(z_star - z).mean()

def loss_velocity(z_star, z_star_prev, z, z_prev):
    """
    Calculate loss originating from errors in the velocities
    @param z_star_prev: previous (s x n_components) predictions
    @param z_prev: previous (s x n_components) feature vector
    @param z_star: predicted (s x n_components) feature vector
    @param z: actual (s x n_components) feature vector
    @return: 1 x n_component loss vector
    """
    return torch.abs((z_star - z_star_prev) / delta_t - (z - z_prev) / delta_t).mean()

def loss_fn(z_star, z_star_prev, z, z_prev):
    """
    Total loss function
    @param z_star_prev: previous (s x n_components) predictions
    @param z_prev: previous (s x n_components) feature vector
    @param z_star: predicted (s x n_components) feature vector
    @param z: actual (s x n_components) feature vector
    @return: 1 x n_component loss vector
    """
    loss = loss_position(z_star, z) + loss_velocity(z_star, z_star_prev, z, z_prev)
    return loss
