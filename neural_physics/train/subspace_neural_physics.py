import numpy as np
import torch
from torch import nn

# 60 fps
dt = 1 / 60

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class SubSpaceNeuralNetwork(nn.Module):
    def __init__(self, n_hidden_layers: int, n_components: int):
        super().__init__()

        hidden_units = round(1.5 * n_components)

        self.encode = nn.Linear(n_components, hidden_units)

        self.feed_forward = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.ReLU())
                for _ in range(n_hidden_layers)
            ],
        )

        self.decode = nn.Linear(hidden_units, n_components)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input and then decodes it.

        @param inputs: The input to the network. Shape: (batch_size, n_components)
        @return: The output of the network. Shape: (batch_size, n_components)
        """
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
    return np.mean(np.abs(z_star - z), axis=0)


def loss_velocity(z_star, z_star_prev, z, z_prev):
    """
    Calculate loss originating from errors in the velocities
    @param z_star_prev: previous (s x n_components) predictions
    @param z_prev: previous (s x n_components) feature vector
    @param z_star: predicted (s x n_components) feature vector
    @param z: actual (s x n_components) feature vector
    @return: 1 x n_component loss vector
    """
    return np.mean(np.abs((z_star - z_star_prev) / dt - (z - z_prev) / dt), axis=0)


def loss_fn(z_star, z_star_prev, z, z_prev):
    """
    Total loss function
    @param z_star_prev: previous (s x n_components) predictions
    @param z_prev: previous (s x n_components) feature vector
    @param z_star: predicted (s x n_components) feature vector
    @param z: actual (s x n_components) feature vector
    @return: 1 x n_component loss vector
    """
    return loss_position(z_star, z) + loss_velocity(z_star, z_star_prev, z, z_prev)
