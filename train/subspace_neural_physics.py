import numpy as np
import torch
from torch import nn

# 60 fps
dt = 1 / 60

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class SubSpaceNeuralNetwork(nn.Module):
    def __int__(self, n_hidden_layers, n_components):
        def input_layer(n_input, n_output):
            return nn.Linear(n_input, n_output)

        def hidden_layer(n_hidden):
            return nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU())

        def output_layer(n_input, n_output):
            return nn.Linear(n_input, n_output)

        self.n_components = n_components
        self.hidden_units = round(1.5 * n_components)
        layers = [input_layer(self.n_components, self.hidden_units)]
        for i in range(n_hidden_layers):
            layers.append(hidden_layer(self.hidden_units))
        layers.append(output_layer(self.hidden_units, self.n_components))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


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
    return np.mean(np.abs((z_star - z_star_prev)/dt - (z - z_prev)/dt), axis=0)


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
