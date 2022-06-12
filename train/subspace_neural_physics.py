import numpy as np

# 60 fps
dt = 1 / 60


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
    Total loss fucntion
    @param z_star_prev: previous (s x n_components) predictions
    @param z_prev: previous (s x n_components) feature vector
    @param z_star: predicted (s x n_components) feature vector
    @param z: actual (s x n_components) feature vector
    @return: 1 x n_component loss vector
    """
    return loss_position(z_star, z) + loss_velocity(z_star, z_star_prev, z, z_prev)
