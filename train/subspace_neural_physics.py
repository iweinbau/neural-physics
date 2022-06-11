import numpy as np

# 60 fps
dt = 1 / 60


def loss_position(z_star, z):
    return np.mean(np.abs(z_star - z))


def loss_velocity(z_star, z_star_prev, z, z_prev):
    return np.mean(np.abs((z_star - z_star_prev)/dt - (z - z_prev)/dt))


def loss_fn(z_star, z_star_prev, z, z_prev):
    return loss_position(z_star, z) + loss_velocity((z_star, z_star_prev, z, z_prev))

