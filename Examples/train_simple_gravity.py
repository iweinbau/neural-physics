# read yml file stored in bin folder

from matplotlib import pyplot as plt
import yaml
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from neural_physics.train.subspace_neural_physics import *
from neural_physics.utils.data_preprocess import *

config_file_dir = "bin"
config_file_name = "config.yml"

learing_rate = 0.0001
num_epochs = 100
batch_size = 16
window_size = 32

with open(os.path.join(config_file_dir,config_file_name), 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

data_file_name = config['data_file']
external_file_name = config['external_force']

# read data file
data = np.load(os.path.join(config_file_dir, data_file_name))
# read external force file
external = np.load(os.path.join(config_file_dir, external_file_name))

# get time step
dt = config['dt']


X = data.T # 2nd column is the position data and data should be in format (dim x samples)
Y = external.T # 2nd column is the position data and data should be in format (dim x samples)

num_components_X = X.shape[0]
num_components_Y = Y.shape[0]

subspace_z = X # no PCA needed for 1D data
subspace_w = Y # no PCA needed for external effects

# pca = PCA(num_components)
# pca.fit(X)
# subspace_z = pca.encode(X)

# setup pytorch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

subspace_z = torch.from_numpy(subspace_z).float().to(device)
subspace_w = torch.from_numpy(subspace_w).float().to(device)

assert subspace_z.shape == (num_components_X, X.shape[1])
assert subspace_w.shape == (num_components_Y, Y.shape[1])
assert subspace_z.shape[1] == subspace_w.shape[1]

# calculate initial model parameters
alphas, betas = initial_model_params(subspace_z)

# create neural network with 
# 2 hidden layers and 
# num_components_X + num_components_Y input neurons

network_correction = SubSpaceNeuralNetwork(
        n_hidden_layers=2, 
        num_components_X=num_components_X, 
        num_components_Y=num_components_Y
    )
network_correction = network_correction.to(device)
optimizer = torch.optim.Adam(network_correction.parameters(), lr=learing_rate, amsgrad=True)

writer = SummaryWriter('runs/experiment_1')

iter = 0
# train model
for epoch in range(num_epochs):

    for subspace_z_window, subspace_w_window in get_windows(subspace_z, subspace_w, window_size=window_size):

        num_components_X, window_size = subspace_z_window.shape
        num_components_Y, _ = subspace_w_window.shape

        z_star = torch.zeros(
            (num_components_X, window_size), dtype=torch.float32, device=device
        )

        # create gaussian noise with std = 0.01 and size of z_star
        r0 = torch.randn(z_star.shape[0]) * 0.01
        r1 = torch.randn(z_star.shape[0]) * 0.01

        z_star[:, 0] = subspace_z_window[:, 0]
        z_star[:, 1] = subspace_z_window[:, 1]

        z_star[:, 0] += r0
        z_star[:, 1] += r1

        for frame in range(2, window_size):
            z_star_prev = z_star[:, frame - 1]
            z_star_prev_prev = z_star[:, frame - 2]
            w = subspace_w_window[:, frame]

            z_bar = init_model_for_frame(alphas, betas, z_star_prev, z_star_prev_prev)

            model_inputs = torch.cat((z_bar, z_star_prev, w)).view(1, -1)
            model_inputs = model_inputs.to(device)
            residual_effects = network_correction(model_inputs)

            z_star[:, frame] = z_bar + residual_effects[0]

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

        # ...log the running loss
        writer.add_scalar('training loss',
                        loss.item(),
                        iter)
        
        # add figure to tensorboard with z_star and z
        fig = plt.figure()
        plt.plot(z_star[1,:].cpu().detach().numpy(), label='z_star')
        plt.plot(subspace_z_window[1,:].cpu().detach().numpy(), label='z')
        plt.legend()
        writer.add_figure('z_star vs z', fig, global_step=iter)
                          
        iter += 1

    # printing loss
    print(f"Epoch {epoch}")
