import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from ex_3a import SwissRoll, EBM

# Define Langevin dynamics sampler
def langevin_dynamics(x, model, step_size=0.01, num_steps=50):
    x.requires_grad = True
    for i in range(num_steps):
        energy_grad = torch.autograd.grad(model(x).sum(), x)[0]
        
        x = x + 0.5 * step_size * energy_grad + step_size * np.sqrt(2*step_size)*torch.randn_like(x)
    return x

if __name__ == "__main__":

    n_gen = 1000  # number of generated samples
    steps = 1000
    sigma_r = 6

    model = EBM()

    loaded_state_dict = torch.load(f"saved_models/EBM_r{sigma_r}.pt", map_location=torch.device('cpu'))  # when you train with CUDA but evaluate on CPU
    model.load_state_dict(loaded_state_dict)

    model.eval()

    x_data = np.loadtxt('data/swiss_roll_2d.csv', delimiter=',')
    x_0 = sigma_r*torch.randn(n_gen, 2)  # Initialize N(0, 3**2 I)

    plt.figure(figsize=(5,5))
    plt.plot(x_0[:, 0], x_0[:, 1],  'r.')
    plt.title(f"Noise sample with standard deviation {sigma_r}")
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.grid()
    # plt.savefig(f'results/noise_r{sigma_r}.eps')
    plt.draw()

    for step_size in [0.1, 0.01, 0.001]:
        x_gen = langevin_dynamics(x_0, model, num_steps=steps, step_size=step_size).detach().numpy()
        plt.figure(figsize=(5,5))
        plt.plot(x_data[:, 0], x_data[:, 1], '.', label='Real')
        plt.plot(x_gen[:, 0], x_gen[:, 1], 'r.', label='Generated')
        plt.title(f'Noise standard deviation: {sigma_r}, step size: {step_size}')
        plt.xlim([-15, 15])
        plt.ylim([-15, 15])
        plt.legend(loc='upper left')
        plt.grid()
        # plt.savefig(f'results/gen_r{sigma_r}_eps{str(step_size).replace(".","")}.eps')
        plt.draw()

    plt.show()