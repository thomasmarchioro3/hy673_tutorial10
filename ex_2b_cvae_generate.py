import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ex_2a_cvae_train import CVAE


def plot_samples(samples, p, samples_per_row=8):
    n_rows = len(samples) // samples_per_row
    n_tot = int(n_rows*samples_per_row)
    samples = samples[:n_tot]
    fig = plt.figure(figsize=(2.5*samples_per_row, 2*n_rows))
    for i, out in enumerate(samples):
        a = fig.add_subplot(n_rows, samples_per_row, i+1)
        plt.plot(out, '-')
        # a.axis("off")
    fig.suptitle(f'p={p:.2f}')
    fig.tight_layout()
    return fig

def get_n_params(model):
    return np.sum([np.prod(param.shape) for param in model.parameters()])

if __name__ == "__main__":

    input_shape = (300,)
    latent_dim = 32
    cond_dim = 32
    num_samples = 10

    torch.manual_seed(42)

    # load model
    model = CVAE(input_shape, latent_dim, cond_dim=cond_dim)
    loaded_state_dict = torch.load(f"saved_models/cvae_galaxy.pt", map_location=torch.device('cpu'))
    model.load_state_dict(loaded_state_dict)
    model.eval()

    print(f"Total number of parameters: {get_n_params(model):d}")

    for p in np.arange(0.01, 5.02, 1):

        y = p * torch.ones(num_samples, cond_dim)

        x_hat = model.sample(y).detach().numpy()

        fig = plot_samples(x_hat, p, samples_per_row=5)

        plt.savefig(f"results/gen_galaxy_{int(p)}.eps")
    plt.show()


