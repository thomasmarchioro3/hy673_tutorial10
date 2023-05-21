import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ex_1a_cvae_train import CVAE


def plot_samples(samples, samples_per_row=8):
    n_rows = len(samples) // samples_per_row
    n_tot = int(n_rows*samples_per_row)
    samples = samples[:n_tot]
    fig = plt.figure(figsize=(2*samples_per_row, 2*n_rows))
    for i, out in enumerate(samples):
        a = fig.add_subplot(n_rows, samples_per_row, i+1)
        plt.imshow(out, cmap='binary')
        a.axis("off")

    return fig

def get_n_params(model):
    return np.sum([np.prod(param.shape) for param in model.parameters()])

if __name__ == "__main__":

    input_shape = (1, 28, 28)
    latent_dim = 64
    num_classes = 10
    
    torch.manual_seed(42)

    # load model
    model = CVAE(input_shape, latent_dim, num_classes=num_classes)
    loaded_state_dict = torch.load(f"saved_models/cvae_MNIST.pt", map_location=torch.device('cpu'))
    model.load_state_dict(loaded_state_dict)
    model.eval()

    print(f"Total number of parameters: {get_n_params(model):d}")

    label = torch.arange(10)

    y = F.one_hot(label, num_classes)

    x_hat = model.sample(y).detach().numpy().reshape(-1, 28, 28)

    fig = plot_samples(x_hat, samples_per_row=5)

    plt.savefig("results/gen_MNIST.eps")


