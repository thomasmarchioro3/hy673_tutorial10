##############################################
# Reduce together real and generated samples
##############################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from HW3_datasets.dataloader import GalaxyDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from ex_2a_cvae_train import CVAE

if __name__ == "__main__":

    torch.manual_seed(42)

    input_shape = (300,)
    latent_dim = 32
    cond_dim = 32
    num_samples = 100

    method = "tsne"

    if method == "pca":
        from sklearn.decomposition import PCA as RED
    elif method == "tsne":
        from sklearn.manifold import TSNE as RED
    elif method == "umap":
        from umap import UMAP as RED
    else:
        print(f"Invalid reduction method {method}.")
        exit(-1)

    torch.manual_seed(42)


    # load model
    model = CVAE(input_shape, latent_dim, cond_dim=cond_dim)
    loaded_state_dict = torch.load(f"saved_models/cvae_galaxy.pt", map_location=torch.device('cpu'))
    model.load_state_dict(loaded_state_dict)
    model.eval()


    train_dataset = torch.load('data/train_data.pt')
    train_loader = DataLoader(train_dataset, batch_size=14000, shuffle=False)

    x_real, p_real = next(iter(train_loader))

    _, x_real, _, p_real = train_test_split(x_real, p_real, test_size=1000, stratify=p_real, random_state=42)

    x_real = x_real[:, 1:] - x_real[:, :-1]

    p_gen = np.zeros(len(p_real.unique())*100)
    x_gen = np.zeros((len(p_real.unique())*100, 300))

    for i, p in enumerate(p_real.unique()):
        y = p * torch.ones(num_samples, cond_dim)
        x_hat = model.sample(y).detach().numpy()
        x_gen[i*100:(i+1)*100] = x_hat
        p_gen[i*100:(i+1)*100] = p 

    x_gen = x_gen[:, 1:] - x_gen[:, :-1]

    x = np.concatenate([x_real, x_gen], axis=0)

    x = StandardScaler().fit_transform(x)

    s = RED(n_components=2, random_state=42).fit_transform(X=x)

    s_real = s[:len(x_real)]
    s_gen = s[len(x_real):]

    plt.figure()

    for i, p in enumerate(np.unique(p_gen)):

        s_r = s_real[p_real == p]
        # plt.plot(s_, np.zeros_like(s_), '.', label=f'p={p:.2f}')
        color_real = np.asarray([1.-p.item()/7, 0.1, 0.1])
        plt.plot(s_r[:, 0], s_r[:, 1], '.', label=f'p={p:.2f}', color=color_real) # 2d components
        
        s_g = s_gen[p_gen == p]
        color_gen = np.asarray([0.1, 1.-p.item()/7, 0.1])
        plt.plot(s_g[:, 0], s_g[:, 1], '.', label=f'p={p:.2f}', color=color_gen) # 2d components

    plt.draw()
    plt.title(f'Method: {method.upper()}')
    plt.savefig(f'results/red_both_{method}.pdf')
    plt.show()