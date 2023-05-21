import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from HW3_datasets.dataloader import GalaxyDataset

import os
from tqdm import tqdm

class CVAE(nn.Module):

    def __init__(self, input_shape:tuple=(32,), latent_dim:int=32, cond_dim:int=32):
        super(CVAE, self).__init__()

        self.input_shape = input_shape
        self.input_dim = np.prod(input_shape)
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.encoder = LinearEncoder(input_dim=self.input_dim, latent_dim=self.latent_dim, cond_dim=self.cond_dim)
        self.decoder = LinearDecoder(input_dim=self.input_dim, latent_dim=self.latent_dim, cond_dim=cond_dim)

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z, y).view(-1, *self.input_shape)

        return x_hat, mu, logvar, z
    
    def sample(self, y):

        num_samples = len(y)
        z = torch.randn(num_samples, self.latent_dim)
        return self.decoder(z, y).view(-1, *self.input_shape)
        
    @staticmethod
    def reparametrize(mu, logvar):
        std = torch.exp(.5*logvar)
        z = torch.randn_like(std)

        return std * z + mu
    

class LinearEncoder(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int=32, cond_dim:int=32):
        super(LinearEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.fc1 = nn.Linear(input_dim+cond_dim, 128)
        self.fc2 = nn.Linear(128, latent_dim * 2)

    def forward(self, x, y):        
        x_cond = torch.cat([x, y], dim=-1)
        x_out = torch.relu(self.fc1(x_cond))
        x_out = self.fc2(x_out)  # no activation for the last layer

        # half of the latent values are used for the mean, half for the log-variance
        mu = x_out[:, :self.latent_dim]
        logvar = x_out[:, self.latent_dim:] 
        return mu, logvar

class LinearDecoder(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int=32, cond_dim:int=32):
        super(LinearDecoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.rev_fc2 = nn.Linear(self.latent_dim+cond_dim, 128)
        self.rev_fc1 = nn.Linear(128, self.input_dim)

    def forward(self, z, y):
        z_cond = torch.cat([z, y], dim=-1)
        x = torch.relu(self.rev_fc2(z_cond))
        x = self.rev_fc1(x)
        return x
    
def loss_fn(x, x_hat, mu, logvar):
    # Reconstruction loss (cannot be BCE in this case, since x is unbounded)
    mse = F.mse_loss(x_hat, x, reduction='mean')

    # KLD between Gaussian rvs
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld


if __name__ == "__main__":


    # define hyper parameters
    batch_size = 128
    epochs = 40  # 20
    lr = 1e-3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    torch.manual_seed(42)

    input_shape = (300,)
    latent_dim = 32
    cond_dim = 32

    # load model

    model = CVAE(input_shape, latent_dim, cond_dim=cond_dim)
    model.to(device)

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = torch.load('data/train_data.pt')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    losses = []
    best_loss = 100    

    if not os.path.isdir('saved_models'):
        os.makedirs('saved_models')

    for epoch in range(epochs):

        train_loss = 0
        for x, p in tqdm(train_loader):
            optimizer.zero_grad()

            y = p[:, None] * torch.ones(len(p), cond_dim)  # many other ways to do this, e.g. torch.repeat_interleave
            x = x.to(device)
            y = y.to(device)
            x_hat, mu, logvar, z = model(x, y)

            loss = loss_fn(x, x_hat, mu, logvar)
            train_loss += loss.item()
            loss.backward()


            optimizer.step()
 
        train_loss /= len(train_loader)
        print(f"Average loss per batch at epoch {epoch+1:03d}: {train_loss:.4f}")
        losses.append(train_loss)
        
        # save model if it improved
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), f'saved_models/cvae_galaxy.pt')