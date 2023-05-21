from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.distributions import MultivariateNormal

import matplotlib.pyplot as plt

class SwissRoll(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.data.__len__()
    
# Architecture stolen from http://www.stat.ucla.edu/~ruiqigao/papers/flow_ebm.pdf
class EBM(nn.Module):
    def __init__(self):
        super(EBM, self).__init__()
        self.log_Z = nn.Parameter(torch.tensor([1.], requires_grad=True))  # log of normalization constant

        layers = [
            nn.Linear(2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    # seed + device
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # hyperparameters
    epochs = 40
    batch_size = 128
    lr = 1e-3
    num_steps = 50      # number of Langevin steps during training
    step_size = 0.01    # size of each Langevin steps

    sigma_r = 6         # standard deviation of the initial noise

    # load data
    x_data = np.loadtxt('data/swiss_roll_2d.csv', delimiter=',')
    train_data = SwissRoll(x_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # init model
    model  = EBM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    noise_dist = MultivariateNormal(torch.zeros(2), (sigma_r**2)*torch.eye(2))

    loss_list = []
    for epoch in range(epochs):
        losses = []
        for x in tqdm(train_loader):

            opt.zero_grad()

            x = x.to(device)

            r = sigma_r*torch.randn(len(x), 2)

            logp_x = model(x)  # logp(x)
            logq_x = noise_dist.log_prob(x).unsqueeze(1)  # logq(x)
            logp_y = model(r)  # logp(y)
            logq_y = noise_dist.log_prob(r).unsqueeze(1)  # logq(y)

            v_x = logp_x - torch.logsumexp(torch.cat([logp_x, model.log_Z + logq_x], dim=1), dim=1, keepdim=True)  # logp(x)/(logp(x) + logq(x))
            v_x_hat = logq_y - torch.logsumexp(torch.cat([logp_y, model.log_Z + logq_y], dim=1), dim=1, keepdim=True)  # logq(x_hat)/(logp(x_hat) + logq(x_hat))

            loss = -torch.mean(v_x + v_x_hat)

            loss.backward()
            losses.append(loss.item())

            opt.step()
    
        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1:03d}:{epochs:03d}, Loss: {avg_loss:.4f}")
        loss_list.append(avg_loss)

    import os
    if not os.path.isdir("saved_models"):
        os.makedirs("saved_models")
    if not os.path.isdir("results"):
        os.makedirs("results")

    torch.save(model.state_dict(), f"saved_models/EBM_r{sigma_r}.pt")
    np.savetxt(f"results/ex3_loss_r{sigma_r}.csv", np.asarray(loss_list))