import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

def dataloader(rootdir='./HW3_datasets'):
    # Read the labels.dat file
    data = pd.read_csv(os.path.join(rootdir, 'labels.dat'), delim_whitespace=True)

    # You can access the data in the dictionary by using keys of the form "(name of the folder)_(name of file)"
    # For example x[12345_01] contains the data of file 01.dat in the 12345 folder
    x_dict = {}

    # Same holds for the data dictionary
    # For example y[12345_01] contains the labels of each sample in file 01.dat under the folder 12345
    y_dict = {}

    for i in range(len(data)):
        map_id = data['map_id'][i]
        disc_id = str(data['disc_id'][i]).zfill(2)

        # Labels vector [k, g, s, p]
        labels = np.array([data['k'][i], data['g'][i], data['s'][i], data['p'][i]], dtype=float)

        # Storing the current .dat file to x
        x_dict[f'{map_id}_{disc_id}'] = pd.read_csv(f'{rootdir}/{map_id}/{disc_id}.dat', header=None).to_numpy()

        # Storing the labels of file x
        labels = np.tile(labels, (len(x_dict[f'{map_id}_{disc_id}']), 1))
        y_dict[f'{map_id}_{disc_id}'] = labels
    
    return x_dict, y_dict

class GalaxyDataset(Dataset):

    def __init__(self, included:list=['12345_01','12345_05', '12345_09']) -> None:
        super().__init__()

        included = list(set(included))  # remove duplicates

        x_dict, y_dict = dataloader()

        # IMP: If needed, change the type of x_data and y_data to float64

        self.x_data = torch.cat([torch.tensor(x_dict[key], dtype=torch.float32) for key in x_dict.keys() if key in included])

        # keep only last attribute of y (namely, p)
        self.p_data = torch.cat([torch.tensor(y_dict[key][:, -1], dtype=torch.float32) for key in y_dict.keys() if key in included])

    def __getitem__(self, index):
        return self.x_data[index], self.p_data[index]
    
    def __len__(self):
        return len(self.x_data)



if __name__ == "__main__":

    torch.manual_seed(42)

    x_dict, y_dict = dataloader(rootdir='./HW3_datasets')

    train_data = GalaxyDataset(included=['12345_01', '12345_05', '12345_09', '12345_13', '12345_17', '12345_21', '12345_25'])

    # example of batch fetched from a dataloader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    x_batch, p_batch = next(iter(train_loader))

    print("Data shape:", x_batch.shape)
    print("Labels (p) shape:", p_batch.shape)

    print(f"Example of values of p: {p_batch}")

    print("\nSaving the training dataset in train_data.pt")
    torch.save(train_data, "data/train_data.pt")
    print("Dataset saved successfully.")

    # You can load the training data using train_data = torch.load("train_data.pt")