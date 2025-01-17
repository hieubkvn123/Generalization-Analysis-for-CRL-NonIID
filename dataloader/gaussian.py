import os
import json
import pickle
import numpy as np

import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

# Some global constants 
CLUSTER_STD = 0.5
CLASS_PROBS = [0.15, 0.12, 0.10, 0.08, 0.05, 0.20, 0.10, 0.07, 0.08, 0.05] 

def _init_gaussian_centers(C=10, d=64):
    '''
    @C: The number of Gaussian centers (classes) to initialize.
    @d: The dimensionality of the Gaussian centers.
    '''
    centers = []
    for c in range(C):
        vec = np.random.randint(low=1, high=100, size=(d,))
        vec = vec / np.linalg.norm(vec, ord=2)
        centers.append(vec)
    return np.array(centers)

def _save_data(X, Y, configs, savedir):
    '''
    @X      : The input vectors.
    @Y      : The corresponding labels.
    @savedir: Save directory.
    '''
    with open(os.path.join(savedir, 'X.pkl'), 'wb') as f:
        pickle.dump(X, f)
    with open(os.path.join(savedir, 'Y.pkl'), 'wb') as f:
        pickle.dump(Y, f)
    with open(os.path.join(savedir, 'configs.json'), 'w') as f:
        json.dump(configs, f) 
    print(f'Data saved to {savedir} successfully')

def _load_data(savedir):
    '''
    @savedir: The save directory with X.pkl and Y.pkl files.
    '''
    X = pickle.load(open(os.path.join(savedir, 'X.pkl'), 'rb'))
    Y = pickle.load(open(os.path.join(savedir, 'Y.pkl'), 'rb'))
    configs = json.load(open(os.path.join(savedir, 'configs.json'), 'r'))
    return X, Y, configs
    
def _data_exists(savedir):
    Xfile = os.path.join(savedir, 'X.pkl')
    Yfile = os.path.join(savedir, 'Y.pkl')
    config_file = os.path.join(savedir, 'configs.json')
    return os.path.exists(Xfile) and os.path.exists(Yfile) and os.path.exists(config_file)

def _configuration_matches(savedir, configs):
    with open(os.path.join(os.path.join(savedir, 'configs.json')), 'r') as f:
        src_configs = json.load(f) 
        for k, v in src_configs.items():
            if configs[k] != v:
                return False
    return True

def _generate_raw_gaussian_clusters(savedir, class_probs=None, C=10, d=64, N=10e5, reinit=False):
    '''
    @centers    : Gaussian centers.
    @class_probs: A list of class probabilities (sums up to 1, length equals number of centers).
    @C          : The number of Gaussian centers (classes) to initialize.
    @d          : The dimensionality of the Gaussian centers.
    @N          : Total number of data points to generate.
    '''
    # Generate class distribution if not provided
    if class_probs is not None:
        assert C == len(class_probs)
    else:
        ints = np.random.randint(low=1, high=100, size=(C,))
        class_probs = ints / np.linalg.norm(ints, ord=1) 
        class_probs = class_probs.tolist()

    # Assemble configurations
    configs = {'N' : N, 'C' : C, 'd' : d, 'probs' : class_probs}

    # Check if need to reinit
    if not reinit:
        if _data_exists(savedir):
            if _configuration_matches(savedir, configs):
                print('Data exists and configurations matches, reloading...')
                return _load_data(savedir)
    print('Data does not exist or configurations mismatches, re-creating data...')

    # Generate Gaussian centers
    centers = _init_gaussian_centers(C=C, d=d)
    sample_sizes = np.random.multinomial(N, class_probs, size=1)[0]

    # Generate samples
    X = np.array([])
    Y = np.array([])
    for i, (n_c, c) in enumerate(zip(sample_sizes, centers)):
        X_c = np.random.normal(loc=c, scale=CLUSTER_STD, size=(n_c, d))
        if(i == 0):
            X, Y = X_c, np.full(n_c, i)
            continue
        X = np.concatenate([X, X_c])
        Y = np.concatenate([Y, np.full(n_c, i)])

    # Save raw data
    _save_data(X, Y, configs, savedir)
    return X, Y, configs

# Define the dataset and the dataloader
class GaussianDataset(Dataset):
    def __init__(self, X, Y): 
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Get Gaussian data
def generate_gaussian_clusters(savedir):
    # Generate raw data
    X, Y, _ = _generate_raw_gaussian_clusters(savedir, CLASS_PROBS)
    dataset = GaussianDataset(X, Y)
    
    # Split
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


if __name__ == '__main__':
    # Some constants
    savedir = 'data/gaussian'
    X, Y, _ = _generate_raw_gaussian_clusters(savedir, CLASS_PROBS) 
