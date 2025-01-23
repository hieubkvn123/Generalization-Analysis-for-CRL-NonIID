import os
import json
import pickle
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

# Some global constants 
DEFAULT_CLUSTER_STD = 0.1
DEFAULT_NUM_CLASSES = 20
DEFAULT_INPUT_DIM   = 128
DEFAULT_CLASS_PROBS = [1/DEFAULT_NUM_CLASSES]*DEFAULT_NUM_CLASSES 

def _init_gaussian_centers(savedir, C=DEFAULT_NUM_CLASSES, d=DEFAULT_INPUT_DIM):
    '''
    @savedir: Directory to save gaussian centers.
    @C      : The number of Gaussian centers (classes) to initialize.
    @d      : The dimensionality of the Gaussian centers.
    '''
    # Create directory in case it's not created yet
    pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
    savefile = os.path.join(savedir, f'C{C}_d{d}.pkl')

    # Check if savefile exists
    if os.path.exists(savefile):
        print(f'Loading saved gaussian centers from {savefile}')
        return pickle.load(open(savefile, 'rb'))

    centers = []
    for c in range(C):
        vec = np.random.randint(low=1, high=100, size=(d,))
        vec = vec / np.linalg.norm(vec, ord=2)
        centers.append(vec)
    centers = np.array(centers)

    # Save the pickle file
    with open(savefile, 'wb') as f:
        print(f'Saving gaussian centers to {savefile}')
        pickle.dump(centers, f)

    return centers 

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

def _onehot_encode_ints_array(arr):
    n_values = np.max(arr) + 1
    return np.eye(n_values)[arr]

def _generate_raw_gaussian_clusters(savedir, class_probs=None, C=DEFAULT_NUM_CLASSES, d=DEFAULT_INPUT_DIM, N=10e5, reinit=False):
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
    savedir_data = os.path.join(savedir, f'C{C}_d{d}_N{int(N)}')
    pathlib.Path(savedir_data).mkdir(parents=True, exist_ok=True)

    # Check if need to reinit
    if not reinit:
        if _data_exists(savedir_data):
            if _configuration_matches(savedir_data, configs):
                print('Data exists and configurations matches, reloading...')
                return _load_data(savedir_data)
    print('Data does not exist or configurations mismatches, re-creating data...')

    # Generate Gaussian centers
    savedir_center = os.path.join(savedir, 'gaussian_centers')
    centers = _init_gaussian_centers(savedir_center, C=C, d=d)
    sample_sizes = np.random.multinomial(N, class_probs, size=1)[0]

    # Generate samples
    X = np.array([])
    Y = np.array([])
    for i, (n_c, c) in enumerate(zip(sample_sizes, centers)):
        X_c = np.random.normal(loc=c, scale=DEFAULT_CLUSTER_STD, size=(n_c, d))
        if(i == 0):
            X, Y = X_c, np.full(n_c, i)
            continue
        X = np.concatenate([X, X_c])
        Y = np.concatenate([Y, np.full(n_c, i)])
    X = X.astype(np.float32)

    # Save raw data
    _save_data(X, Y, configs, savedir_data)
    return X, Y, configs

# Define the dataset and the dataloader
class GaussianDataset(Dataset):
    def __init__(self, X, Y): 
        self.X = X
        self.Y = Y
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Get Gaussian data
def generate_gaussian_clusters(N, savedir, test_ratio=0.8): # Test ratio w.r.t training dataset
    # Get N-train and N-test
    N_train = N 
    N_test  = int(N * test_ratio)

    # Generate raw data
    X, Y, _ = _generate_raw_gaussian_clusters(savedir, DEFAULT_CLASS_PROBS, N=N_train+N_test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=test_ratio/(1+test_ratio))

    train_dataset = GaussianDataset(X_train, Y_train)
    test_dataset = GaussianDataset(X_test, Y_test)
    return train_dataset, test_dataset 


if __name__ == '__main__':
    # Some constants
    savedir = 'data/gaussian'
    X, Y, _ = _generate_raw_gaussian_clusters(savedir, DEFAULT_CLASS_PROBS) 
