import os
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
DEFAULT_SAVE_FOLDER = 'gaussian'
DEFAULT_CLASS_PROBS = [1/DEFAULT_NUM_CLASSES]*DEFAULT_NUM_CLASSES 

# Define the dataset and the dataloader
class GaussianDataset(Dataset):
    def __init__(self, X, Y): 
        self.X = X
        self.Y = Y
        self.transform = transforms.ToTensor()

        # Calculate statistics
        self.all_labels = sorted(np.unique(self.Y).tolist())
        self.Ns, self.rho_hat = self._calculate_rho_hat(self.Y)

    def _calculate_rho_hat(self, nums):
        nums_sorted = sorted(nums)
        total = len(nums_sorted)
        freq = {}
        for n in nums_sorted:
            freq[n] = freq.get(n, 0) + 1
        rho_hat = {k: v / total for k, v in freq.items()}

        return freq, rho_hat 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

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
    with open(os.path.join(savedir, 'configs.pkl'), 'wb') as f:
        pickle.dump(configs, f) 
    print(f'Data saved to {savedir} successfully')

def _load_data(savedir):
    '''
    @savedir: The save directory with X.pkl and Y.pkl files.
    '''
    X = pickle.load(open(os.path.join(savedir, 'X.pkl'), 'rb'))
    Y = pickle.load(open(os.path.join(savedir, 'Y.pkl'), 'rb'))
    configs = pickle.load(open(os.path.join(savedir, 'configs.pkl'), 'rb'))
    return X, Y, configs
    
def _data_exists(savedir):
    Xfile = os.path.join(savedir, 'X.pkl')
    Yfile = os.path.join(savedir, 'Y.pkl')
    config_file = os.path.join(savedir, 'configs.pkl')
    return os.path.exists(Xfile) and os.path.exists(Yfile) and os.path.exists(config_file)

def _configuration_matches(savedir, configs):
    with open(os.path.join(os.path.join(savedir, 'configs.pkl')), 'rb') as f:
        src_configs = pickle.load(f) 
        for k, v in src_configs.items():
            if configs[k] != v:
                return False
    return True

def _onehot_encode_ints_array(arr):
    n_values = np.max(arr) + 1
    return np.eye(n_values)[arr]

def _generate_raw_gaussian_clusters(savedir, mus, sigmas, class_probs=None, d=None, N=10e5, reinit=False):
    '''
    @centers    : Gaussian centers.
    @class_probs: A list of class probabilities (sums up to 1, length equals number of centers).
    @C          : The number of Gaussian centers (classes) to initialize.
    @d          : The dimensionality of the Gaussian centers.
    @N          : Total number of data points to generate.
    '''
    # Generate class distribution if not provided
    if class_probs is not None:
        C = len(class_probs)
    else:
        ints = np.random.randint(low=1, high=100, size=(C,))
        class_probs = ints / np.linalg.norm(ints, ord=1) 
        class_probs = class_probs.tolist()

    # Check data dimensionality
    if d is None:
        d = DEFAULT_INPUT_DIM

    # Assemble configurations
    configs = {'N' : N, 'C' : C, 'd' : d, 
                'probs' : list(class_probs),
                'mu': mus, 'sigma': sigmas}
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
    sample_sizes = np.random.multinomial(N, class_probs, size=1)[0]

    # Generate samples
    X = np.array([])
    Y = np.array([])
    for i, (n_c, mu, sigma) in enumerate(zip(sample_sizes, mus, sigmas)):
        X_c = np.random.normal(loc=mu, scale=sigma, size=(n_c, d))
        if(i == 0):
            X, Y = X_c, np.full(n_c, i)
            continue
        X = np.concatenate([X, X_c])
        Y = np.concatenate([Y, np.full(n_c, i)])
    X = X.astype(np.float32)

    # Save raw data
    _save_data(X, Y, configs, savedir_data)
    return X, Y, configs

# Get Gaussian data
def generate_gaussian_clusters(N, savedir=None, test_ratio=0.8, class_probs=None): # Test ratio w.r.t training dataset
    # Get N-train and N-test
    N_train = N 
    N_test  = int(N * test_ratio)

    # Generate class probabilities
    if class_probs is None:
        class_probs = DEFAULT_CLASS_PROBS
    if savedir is None:
        savedir = DEFAULT_SAVE_FOLDER

    # Generate Gaussian distributions
    sigmas = [DEFAULT_CLUSTER_STD] * len(class_probs)
    mus = _init_gaussian_centers(savedir, C=len(class_probs), d=DEFAULT_INPUT_DIM)

    # Generate raw data
    X, Y, configs = _generate_raw_gaussian_clusters(savedir, mus, sigmas, class_probs, N=N_train+N_test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=test_ratio/(1+test_ratio))

    train_dataset = GaussianDataset(X_train, Y_train)
    test_dataset = GaussianDataset(X_test, Y_test)
    return train_dataset, test_dataset, configs 


if __name__ == '__main__':
    # Some constants
    savedir = 'data/gaussian'
    X, Y, _ = _generate_raw_gaussian_clusters(savedir, DEFAULT_CLASS_PROBS) 
