import os
import pickle
import pathlib
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

# Some global constants 
DEFAULT_CLUSTER_STD = 0.1
DEFAULT_NUM_CLASSES = 20
DEFAULT_INPUT_DIM   = 256
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

class GaussianTestDataset(Dataset):
    def __init__(self, rhos, mus, sigmas, N, k, seed=None):
        self.rhos = torch.as_tensor(rhos, dtype=torch.float32)
        self.mus = torch.as_tensor(mus, dtype=torch.float32)
        self.sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
        self.N = N
        self.k = k
        self.R = len(rhos)
        self.d = mus.shape[1]
        
        # Validate inputs
        assert self.rhos.shape == (self.R,), f"rhos shape mismatch: {self.rhos.shape}"
        assert self.mus.shape == (self.R, self.d), f"mus shape mismatch: {self.mus.shape}"
        assert self.sigmas.shape == (self.R,), f"sigmas shape mismatch: {self.sigmas.shape}"
        assert torch.allclose(self.rhos.sum(), torch.tensor(1.0), atol=1e-5), "rhos must sum to 1"
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Precompute negative sampling distributions for each component
        self.neg_distributions = self._compute_negative_distributions()
        
    def _compute_negative_distributions(self):
        neg_dists = []
        for r in range(self.R):
            # Get weights for all components except r
            weights = self.rhos.clone()
            weights[r] = 0
            # Normalize by (1 - rho_r)
            normalizer = 1.0 - self.rhos[r]
            if normalizer > 0:
                weights = weights / normalizer
            else:
                # Edge case: if rho_r = 1, set uniform weights
                weights = torch.ones(self.R) / (self.R - 1)
                weights[r] = 0
            neg_dists.append(weights)
        
        return torch.stack(neg_dists)
    
    def _sample_from_gaussian(self, mu, sigma, n_samples=1):
        return torch.randn(n_samples, self.d) * sigma + mu
    
    def _sample_negatives(self, r, k):
        # Sample which components to draw from (with replacement)
        weights = self.neg_distributions[r]
        component_indices = torch.multinomial(weights, k, replacement=True)
        
        # Sample from the selected components
        negatives = []
        for idx in component_indices:
            neg_sample = self._sample_from_gaussian(self.mus[idx], self.sigmas[idx], n_samples=1)
            negatives.append(neg_sample.squeeze(0))
        
        return torch.stack(negatives)
    
    def __len__(self):
        return self.R * self.N
    
    def __getitem__(self, idx):
        # Determine which component r this sample belongs to
        r = idx // self.N
        
        # Sample anchor and positive from N(mu_r, sigma_r^2)
        X_anc = self._sample_from_gaussian(self.mus[r], self.sigmas[r])
        X_pos = self._sample_from_gaussian(self.mus[r], self.sigmas[r])
        
        # Sample k negatives from \bar{D}_r
        X_negs = self._sample_negatives(r, self.k)
        
        # Return tuple as (X_anc, X_pos, list of negatives, rho_r)
        return (
            X_anc.squeeze(0),
            X_pos.squeeze(0),
            [X_negs[i] for i in range(self.k)],
            self.rhos[r]
        )

def _init_gaussian_centers(savedir, C=DEFAULT_NUM_CLASSES, d=DEFAULT_INPUT_DIM):
    # Create directory in case it's not created yet
    pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
    savefile = os.path.join(savedir, f'C{C}_d{d}.pkl')

    # Check if savefile exists
    if os.path.exists(savefile):
        print(f'Loading saved gaussian centers from {savefile}')
        return pickle.load(open(savefile, 'rb'))

    centers = []
    for c in range(C):
        vec = np.random.randint(low=1, high=1000, size=(d,))
        vec = vec / np.linalg.norm(vec, ord=2)
        centers.append(vec)
    centers = np.array(centers)

    # Save the pickle file
    with open(savefile, 'wb') as f:
        print(f'Saving gaussian centers to {savefile}')
        pickle.dump(centers, f)

    return centers 

def _generate_maximally_separated_vectors(savedir, C=DEFAULT_NUM_CLASSES, d=DEFAULT_INPUT_DIM, n_iterations=1000, lr=0.1):
    # Create directory in case it's not created yet
    pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
    savefile = os.path.join(savedir, f'C{C}_d{d}.pkl')

    # Initialize randomly on unit sphere
    vectors = torch.randn(C, d)
    vectors = vectors / vectors.norm(dim=1, keepdim=True)
    vectors.requires_grad = True

    optimizer = torch.optim.Adam([vectors], lr=lr)

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Normalize to unit sphere
        normalized = vectors / vectors.norm(dim=1, keepdim=True)

        # Compute pairwise distances
        dists = torch.cdist(normalized, normalized)

        # Mask out diagonal (self-distances)
        mask = ~torch.eye(C, dtype=bool)
        dists_valid = dists[mask]

        # Maximize minimum distance (or minimize negative of min distance)
        # We use soft minimum for differentiability
        min_dist = -torch.logsumexp(-dists_valid * 10, dim=0) / 10

        # Also add repulsion term to push all pairs apart
        repulsion = -1.0 / (dists_valid + 1e-6)
        loss = -min_dist + 0.01 * repulsion.mean()

        loss.backward()
        optimizer.step()

        if (iteration + 1) % 200 == 0:
            with torch.no_grad():
                normalized = vectors / vectors.norm(dim=1, keepdim=True)
                dists = torch.cdist(normalized, normalized)
                mask = ~torch.eye(C, dtype=bool)
                min_d = dists[mask].min().item()
                mean_d = dists[mask].mean().item()
                print(f"Iter {iteration+1}: min_dist={min_d:.4f}, mean_dist={mean_d:.4f}")

    # Final normalization
    with torch.no_grad():
        vectors = vectors / vectors.norm(dim=1, keepdim=True)
    centers = vectors.detach()

    # Save the pickle file
    with open(savefile, 'wb') as f:
        print(f'Saving gaussian centers to {savefile}')
        pickle.dump(centers, f)

    return vectors.detach()


def _save_data(X, Y, configs, savedir):
    with open(os.path.join(savedir, 'X.pkl'), 'wb') as f:
        pickle.dump(X, f)
    with open(os.path.join(savedir, 'Y.pkl'), 'wb') as f:
        pickle.dump(Y, f)
    with open(os.path.join(savedir, 'configs.pkl'), 'wb') as f:
        pickle.dump(configs, f) 
    print(f'Data saved to {savedir} successfully')

def _load_data(savedir):
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
        for k, v in configs.items():
            if isinstance(configs[k], float) or isinstance(configs[k], int):
                if configs[k] != v: 
                    return False
    return True

def _onehot_encode_ints_array(arr):
    n_values = np.max(arr) + 1
    return np.eye(n_values)[arr]

def _generate_raw_gaussian_clusters(savedir, class_probs=None, d=None, N=10e5, reinit=False):
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

    # Generate Gaussian distributions
    sigmas = [DEFAULT_CLUSTER_STD] * len(class_probs)
    mus = _generate_maximally_separated_vectors(savedir, C=len(class_probs), d=DEFAULT_INPUT_DIM)

    # Store in configs
    configs['mu'], configs['sigma'] = mus, sigmas

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
def generate_gaussian_clusters(N, savedir=None, class_probs=None): # Test ratio w.r.t training dataset
    # Generate class probabilities
    if class_probs is None:
        class_probs = DEFAULT_CLASS_PROBS
    if savedir is None:
        savedir = DEFAULT_SAVE_FOLDER

    # Generate raw data
    X, Y, configs = _generate_raw_gaussian_clusters(savedir, class_probs, N=N)
    train_dataset = GaussianDataset(X, Y)
    return train_dataset, configs 


if __name__ == '__main__':
    # Some constants
    savedir = 'data/gaussian'
    X, Y, _ = _generate_raw_gaussian_clusters(savedir, DEFAULT_CLASS_PROBS) 
