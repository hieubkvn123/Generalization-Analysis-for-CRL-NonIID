import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class GaussianMixtureDataset(Dataset):
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


# Example usage
if __name__ == "__main__":
    # Define parameters
    R = 3  # number of components
    d = 2  # dimensionality
    N = 100  # samples per component
    k = 5  # number of negatives per tuple
    
    # Create mixture parameters
    rhos = torch.tensor([0.5, 0.3, 0.2])
    mus = torch.tensor([[0.0, 0.0], [3.0, 3.0], [-2.0, 2.0]])
    sigmas = torch.tensor([1.0, 0.5, 0.8])
    
    # Create dataset and dataloader
    dataset = GaussianMixtureDataset(rhos, mus, sigmas, N, k, seed=42)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Sample a batch
    batch = next(iter(dataloader))
    print(f"Batch structure: tuple of length {len(batch)}")
    print(f"Anchors shape: {batch[0].shape}")
    print(f"Positives shape: {batch[1].shape}")
    print(f"Negatives (list length): {len(batch[2])}")
    print(f"Negatives[0] shape: {batch[2][0].shape}")
    print(f"Rhos shape: {batch[3].shape}")
    print(f"\nExample rho values in batch: {batch[3][:5]}")
