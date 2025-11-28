import math
import torch
import pprint
import itertools
import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset 

# Utility functions
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def save_json_dict(json_dict, output):
    with open(output, 'w') as f:
        pprint_json_string = pprint.pformat(json_dict, compact=True).replace("'", '"')
        f.write(pprint_json_string)
        f.close()

def apply_model_to_batch(model, batch, device=None):
    # Unpack the batch
    x1, x2, x3 = batch[0].to(device), batch[1].to(device), [x.to(device) for x in batch[2]]

    # Apply model to batch
    y1, y2, y3 = model(x1), model(x2), [model(x) for x in x3]
    return y1, y2, y3

class UnsupervisedDatasetWrapper(object):
    def __init__(self, dataset, M, k, regime='subsample'):
        # Just to make sure
        super().__init__()
        assert regime in ['subsample', 'weighted_subsample'], f'Invalid regime {regime}'

        # Store configurations
        self.dataset = dataset
        self.k = k
        self.M = M
        self.N = len(self.dataset)
        self.Ns = self.dataset.Ns
        self.rho_hat = self.dataset.rho_hat
        self.all_labels = self.dataset.all_labels
        self.tau_hat = self._calculate_tau_hat()
        self.default_weight = 1/(1 - self.tau_hat)
        self.collided_weights = self._calculate_collided_weights()
        
        # Map labels to instance indices
        self.label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset):
            self.label_to_indices[label].append(idx)
        self.label_to_indices = {label: np.array(indices) for label, indices in self.label_to_indices.items()}

        # Get contrastive tuples
        if regime == 'subsample':
            self.all_tuples = self._subsample(self.M)
        else:
            self.all_tuples = self._weighted_subsample(self.M)

    def _calculate_tau_hat(self):
        non_col_prob = 0.0
        for _class, rho_r in self.rho_hat.items():
            non_col_prob += rho_r * ((1 - rho_r)**self.k)
        return 1 - non_col_prob

    def _calculate_omegas(self):
        omegas = {}
        for _class, N_r in self.Ns.items():
            omegas[_class] = math.comb(N_r, 2) * math.comb(self.N - 2, self.k)
        return omegas
    
    def _calculate_lambdas(self):
        lambdas = {}
        for _class, N_r in self.Ns.items():
            lambdas[_class] = math.comb(N_r, 3) * math.comb(self.N - 3, self.k - 1)
        return lambdas

    def _calculate_collided_weights(self):
        weights = {}
        omegas = self._calculate_omegas()
        lambdas = self._calculate_lambdas()
        for r in list(self.all_labels):
            weights[r] = self.default_weight - (omegas[r]/lambdas[r])*(self.tau_hat / (1 - self.tau_hat)) 
        return weights

    # Sample from the original dataset M tuples of k+2 vectors
    def _subsample(self, M):
        all_tuples = []
        classes, p = zip(*self.rho_hat.items())
        for _ in range(M):
            # Choose a class according to rho-hat
            r = np.random.choice(classes, p=p) 

            # Choose the anchor-positive pair
            index, positive_index = np.random.choice(self.label_to_indices[r], size=2, replace=False)
            current_instance = [index, positive_index]

            # Get negative indices strictly from different classes
            negative_labels = np.random.choice([l for l in self.all_labels if l != r], self.k, replace=True)
            for neg_label in negative_labels:
                negative_index = np.random.choice(self.label_to_indices[neg_label])
                current_instance.append(negative_index)
            all_tuples.append({'tuple': current_instance, 'weight': 1.0})
        return all_tuples

    def _weighted_subsample(self, M):
        all_tuples = []
        classes, p = zip(*self.rho_hat.items())
        for _ in range(M):
           # Set to default weight
            weight = self.default_weight

            # Choose a class according to rho-hat
            r = np.random.choice(classes, p=p) 
            rho_hat_r = self.rho_hat[r]

            # Choose the anchor-positive pair
            index, positive_index = np.random.choice(self.label_to_indices[r], size=2, replace=False)
            current_instance = [index, positive_index]

            # Get negative indices from all classes
            negative_labels = np.random.choice(classes, p=p, size=self.k, replace=True)

            # Check collision
            if r in negative_labels:
                weight = self.collided_weights[r]
            weight = weight * rho_hat_r

            for neg_label in negative_labels:
                negative_index = np.random.choice(self.label_to_indices[neg_label])
                current_instance.append(negative_index)
            all_tuples.append({'tuple': current_instance, 'weight': weight})
        return all_tuples

    def get_dataset(self):
        return UnsupervisedDataset(self.dataset, self.all_tuples)

# Tuple data loader definition
class UnsupervisedDataset(Dataset):
    def __init__(self, dataset, tuples): 
        # Just to make sure
        super().__init__()

        # Store configurations
        self.dataset = dataset
        self.all_tuples = tuples
        
    def __getitem__(self, index):
        # Get tuple of instance indices
        weight = self.all_tuples[index]['weight']
        current_instance = self.all_tuples[index]['tuple']
        anchor_idx, positive_idx = current_instance[0], current_instance[1]

        # Get positive + anchor instance
        x, _ = self.dataset[anchor_idx]
        x_positive, _ = self.dataset[positive_idx]

        # Flatten
        if len(x.shape) >= 2:
            x = x.view(-1)
            x_positive = x_positive.view(-1)
        
        # Get negative samples 
        negative_samples = []
        for negative_idx in current_instance[2:]:
            x_negative, _ = self.dataset[negative_idx]
            if len(x_negative.shape) >= 2:
                x_negative = x_negative.view(-1)
            negative_samples.append(x_negative)
        return (x, x_positive, negative_samples, weight)

    def __len__(self):
        return len(self.all_tuples)

