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
    def __init__(self, dataset, k, n, regime='subsample'):
        # Just to make sure
        super().__init__()
        assert regime in ['subsample', 'all'], f'Invalid regime {regime}'

        # Store configurations
        self.k = k
        self.n = n
        self.dataset = dataset
        
        # Map labels to instance indices
        self.label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset):
            self.label_to_indices[label].append(idx)
        self.label_to_indices = {label: np.array(indices) for label, indices in self.label_to_indices.items()}

        # Get contrastive tuples
        self.all_labels = list(self.label_to_indices.keys())
        if regime == 'subsample':
            self.all_tuples = self._subsample_data_tuples(n)
        else:
            self.all_tuples = self._generate_all_data_tuples()

    def get_dataset(self):
        return UnsupervisedDataset(self.dataset, self.all_tuples)

    # Get all possible tuples
    def _generate_all_data_tuples(self):
        # Initialize
        all_tuples = set()

        # For all class
        for label in list(self.label_to_indices.keys()):
            positive_samples = set(self.label_to_indices[label])
            negative_samples = set(np.concatenate([v for k, v in self.label_to_indices.items() if k != label]))

            # Get all permutations from positive and combinations from negative
            permutations_pos = set(itertools.permutations(positive_samples, 2))
            combinations_neg = set(itertools.combinations(negative_samples, self.k))

            # Take the cartesian product
            cartesian = set(itertools.product(permutations_pos, combinations_neg))
            cartesian_flatten = set([p + c for p, c in cartesian])
            all_tuples = all_tuples.union(cartesian_flatten)
        return list(all_tuples)

    # Sample from the original dataset n tuples of k+2 vectors
    def _subsample_data_tuples(self, n):
        all_tuples = []
        for _ in range(n):
            # Get a random index from original dataset
            index = np.random.randint(0, len(self.dataset))
            _, label = self.dataset[index]

            # Get positive index
            positive_index = np.random.choice(self.label_to_indices[label])
            current_instance = [index, positive_index]

            # Get negative indices
            negative_labels = np.random.choice([l for l in self.all_labels if l != label], self.k, replace=True)
            for neg_label in negative_labels:
                negative_index = np.random.choice(self.label_to_indices[neg_label])
                current_instance.append(negative_index)
            all_tuples.append(current_instance)
        return all_tuples

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
        current_instance = self.all_tuples[index]
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
        return (x, x_positive, negative_samples)

    def __len__(self):
        return len(self.all_tuples)

