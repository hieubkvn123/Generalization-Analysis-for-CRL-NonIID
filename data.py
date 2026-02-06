import random
import numpy as np
from collections import Counter

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

DATASET_MAP = { 'mnist': datasets.MNIST, 'fashion_mnist': datasets.FashionMNIST, 'cifar10': datasets.CIFAR10 }

# Distrust random initialization
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------
# Load and subsample 
# -----------------------------------------------------
def load_imbalanced_dataset(config, seed=42):
    set_seed(seed)
    
    # Define transforms
    transform = transforms.Compose([ transforms.ToTensor() ])
    if config.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

    # Load full dataset
    dataclass = DATASET_MAP[config.dataset]
    train_dataset = dataclass(root='./data', train=True,  download=True, transform=transform)
    test_dataset  = dataclass(root='./data', train=False, download=True, transform=transform)
    
    # Convert to numpy for processing
    train_images = []
    train_labels = []
    for img, label in train_dataset:
        train_images.append(img)
        train_labels.append(label)
    train_images = torch.stack(train_images)
    train_labels = torch.tensor(train_labels)
    
    test_images = []
    test_labels = []
    for img, label in test_dataset:
        test_images.append(img)
        test_labels.append(label)
    test_images = torch.stack(test_images)
    test_labels = torch.tensor(test_labels)
    
    # Create imbalanced distribution
    n = config.n_samples
    n_classes = config.n_classes
    
    # Generate class distribution
    class_sizes = np.zeros(n_classes, dtype=int)
    class_sizes[0] = int(config.rho_max * n)
    remaining = n - class_sizes[0]
    
    # Exponentially decreasing for remaining classes
    weights = np.exp(-0.1 * np.arange(1, n_classes))
    weights = weights / weights.sum()
    
    for i in range(1, n_classes):
        class_sizes[i] = max(2, int(weights[i-1] * remaining))  # At least 2 per class
    
    # Adjust to exactly match n_samples
    diff = n - class_sizes.sum()
    class_sizes[-1] += diff
    
    print("\nClass distribution:")
    print(f"  Dominant class 0: {class_sizes[0]} ({100*class_sizes[0]/n:.1f}%)")
    print(f"  Rarest 5 classes: {class_sizes[-5:]}")
    print(f"  Total samples: {class_sizes.sum()}")
    
    # Sample from each class
    selected_indices = []
    for c in range(n_classes):
        class_mask = train_labels == c
        class_indices = torch.where(class_mask)[0]
        
        if len(class_indices) >= class_sizes[c]:
            chosen = np.random.choice(class_indices.numpy(), 
                                    size=class_sizes[c], replace=False)
        else:
            chosen = class_indices.numpy()
        
        selected_indices.extend(chosen)
    
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)
    
    X_train = train_images[selected_indices]
    labels_train = train_labels[selected_indices]
    
    # For test set, sample proportionally
    test_indices = []
    test_samples_per_class = config.test_size // n_classes
    for c in range(n_classes):
        class_mask = test_labels == c
        class_indices = torch.where(class_mask)[0]
        
        if len(class_indices) >= test_samples_per_class:
            chosen = np.random.choice(class_indices.numpy(),
                                    size=test_samples_per_class, replace=False)
        else:
            chosen = class_indices.numpy()
        test_indices.extend(chosen)
    
    test_indices = np.array(test_indices)
    X_test = test_images[test_indices]
    labels_test = test_labels[test_indices]
    
    return X_train, labels_train, X_test, labels_test, class_sizes


# -----------------------------------------------------
# Collate function
# -----------------------------------------------------
def collate_tuples(batch):
    anchors = torch.stack([item[0] for item in batch])
    positives = torch.stack([item[1] for item in batch])
    negatives = torch.stack([item[2] for item in batch])
    weights = torch.stack([item[3] for item in batch])
    return anchors, positives, negatives, weights


# -----------------------------------------------------
# Contrastive Tuple Dataset
# -----------------------------------------------------
class ContrastiveTupleDataset(Dataset):
    def __init__(self, X, labels, k_negatives, num_tuples, use_weighting=True, avoid_collision=False):
        self.X = X
        self.N = len(X)
        self.k = k_negatives
        self.labels = labels.cpu().numpy()
        self.num_tuples = num_tuples
        self.use_weighting = use_weighting
        self.avoid_collision = avoid_collision
        
        # Precompute class information
        self.R = len(np.unique(self.labels)) 
        self.classes = list(np.unique(self.labels))
        self.class_counts = dict(Counter(self.labels))
        self.class_probs  = {x: y/len(self.labels) for x, y in self.class_counts.items()}
        self.tau_hat = self.compute_tauhat()
        
        # Precompute class indices for faster sampling
        self.class_indices = {}
        for c in range(self.R):
            self.class_indices[c] = torch.where(labels == c)[0].tolist()

        # Precompute negative indices
        self.class_to_neg_indices = {c:[] for c in range(self.R)}
        for c in range(self.R):
            self.class_to_neg_indices[c] = [i for i in range(len(self.X)) if self.labels[i] != c]

        # Precompute all weights
        self.weights, self.minor_classes = self.precompute_weights()
        for r in self.classes:
            print(f'Weight for class {r}: {self.weights[(r, True)]}, {self.weights[(r, False)]}')
    
    def __len__(self):
        return self.num_tuples

    def compute_tauhat(self):
        tau = 0.0
        for _, rho_hat in self.class_probs.items():
            tau += rho_hat * ((1 - rho_hat) ** self.k)
        return tau
    
    def sample_tuple(self, class_r):
        class_indices = self.class_indices[class_r]
        
        if len(class_indices) < 2:
            return None
        
        anchor_idx, pos_idx = random.sample(class_indices, 2)
        anchor_label = self.labels[anchor_idx].item()
        
        if self.avoid_collision or anchor_label in self.minor_classes:
            available = self.class_to_neg_indices[class_r]
        else:
            available = [i for i in range(len(self.X)) if i not in (anchor_idx, pos_idx)]
        if len(available) < self.k: return None
        
        neg_indices = random.sample(available, self.k)
        return (anchor_idx, pos_idx, neg_indices)

    def precompute_weights(self):
        weights = {}
        minor_classes = []
        for r in self.classes:
            weights[(r, True)] = weights[(r, False)] = 0.0

        threshold = (3 * self.tau_hat * (self.N - 2)) / self.k
        for r in self.classes:
            if self.class_counts[r] <= threshold:
                product = np.prod((self.N - np.arange(self.k) - self.class_counts[r]) / (self.N - np.arange(self.k) - 2))
                weights[(r, True)] = (1 / (1 - self.tau_hat)) * product
                weights[(r, False)] = weights[(r, True)]
                minor_classes.append(r)
            else:
                weights[(r, True)]  = (1 / (1 - self.tau_hat)) - threshold * (1/((1-self.tau_hat) * (self.class_counts[r] - 2)))
                weights[(r, False)] = 1 / (1 - self.tau_hat)
        return weights, minor_classes

    def compute_weight(self, tuple_indices):
        if not self.use_weighting:
            return 1.0
        anchor_idx, pos_idx, neg_indices = tuple_indices
        anchor_label = self.labels[anchor_idx].item()
        has_collision = any(self.labels[n].item() == anchor_label for n in neg_indices)
        return self.weights[(anchor_label, has_collision)]
    
    def __getitem__(self, idx):
        class_r = np.random.choice(
            list(self.class_probs.keys()),
            p=list(self.class_probs.values())
        ) 
        tuple_indices = self.sample_tuple(class_r)
        anchor_idx, pos_idx, neg_indices = tuple_indices
        weight = self.compute_weight(tuple_indices)
        
        return (self.X[anchor_idx], 
                self.X[pos_idx], 
                self.X[neg_indices],
                torch.tensor(weight, dtype=torch.float32))

