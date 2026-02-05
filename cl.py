import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
from tqdm import tqdm
import os


# ==================== Network Architectures ====================
class DNN(nn.Module):
    """Deep Neural Network for flattened inputs"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], output_dim=64, l2_normalize=False):
        super(DNN, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.l2_normalize = l2_normalize
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.network(x)

        # L2 normalize for contrastive learning
        if self.l2_normalize:
            x = nn.functional.normalize(x, p=2, dim=1)
        return x


class CNN(nn.Module):
    """Convolutional Neural Network"""
    def __init__(self, in_channels, output_dim=64, l2_normalize=False):
        super(CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )
        
        # Will be set dynamically based on input size
        self.fc_layers = None
        self.output_dim = output_dim
        self.l2_normalize = l2_normalize
        
    def forward(self, x):
        x = self.conv_layers(x)
        
        # Initialize fc layers on first forward pass
        if self.fc_layers is None:
            flattened_size = x.view(x.size(0), -1).size(1)
            self.fc_layers = nn.Sequential(
                nn.Linear(flattened_size, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, self.output_dim)
            ).to(x.device)
        
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        # L2 normalize
        if self.l2_normalize:
            x = nn.functional.normalize(x, p=2, dim=1)
        return x


# ==================== Dataset Preparation ====================

def create_imbalanced_dataset(dataset_name, N=10000, class_ratios=None):
    """
    Create an imbalanced dataset with fixed class ratios
    
    Args:
        dataset_name: 'MNIST', 'CIFAR10', or 'FashionMNIST'
        N: Total number of samples
        class_ratios: List of ratios for each class (should sum to 1.0)
                     Default: [0.5, 0.1, 0.1, 0.1, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02]
    
    Returns:
        dataset, indices, class_counts, full_dataset
    """
    if class_ratios is None:
        class_ratios = [0.5, 0.1, 0.1, 0.1, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load dataset
    if dataset_name == 'MNIST':
        full_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                                   download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                      download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'FashionMNIST':
        full_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                          download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Get all labels
    if hasattr(full_dataset, 'targets'):
        all_labels = np.array(full_dataset.targets)
    else:
        all_labels = np.array([full_dataset[i][1] for i in range(len(full_dataset))])
    
    # Calculate samples per class based on ratios
    assert len(class_ratios) == num_classes, f"Need {num_classes} ratios, got {len(class_ratios)}"
    assert abs(sum(class_ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(class_ratios)}"
    
    # Collect indices for each class
    selected_indices = []
    class_counts = {}
    
    for c in range(num_classes):
        class_indices = np.where(all_labels == c)[0]
        n_samples = int(N * class_ratios[c])
        
        # Randomly select n_samples from this class
        selected = np.random.choice(class_indices, size=min(n_samples, len(class_indices)), 
                                    replace=False)
        selected_indices.extend(selected.tolist())
        class_counts[c] = len(selected)
    
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)
    
    # Create subset
    subset = Subset(full_dataset, selected_indices)
    
    print(f"\n{dataset_name} Imbalanced Dataset Statistics:")
    print(f"Total samples: {len(selected_indices)}")
    print(f"Class distribution: {class_counts}")
    print(f"Class ratios: {[f'{class_counts[c]/len(selected_indices):.3f}' for c in range(num_classes)]}")
    
    return subset, selected_indices, class_counts, full_dataset


def create_balanced_test_dataset(dataset_name, samples_per_class=200):
    """
    Create a balanced test dataset for evaluation
    
    Args:
        dataset_name: 'MNIST', 'CIFAR10', or 'FashionMNIST'
        samples_per_class: Number of samples per class
    
    Returns:
        test_dataset, test_indices, test_class_counts, full_test_dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load test dataset
    if dataset_name == 'MNIST':
        full_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                                        download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                           download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'FashionMNIST':
        full_test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                               download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Get all labels
    if hasattr(full_test_dataset, 'targets'):
        all_labels = np.array(full_test_dataset.targets)
    else:
        all_labels = np.array([full_test_dataset[i][1] for i in range(len(full_test_dataset))])
    
    # Collect balanced samples
    selected_indices = []
    class_counts = {}
    
    for c in range(num_classes):
        class_indices = np.where(all_labels == c)[0]
        n_samples = min(samples_per_class, len(class_indices))
        
        # Randomly select n_samples from this class
        selected = np.random.choice(class_indices, size=n_samples, replace=False)
        selected_indices.extend(selected.tolist())
        class_counts[c] = len(selected)
    
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)
    
    # Create subset
    test_subset = Subset(full_test_dataset, selected_indices)
    
    print(f"\n{dataset_name} Balanced Test Dataset Statistics:")
    print(f"Total samples: {len(selected_indices)}")
    print(f"Class distribution: {class_counts}")
    
    return test_subset, selected_indices, class_counts, full_test_dataset


# ==================== Sampling Procedures ====================

def precompute_tuples_procedure_a(dataset, labels, k, M, device):
    """
    Procedure A: Sample without replacement from S_r for positives, 
                 and from S \ S_r for negatives
    
    Returns:
        List of tuples (anchor_idx, positive_idx, negative_indices)
    """
    print("\nPrecomputing tuples for Procedure A...")
    
    # Get class distribution
    unique_classes = np.unique(labels)
    R = len(unique_classes)
    N = len(labels)
    
    # Calculate class probabilities
    class_counts = {c: np.sum(labels == c) for c in unique_classes}
    class_probs = {c: class_counts[c] / N for c in unique_classes}
    
    # Organize indices by class
    class_indices = {c: np.where(labels == c)[0] for c in unique_classes}
    
    tuples = []
    
    for _ in tqdm(range(M), desc="Sampling tuples (Procedure A)"):
        # Choose a label r with probability rho_r
        r = np.random.choice(unique_classes, p=[class_probs[c] for c in unique_classes])
        
        # Choose positive pair without replacement from S_r
        if len(class_indices[r]) < 2:
            continue  # Skip if not enough samples in this class
        
        anchor_idx, positive_idx = np.random.choice(class_indices[r], size=2, replace=False)
        
        # Choose k negatives without replacement from S \ S_r
        negative_pool = np.concatenate([class_indices[c] for c in unique_classes if c != r])
        
        if len(negative_pool) < k:
            continue  # Skip if not enough negatives
        
        negative_indices = np.random.choice(negative_pool, size=k, replace=False)
        
        tuples.append((anchor_idx, positive_idx, negative_indices))
    
    print(f"Generated {len(tuples)} tuples for Procedure A")
    return tuples


def precompute_tuples_procedure_b(dataset, labels, k, M, device):
    """
    Procedure B: Sub-sampled estimation with collision handling
    
    Returns:
        List of tuples (anchor_idx, positive_idx, negative_indices, weight)
    """
    print("\nPrecomputing tuples for Procedure B...")
    
    # Get class distribution
    unique_classes = np.unique(labels)
    R = len(unique_classes)
    N = len(labels)
    
    # Calculate class probabilities and counts
    class_counts = {c: np.sum(labels == c) for c in unique_classes}
    class_probs = {c: class_counts[c] / N for c in unique_classes}
    rho_hat = np.array([class_probs[c] for c in unique_classes])
    
    # Calculate tau_hat
    tau_hat = np.sum(rho_hat * (1 - rho_hat) ** k)
    
    # Organize indices by class
    class_indices = {c: np.where(labels == c)[0] for c in unique_classes}
    
    tuples = []
    
    for _ in tqdm(range(M), desc="Sampling tuples (Procedure B)"):
        # Select r with probability rho_r
        r = np.random.choice(unique_classes, p=[class_probs[c] for c in unique_classes])
        N_r = class_counts[r]
        
        # Determine if we should avoid collision
        threshold = (3 * tau_hat * (N - 2)) / k + 2
        avoid_collision = (N_r <= threshold)
        
        if avoid_collision:
            # Sample from Theta_r (no collision with anchor class)
            if len(class_indices[r]) < 2:
                continue
            
            anchor_idx, positive_idx = np.random.choice(class_indices[r], size=2, replace=False)
            
            # Negatives from S \ S_r
            negative_pool = np.concatenate([class_indices[c] for c in unique_classes if c != r])
            
            if len(negative_pool) < k:
                continue
            
            negative_indices = np.random.choice(negative_pool, size=k, replace=False)
            
            # Calculate weight
            weight = (1 / (1 - tau_hat)) * np.prod([(N - ell - N_r) / (N - ell - 2) 
                                                     for ell in range(k)])
        else:
            # Sample from Omega_r (allow collision)
            if len(class_indices[r]) < 2:
                continue
            
            anchor_idx, positive_idx = np.random.choice(class_indices[r], size=2, replace=False)
            
            # Negatives from S \ {X, X^+}
            available_pool = np.array([idx for idx in range(N) 
                                       if idx != anchor_idx and idx != positive_idx])
            
            if len(available_pool) < k:
                continue
            
            negative_indices = np.random.choice(available_pool, size=k, replace=False)
            
            # Check if any negative is from class r (Lambda_r indicator)
            is_in_lambda_r = any(labels[neg_idx] == r for neg_idx in negative_indices)
            
            # Calculate weight
            weight = (1 / (1 - tau_hat)) - (3 * tau_hat * (N - 2)) / \
                     (k * (1 - tau_hat) * (N_r - 2)) * float(is_in_lambda_r)
        
        tuples.append((anchor_idx, positive_idx, negative_indices, weight))
    
    print(f"Generated {len(tuples)} tuples for Procedure B")
    return tuples


# ==================== Contrastive Loss ====================

def contrastive_loss_phi(anchor, positive, negatives):
    """
    Compute phi(x, x^+, x_1^-, ..., x_k^-)
    = log(1 + sum_j exp(f(x)^T f(x_j^-) - f(x)^T f(x^+)))
    
    Args:
        anchor: (batch_size, feature_dim)
        positive: (batch_size, feature_dim)
        negatives: (batch_size, k, feature_dim)
    """
    # Compute f(x)^T f(x^+)
    pos_sim = torch.sum(anchor * positive, dim=1, keepdim=True)  # (batch_size, 1)
    
    # Compute f(x)^T f(x_j^-) for all j
    # anchor: (batch_size, feature_dim) -> (batch_size, 1, feature_dim)
    # negatives: (batch_size, k, feature_dim)
    neg_sim = torch.sum(anchor.unsqueeze(1) * negatives, dim=2)  # (batch_size, k)
    
    # Compute differences
    logits = neg_sim - pos_sim  # (batch_size, k)
    
    # Compute log(1 + sum exp(...))
    loss = torch.log(1 + torch.sum(torch.exp(logits), dim=1))  # (batch_size,)
    
    return loss


def compute_loss_procedure_a(model, tuples, dataset, device, batch_size=256):
    """Compute R_a(f) for Procedure A"""
    model.eval()
    total_loss = 0.0
    num_tuples = len(tuples)
    
    with torch.no_grad():
        for i in range(0, num_tuples, batch_size):
            batch_tuples = tuples[i:min(i + batch_size, num_tuples)]
            
            # Extract indices
            anchor_indices = [t[0] for t in batch_tuples]
            positive_indices = [t[1] for t in batch_tuples]
            negative_indices_list = [t[2] for t in batch_tuples]
            
            # Get data
            anchors = torch.stack([dataset[idx][0] for idx in anchor_indices]).to(device)
            positives = torch.stack([dataset[idx][0] for idx in positive_indices]).to(device)
            
            # Get negatives (k per anchor)
            k = len(negative_indices_list[0])
            negatives_list = []
            for neg_indices in negative_indices_list:
                negs = torch.stack([dataset[idx][0] for idx in neg_indices]).to(device)
                negatives_list.append(negs)
            negatives = torch.stack(negatives_list)  # (batch, k, C, H, W)
            
            # Get embeddings
            anchor_emb = model(anchors)
            positive_emb = model(positives)
            
            # Reshape negatives for batch processing
            batch_sz, k_val = negatives.shape[0], negatives.shape[1]
            negatives_flat = negatives.view(batch_sz * k_val, *negatives.shape[2:])
            negative_emb = model(negatives_flat)
            negative_emb = negative_emb.view(batch_sz, k_val, -1)
            
            # Compute loss
            loss = contrastive_loss_phi(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.sum().item()
    
    return total_loss / num_tuples


def compute_loss_procedure_b(model, tuples, dataset, device, batch_size=256):
    """Compute R_b(f) for Procedure B"""
    model.eval()
    total_loss = 0.0
    num_tuples = len(tuples)
    
    with torch.no_grad():
        for i in range(0, num_tuples, batch_size):
            batch_tuples = tuples[i:min(i + batch_size, num_tuples)]
            
            # Extract indices and weights
            anchor_indices = [t[0] for t in batch_tuples]
            positive_indices = [t[1] for t in batch_tuples]
            negative_indices_list = [t[2] for t in batch_tuples]
            weights = torch.tensor([t[3] for t in batch_tuples], device=device)
            
            # Get data
            anchors = torch.stack([dataset[idx][0] for idx in anchor_indices]).to(device)
            positives = torch.stack([dataset[idx][0] for idx in positive_indices]).to(device)
            
            # Get negatives
            k = len(negative_indices_list[0])
            negatives_list = []
            for neg_indices in negative_indices_list:
                negs = torch.stack([dataset[idx][0] for idx in neg_indices]).to(device)
                negatives_list.append(negs)
            negatives = torch.stack(negatives_list)
            
            # Get embeddings
            anchor_emb = model(anchors)
            positive_emb = model(positives)
            
            # Reshape negatives for batch processing
            batch_sz, k_val = negatives.shape[0], negatives.shape[1]
            negatives_flat = negatives.view(batch_sz * k_val, *negatives.shape[2:])
            negative_emb = model(negatives_flat)
            negative_emb = negative_emb.view(batch_sz, k_val, -1)
            
            # Compute weighted loss
            loss = contrastive_loss_phi(anchor_emb, positive_emb, negative_emb)
            weighted_loss = loss * weights
            total_loss += weighted_loss.sum().item()
    
    return total_loss / num_tuples


# ==================== Linear Classifier Evaluation ====================

class LinearClassifier(nn.Module):
    """Linear classifier on top of frozen representations"""
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


def extract_features(model, dataset, device, batch_size=256):
    """Extract features from a dataset using the representation model"""
    model.eval()
    features_list = []
    labels_list = []
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feats = model(images)
            features_list.append(feats.cpu())
            labels_list.append(labels)
    
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    return features, labels


def train_linear_classifier(features, labels, num_classes, device, 
                            num_epochs=100, lr=0.01, batch_size=256):
    """Train a linear classifier on top of frozen features"""
    # Create dataset and dataloader
    feature_dataset = torch.utils.data.TensorDataset(features, labels)
    dataloader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=True)
    
    # Create classifier
    input_dim = features.shape[1]
    classifier = LinearClassifier(input_dim, num_classes).to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining linear classifier...")
    print(f"Input dim: {input_dim}, Output classes: {num_classes}")
    print(f"Training samples: {len(features)}")
    
    classifier.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = classifier(batch_features)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        if (epoch + 1) % 20 == 0:
            acc = 100. * correct / total
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, "
                  f"Train Acc: {acc:.2f}%")
    
    return classifier


def evaluate_classifier(classifier, features, labels, device, 
                       rare_classes=None, batch_size=256):
    """
    Evaluate classifier and compute metrics for rare classes
    
    Args:
        classifier: Trained linear classifier
        features: Feature tensor
        labels: Label tensor
        device: Device to use
        rare_classes: List of rare class indices (e.g., [5, 6, 7, 8, 9] for top 5 rarest)
    
    Returns:
        Dictionary with overall and rare class metrics
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
    
    classifier.eval()
    
    # Create dataloader
    feature_dataset = torch.utils.data.TensorDataset(features, labels)
    dataloader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            outputs = classifier(batch_features)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall metrics
    overall_acc = (all_preds == all_labels).mean()
    
    # Metrics for rare classes
    if rare_classes is not None:
        # Filter to only rare classes
        rare_mask = np.isin(all_labels, rare_classes)
        rare_labels = all_labels[rare_mask]
        rare_preds = all_preds[rare_mask]
        
        if len(rare_labels) > 0:
            # Micro-averaged metrics (treat all samples equally)
            rare_precision = precision_score(rare_labels, rare_preds, 
                                            labels=rare_classes, average='micro', zero_division=0)
            rare_recall = recall_score(rare_labels, rare_preds, 
                                      labels=rare_classes, average='micro', zero_division=0)
            rare_f1 = f1_score(rare_labels, rare_preds, 
                              labels=rare_classes, average='micro', zero_division=0)
            
            # Per-class metrics for rare classes
            rare_precision_per_class = precision_score(rare_labels, rare_preds, 
                                                       labels=rare_classes, average=None, zero_division=0)
            rare_recall_per_class = recall_score(rare_labels, rare_preds, 
                                                 labels=rare_classes, average=None, zero_division=0)
            rare_f1_per_class = f1_score(rare_labels, rare_preds, 
                                        labels=rare_classes, average=None, zero_division=0)
        else:
            rare_precision = rare_recall = rare_f1 = 0.0
            rare_precision_per_class = rare_recall_per_class = rare_f1_per_class = np.zeros(len(rare_classes))
    else:
        rare_precision = rare_recall = rare_f1 = None
        rare_precision_per_class = rare_recall_per_class = rare_f1_per_class = None
    
    results = {
        'overall_accuracy': overall_acc,
        'rare_precision_micro': rare_precision,
        'rare_recall_micro': rare_recall,
        'rare_f1_micro': rare_f1,
        'rare_precision_per_class': rare_precision_per_class,
        'rare_recall_per_class': rare_recall_per_class,
        'rare_f1_per_class': rare_f1_per_class,
        'all_predictions': all_preds,
        'all_labels': all_labels
    }
    
    return results


# ==================== Training ====================

def train_epoch(model, optimizer, tuples, dataset, procedure, device, batch_size=128):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Shuffle tuples
    indices = np.random.permutation(len(tuples))
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:min(i + batch_size, len(indices))]
        batch_tuples = [tuples[idx] for idx in batch_indices]
        
        if procedure == 'a':
            # Extract indices
            anchor_indices = [t[0] for t in batch_tuples]
            positive_indices = [t[1] for t in batch_tuples]
            negative_indices_list = [t[2] for t in batch_tuples]
            weights = None
        else:  # procedure == 'b'
            # Extract indices and weights
            anchor_indices = [t[0] for t in batch_tuples]
            positive_indices = [t[1] for t in batch_tuples]
            negative_indices_list = [t[2] for t in batch_tuples]
            weights = torch.tensor([t[3] for t in batch_tuples], device=device)
        
        # Get data
        anchors = torch.stack([dataset[idx][0] for idx in anchor_indices]).to(device)
        positives = torch.stack([dataset[idx][0] for idx in positive_indices]).to(device)
        
        # Get negatives
        k = len(negative_indices_list[0])
        negatives_list = []
        for neg_indices in negative_indices_list:
            negs = torch.stack([dataset[idx][0] for idx in neg_indices]).to(device)
            negatives_list.append(negs)
        negatives = torch.stack(negatives_list)
        
        # Forward pass
        anchor_emb = model(anchors)
        positive_emb = model(positives)
        
        # Reshape negatives for batch processing
        batch_sz, k_val = negatives.shape[0], negatives.shape[1]
        negatives_flat = negatives.view(batch_sz * k_val, *negatives.shape[2:])
        negative_emb = model(negatives_flat)
        negative_emb = negative_emb.view(batch_sz, k_val, -1)
        
        # Compute loss
        loss_per_sample = contrastive_loss_phi(anchor_emb, positive_emb, negative_emb)
        
        if weights is not None:
            loss_per_sample = loss_per_sample * weights
        
        loss = loss_per_sample.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def train_model(model, dataset, tuples, procedure, device, 
                num_epochs=50, lr=0.001, batch_size=128):
    """
    Train the model with the specified procedure
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"\nTraining with Procedure {procedure.upper()}...")
    print(f"Number of tuples: {len(tuples)}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, optimizer, tuples, dataset, procedure, 
                                device, batch_size)
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return model


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Contrastive Learning Experiment')
    parser.add_argument('--dataset', type=str, default='MNIST', 
                       choices=['MNIST', 'CIFAR10', 'FashionMNIST'],
                       help='Dataset to use')
    parser.add_argument('--model', type=str, default='CNN',
                       choices=['DNN', 'CNN'],
                       help='Model architecture')
    parser.add_argument('--procedure', type=str, default='a',
                       choices=['a', 'b'],
                       help='Sampling procedure (a or b)')
    parser.add_argument('--M', type=int, default=5000,
                       help='Number of tuples to sample')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of negative samples')
    parser.add_argument('--N', type=int, default=10000,
                       help='Total dataset size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output_dim', type=int, default=64,
                       help='Output embedding dimension')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--eval_classifier', action='store_true',
                       help='Train and evaluate linear classifier after contrastive learning')
    parser.add_argument('--classifier_epochs', type=int, default=100,
                       help='Number of epochs for training linear classifier')
    parser.add_argument('--test_samples_per_class', type=int, default=200,
                       help='Number of samples per class in balanced test set')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Fixed class ratios as specified
    class_ratios = [0.5, 0.1, 0.1, 0.1, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02]
    
    # Create imbalanced dataset for training
    subset, selected_indices, class_counts, full_dataset = create_imbalanced_dataset(
        args.dataset, N=args.N, class_ratios=class_ratios
    )
    
    # Get labels for selected indices
    if hasattr(full_dataset, 'targets'):
        all_labels = np.array(full_dataset.targets)
    else:
        all_labels = np.array([full_dataset[i][1] for i in range(len(full_dataset))])
    
    labels = all_labels[selected_indices]
    
    # Identify the 5 rarest classes (indices 5-9 have ratio 0.02 each)
    rare_classes = [5, 6, 7, 8, 9]
    print(f"\nRarest classes (for evaluation): {rare_classes}")
    
    # Precompute tuples
    if args.procedure == 'a':
        tuples = precompute_tuples_procedure_a(subset, labels, args.k, args.M, device)
    else:
        tuples = precompute_tuples_procedure_b(subset, labels, args.k, args.M, device)
    
    # Create model
    sample_data = subset[0][0]
    
    if args.model == 'DNN':
        input_dim = np.prod(sample_data.shape)
        model = DNN(input_dim=input_dim, output_dim=args.output_dim).to(device)
    else:  # CNN
        in_channels = sample_data.shape[0]
        model = CNN(in_channels=in_channels, output_dim=args.output_dim).to(device)
    
    print(f"\nModel architecture: {args.model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    model = train_model(model, subset, tuples, args.procedure, device,
                       num_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{args.dataset}_{args.model}_proc{args.procedure}_M{args.M}_k{args.k}.pt'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Final evaluation of contrastive loss
    if args.procedure == 'a':
        final_loss = compute_loss_procedure_a(model, tuples, subset, device)
    else:
        final_loss = compute_loss_procedure_b(model, tuples, subset, device)
    
    print(f"Final contrastive loss: {final_loss:.4f}")
    
    # Linear classifier evaluation
    if args.eval_classifier:
        print("\n" + "="*80)
        print("LINEAR CLASSIFIER EVALUATION")
        print("="*80)
        
        # Create balanced test dataset
        test_subset, test_indices, test_class_counts, full_test_dataset = \
            create_balanced_test_dataset(args.dataset, samples_per_class=args.test_samples_per_class)
        
        # Extract features from training set (imbalanced)
        print("\nExtracting features from training set (imbalanced)...")
        train_features, train_labels = extract_features(model, subset, device)
        
        # Extract features from test set (balanced)
        print("Extracting features from test set (balanced)...")
        test_features, test_labels = extract_features(model, test_subset, device)
        
        # Train linear classifier on imbalanced training set
        num_classes = 10
        classifier = train_linear_classifier(
            train_features, train_labels, num_classes, device,
            num_epochs=args.classifier_epochs, lr=0.01
        )
        
        # Evaluate on balanced test set
        print("\nEvaluating on balanced test set...")
        results = evaluate_classifier(classifier, test_features, test_labels, 
                                     device, rare_classes=rare_classes)
        
        # Print results
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"\nOverall Test Accuracy: {results['overall_accuracy']*100:.2f}%")
        print(f"\nRare Classes (5 rarest: {rare_classes}):")
        print(f"  Micro-averaged Precision: {results['rare_precision_micro']*100:.2f}%")
        print(f"  Micro-averaged Recall:    {results['rare_recall_micro']*100:.2f}%")
        print(f"  Micro-averaged F1-score:  {results['rare_f1_micro']*100:.2f}%")
        
        print(f"\nPer-class metrics for rare classes:")
        for i, cls in enumerate(rare_classes):
            print(f"  Class {cls}: Precision={results['rare_precision_per_class'][i]*100:.2f}%, "
                  f"Recall={results['rare_recall_per_class'][i]*100:.2f}%, "
                  f"F1={results['rare_f1_per_class'][i]*100:.2f}%")
        
        # Save classifier and results
        classifier_path = f'models/{args.dataset}_{args.model}_proc{args.procedure}_M{args.M}_k{args.k}_classifier.pt'
        torch.save(classifier.state_dict(), classifier_path)
        print(f"\nClassifier saved to {classifier_path}")
        
        # Save results to file
        results_path = f'models/{args.dataset}_{args.model}_proc{args.procedure}_M{args.M}_k{args.k}_results.txt'
        with open(results_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"Evaluation Results - {args.dataset} {args.model} Procedure {args.procedure.upper()}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Dataset: {args.dataset}\n")
            f.write(f"  Model: {args.model}\n")
            f.write(f"  Procedure: {args.procedure}\n")
            f.write(f"  M (tuples): {args.M}\n")
            f.write(f"  k (negatives): {args.k}\n")
            f.write(f"  Contrastive epochs: {args.epochs}\n")
            f.write(f"  Classifier epochs: {args.classifier_epochs}\n\n")
            f.write(f"Final Contrastive Loss: {final_loss:.4f}\n\n")
            f.write(f"Overall Test Accuracy: {results['overall_accuracy']*100:.2f}%\n\n")
            f.write(f"Rare Classes ({rare_classes}):\n")
            f.write(f"  Micro-averaged Precision: {results['rare_precision_micro']*100:.2f}%\n")
            f.write(f"  Micro-averaged Recall:    {results['rare_recall_micro']*100:.2f}%\n")
            f.write(f"  Micro-averaged F1-score:  {results['rare_f1_micro']*100:.2f}%\n\n")
            f.write(f"Per-class metrics for rare classes:\n")
            for i, cls in enumerate(rare_classes):
                f.write(f"  Class {cls}: Precision={results['rare_precision_per_class'][i]*100:.2f}%, "
                       f"Recall={results['rare_recall_per_class'][i]*100:.2f}%, "
                       f"F1={results['rare_f1_per_class'][i]*100:.2f}%\n")
        
        print(f"Results saved to {results_path}")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
