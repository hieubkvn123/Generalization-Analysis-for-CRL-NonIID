import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from dataclasses import dataclass
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------
# CONSTANTS 
# -----------------------------------------------------
EPOCHS = 100
CLF_EPOCHS = 100
DATASET_MAP = { 'mnist': datasets.MNIST, 'fashion_mnist': datasets.FashionMNIST, 'cifar10': datasets.CIFAR10 }
DATASET_TO_SHAPE = { 'mnist': (1, 28, 28), 'fashion_mnist': (1, 28, 28), 'cifar10': (3, 32, 32) }

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
@dataclass
class ContrastiveConfig:
    n_samples: int = 9000  
    n_classes: int = 10
    k_negatives: int = 5
    rho_max: float = 0.45
    temperature: float = 0.5
    batch_size: int = 64 
    m_incomplete: int = 5000 
    test_size: int = 4000 
    dataset: str = 'mnist'

# -----------------------------------------------------
# ResNet-style CNN encoder
# -----------------------------------------------------
class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = F.relu(out)
        return out


class CNNEncoder(nn.Module):
    """ResNet-style CNN encoder for contrastive learning"""
    def __init__(self, in_channels=1, hidden_dim=128, output_dim=64):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection head
        self.fc1 = nn.Linear(256, hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Projection head
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        
        return F.normalize(x, dim=1)

class CNNEncoder(nn.Module):
    """Simple CNN encoder for contrastive learning on image datasets"""
    def __init__(self, in_channels=1, hidden_dim=128, output_dim=64):
        super().__init__()

        # Convolutional layers
        # For MNIST/Fashion-MNIST (28x28) and CIFAR-10 (32x32)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Projection head
        self.fc1 = nn.Linear(128 * 4 * 4, hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Conv block 1: 28x28 or 32x32 -> 14x14 or 16x16
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv block 2: 14x14 or 16x16 -> 7x7 or 8x8
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Conv block 3: 7x7 or 8x8 -> 3x3 or 4x4
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Adaptive pooling to 4x4
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        # Projection head
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return F.normalize(x, dim=1)

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x)

# -----------------------------------------------------
# Load and subsample 
# -----------------------------------------------------
def load_imbalanced_dataset(config, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
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
                weights[(r, True)] *= 0.5
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
        
        max_attempts = 10
        for _ in range(max_attempts):
            tuple_indices = self.sample_tuple(class_r)
            if tuple_indices is not None:
                break
        else:
            anchor_idx = 0
            pos_idx = 1 if len(self.X) > 1 else 0
            neg_indices = list(range(2, min(2 + self.k, len(self.X))))
            tuple_indices = (anchor_idx, pos_idx, neg_indices)
            return (self.X[anchor_idx], 
                   self.X[pos_idx], 
                   self.X[neg_indices],
                   torch.tensor(0.0))
        
        anchor_idx, pos_idx, neg_indices = tuple_indices
        weight = self.compute_weight(tuple_indices)
        
        return (self.X[anchor_idx], 
                self.X[pos_idx], 
                self.X[neg_indices],
                torch.tensor(weight, dtype=torch.float32))

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
# Batched contrastive loss
# -----------------------------------------------------
def batched_contrastive_loss(anchors, positives, negatives, temperature):
    batch_size = anchors.shape[0]
    
    pos_sim = (anchors * positives).sum(dim=1) / temperature
    neg_sims = torch.bmm(negatives, anchors.unsqueeze(2)).squeeze(2) / temperature
    v = pos_sim.unsqueeze(1) - neg_sims
    losses = torch.log1p(torch.exp(-v).sum(dim=1))
    
    return losses

# -----------------------------------------------------
# Single tuple loss
# -----------------------------------------------------
def contrastive_loss(anchor, positive, negatives, temperature):
    pos_sim = (anchor * positive).sum() / temperature
    neg_sims = (negatives @ anchor) / temperature
    v = pos_sim - neg_sims
    loss = torch.log1p(torch.exp(-v).sum())
    return loss

# -----------------------------------------------------
# Evaluate on specific classes
# -----------------------------------------------------
def evaluate_on_classes(test_dataset, encoder, config, device, target_classes):
    encoder.eval()
    X = test_dataset.X
    
    losses_per_class = {c: [] for c in target_classes}
    n_samples_per_class = 50
    
    with torch.no_grad():
        for c in target_classes:
            for _ in range(n_samples_per_class):
                t = test_dataset.sample_tuple(c)
                if t is None:
                    continue
                anchor_idx, pos_idx, neg_indices = t
                
                z_anchor = encoder(X[anchor_idx:anchor_idx+1])[0]
                z_pos = encoder(X[pos_idx:pos_idx+1])[0]
                z_negs = encoder(X[neg_indices])
                
                loss = contrastive_loss(z_anchor, z_pos, z_negs, config.temperature)
                losses_per_class[c].append(float(loss))
    
    avg_losses = {}
    for c in target_classes:
        if len(losses_per_class[c]) > 0:
            avg_losses[c] = np.mean(losses_per_class[c])
        else:
            avg_losses[c] = float('nan')
    
    encoder.train()
    return avg_losses

# -----------------------------------------------------
# Training loop
# -----------------------------------------------------
def train_contrastive_model(X_train, labels_train, X_test, labels_test, 
                           config, n_epochs=100, use_weighting=True, avoid_collision=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train = X_train.to(device)
    labels_train = labels_train.to(device)
    X_test = X_test.to(device)
    labels_test = labels_test.to(device)
    
    # Create encoder based on config
    in_channels = DATASET_TO_SHAPE[config.dataset][0]
    encoder = CNNEncoder(in_channels=in_channels, hidden_dim=128, output_dim=64).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, amsgrad=True)
    print(f"\nUsing CNN encoder with {in_channels} input channels")
    
    loss_history = []
    test_loss_history = []
    
    # Identify 5 rarest classes
    class_counts = np.bincount(labels_train.cpu().numpy())
    rarest_classes = np.argsort(class_counts)[:5].tolist()
    
    method_name = "WEIGHTED" if use_weighting else "UNWEIGHTED"
    print(f"\n{'='*60}")
    print(f"Training with {method_name} incomplete U-statistics")
    print(f"M={config.m_incomplete} tuples per epoch")
    print(f"Batch size={config.batch_size}")
    print(f"Rarest classes: {rarest_classes} with counts {class_counts[rarest_classes]}")
    print(f"{'='*60}")

    dataset = ContrastiveTupleDataset(
        X_train, labels_train, 
        config.k_negatives, 
        config.m_incomplete,
        use_weighting=use_weighting,
        avoid_collision=avoid_collision
    )

    test_dataset = ContrastiveTupleDataset(
        X_test, labels_test, 
        config.k_negatives, 
        config.m_incomplete,
        use_weighting=False,
        avoid_collision=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_tuples,
        num_workers=0
    )
    
    best_model, best_loss = None, np.inf
    for epoch in range(n_epochs):
        start = time.time()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for anchors, positives, negatives, weights in dataloader:
            optimizer.zero_grad()
            
            weights = weights.to(device)
            z_anchors = encoder(anchors)
            z_positives = encoder(positives)
            
            batch_size, k, C, H, W = negatives.shape
            z_negatives = encoder(negatives.view(batch_size * k, C, H, W))
            z_negatives = z_negatives.view(batch_size, k, -1)
            
            losses = batched_contrastive_loss(
                z_anchors, z_positives, z_negatives, 
                config.temperature
            )
            
            weighted_losses = losses * weights
            batch_loss = weighted_losses.sum() / weights.sum()
            
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)
        elapsed = time.time() - start
        
        if avg_epoch_loss < best_loss:
            best_model = encoder.state_dict()
            best_loss = avg_epoch_loss
            print(f' - Update model at epoch {epoch}, new best = {best_loss:.5f}')
        
        if (epoch + 1) % 20 == 0:
            test_losses = evaluate_on_classes(test_dataset, encoder, config, device, rarest_classes)
            avg_test_loss = np.mean([v for v in test_losses.values() if not np.isnan(v)])
            test_loss_history.append(avg_test_loss)
            
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_epoch_loss:.4f} | "
                  f"Test Loss (rare): {avg_test_loss:.4f} | Time: {elapsed:.2f}s")
    
    # Load best model
    encoder.load_state_dict(best_model)
    final_test_losses = evaluate_on_classes(test_dataset, encoder, config, device, rarest_classes)
    
    return encoder, loss_history, test_loss_history, final_test_losses, rarest_classes

def train_linear_classifier(encoder, X_train, labels_train, X_test, labels_test, config, device, n_epochs=100):
    """
    Train a linear classifier on top of frozen encoder representations
    """
    print("\n" + "="*60)
    print("TRAINING LINEAR CLASSIFIER")
    print("="*60)
    
    encoder.eval()
    
    # Extract representations
    with torch.no_grad():
        batch_size = 256
        train_reps = []
        for i in range(0, len(X_train), batch_size):
            batch = X_train[i:i+batch_size].to(device)
            reps = encoder(batch)
            train_reps.append(reps.cpu())
        train_reps = torch.cat(train_reps, dim=0).to(device)
        
        test_reps = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size].to(device)
            reps = encoder(batch)
            test_reps.append(reps.cpu())
        test_reps = torch.cat(test_reps, dim=0).to(device)
    
    embedding_dim = train_reps.shape[1]
    classifier = LinearClassifier(embedding_dim, config.n_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, amsgrad=True)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataloader
    train_dataset = torch.utils.data.TensorDataset(train_reps, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    print(f"Training for {n_epochs} epochs...")
    print(f"Train representations: {train_reps.shape}")
    print(f"Test representations: {test_reps.shape}")
    
    for epoch in range(n_epochs):
        classifier.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_reps, batch_labels in train_loader:
            optimizer.zero_grad()
            batch_labels = batch_labels.to(device)
            
            logits = classifier(batch_reps)
            loss = criterion(logits, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = logits.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        train_acc = 100. * correct / total
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Train Acc: {train_acc:.2f}%")
    
    return classifier

def evaluate_classifier_rare_classes(classifier, encoder, X_test, labels_test, rarest_classes, config, device):
    print("\n" + "="*60)
    print("EVALUATING CLASSIFIER ON RARE CLASSES")
    print("="*60)
    
    encoder.eval()
    classifier.eval()
    
    # Extract test representations
    with torch.no_grad():
        batch_size = 256
        test_reps = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size].to(device)
            reps = encoder(batch)
            test_reps.append(reps.cpu())
        test_reps = torch.cat(test_reps, dim=0).to(device)
        
        # Get predictions
        all_logits = []
        for i in range(0, len(test_reps), batch_size):
            batch = test_reps[i:i+batch_size]
            logits = classifier(batch)
            all_logits.append(logits.cpu())
        all_logits = torch.cat(all_logits, dim=0)
        predictions = all_logits.argmax(dim=1).cpu().numpy()
    
    labels_np = labels_test.cpu().numpy()
    
    # Calculate overall metrics
    overall_acc = (predictions == labels_np).mean() * 100
    
    # Compute precision and recall manually for each class
    f1, recall, precision, accuracy, support = {}, {}, {}, {}, {}
    
    for c in range(config.n_classes):
        # True positives: predicted as c AND actually c
        tp = np.sum((predictions == c) & (labels_np == c))
        
        # False positives: predicted as c BUT actually not c
        fp = np.sum((predictions == c) & (labels_np != c))
        
        # False negatives: predicted as not c BUT actually c
        fn = np.sum((predictions != c) & (labels_np == c))
        
        # Support: total number of samples in class c
        support[c] = np.sum(labels_np == c)
        
        # Precision: TP / (TP + FP)
        if tp + fp > 0:
            precision[c] = tp / (tp + fp)
        else:
            precision[c] = 0.0
        
        # Recall: TP / (TP + FN)
        if tp + fn > 0:
            recall[c] = tp / (tp + fn)
        else:
            recall[c] = 0.0
        
        # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
        if precision[c] + recall[c] > 0:
            f1[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c])
        else:
            f1[c] = 0.0
    
    print(f"\nOverall Test Accuracy: {overall_acc:.2f}%")
    print("\n" + "-"*60)
    print("RARE CLASS METRICS")
    print("-"*60)
    print(f"{'Class':<8} {'Support':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*60)
    
    rare_metrics = {}
    for c in rarest_classes:
        rare_metrics[c] = {
            'precision': precision[c],
            'recall': recall[c],
            'f1': f1[c],
            'support': support[c]
        }
        print(f"{c:<8} {support[c]:<10} {precision[c]:<12.4f} {recall[c]:<12.4f} {f1[c]:<12.4f}")
    
    print("-"*60)
    
    # Calculate average metrics for rare classes
    avg_precision = np.mean([rare_metrics[c]['precision'] for c in rarest_classes])
    avg_recall = np.mean([rare_metrics[c]['recall'] for c in rarest_classes])
    avg_f1 = np.mean([rare_metrics[c]['f1'] for c in rarest_classes])
    results = {
        'overall_acc': overall_acc,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1
    }
    
    print(f"\nAverage across rare classes:")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall:    {avg_recall:.4f}")
    print(f"  F1-Score:  {avg_f1:.4f}")
    print("="*60)
    
    return results

# -----------------------------------------------------
# PCA Visualization
# -----------------------------------------------------
def pca_2d(X):
    """
    Perform PCA on an (N, d) data matrix and return its projection to 2 dimensions.
    """
    X_centered = X - np.mean(X, axis=0)
    cov = np.cov(X_centered, rowvar=False)

    # Eigen-decomposition 
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Get top eigenvectors
    idx = np.argsort(eigvals)[::-1][:2]
    top2 = eigvecs[:, idx]   # shape (d, 2)

    # Projection
    Z = X_centered @ top2

    return Z

def visualize_rare_class_embeddings(X_test, labels_test, encoder_weighted, encoder_unweighted, 
                                   rarest_classes, config, device, title='mnist'):
    mask = np.isin(labels_test, rarest_classes)
    X_rare = X_test[mask].to(device)
    labels_rare = labels_test[mask].to(device)
    
    print(f"\nVisualizing {len(X_rare)} samples from 5 rarest classes...")
    
    encoder_weighted.eval()
    encoder_unweighted.eval()
    
    with torch.no_grad():
        Z_weighted = encoder_weighted(X_rare).cpu().numpy()
        Z_unweighted = encoder_unweighted(X_rare).cpu().numpy()
    
    # Apply PCA to reduce to 2D
    Z_weighted_2d = pca_2d(Z_weighted)
    Z_unweighted_2d = pca_2d(Z_unweighted) 
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    class_colors = {c: colors[i % 10] for i, c in enumerate(rarest_classes)}
    
    # Plot weighted
    ax1 = axes[0]
    for c in rarest_classes:
        mask_c = (labels_rare == c).cpu().numpy()
        if mask_c.sum() > 0:
            ax1.scatter(Z_weighted_2d[mask_c, 0], Z_weighted_2d[mask_c, 1],
                       c=[class_colors[c]], label=f'Class {c} (n={mask_c.sum()})',
                       alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
            
            centroid = Z_weighted_2d[mask_c].mean(axis=0)
            ax1.scatter(centroid[0], centroid[1], c=[class_colors[c]], 
                       marker='X', s=300, edgecolors='black', linewidth=2)
    
    ax1.set_xlabel('PC1', fontsize=16, fontweight='bold')
    ax1.set_ylabel('PC2', fontsize=16, fontweight='bold')
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    intra_dist_w = []
    for c in rarest_classes:
        mask_c = (labels_rare == c).cpu().numpy()
        if mask_c.sum() > 1:
            points = Z_weighted_2d[mask_c]
            centroid = points.mean(axis=0)
            dists = np.linalg.norm(points - centroid, axis=1)
            intra_dist_w.append(dists.mean())
    avg_intra_w = np.mean(intra_dist_w) if intra_dist_w else 0
    
    # Plot unweighted
    ax2 = axes[1]
    for c in rarest_classes:
        mask_c = (labels_rare == c).cpu().numpy()
        if mask_c.sum() > 0:
            ax2.scatter(Z_unweighted_2d[mask_c, 0], Z_unweighted_2d[mask_c, 1],
                       c=[class_colors[c]], label=f'Class {c} (n={mask_c.sum()})',
                       alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
            
            centroid = Z_unweighted_2d[mask_c].mean(axis=0)
            ax2.scatter(centroid[0], centroid[1], c=[class_colors[c]], 
                       marker='X', s=300, edgecolors='black', linewidth=2)
    
    ax2.set_xlabel('PC1', fontsize=16, fontweight='bold')
    ax2.set_ylabel('PC2', fontsize=16, fontweight='bold')
    ax2.legend(loc='best', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    intra_dist_uw = []
    for c in rarest_classes:
        mask_c = (labels_rare == c).cpu().numpy()
        if mask_c.sum() > 1:
            points = Z_unweighted_2d[mask_c]
            centroid = points.mean(axis=0)
            dists = np.linalg.norm(points - centroid, axis=1)
            intra_dist_uw.append(dists.mean())
    avg_intra_uw = np.mean(intra_dist_uw) if intra_dist_uw else 0
    
    ax1.set_title(f'$U_N$: Rare Class Embeddings (Avg Intra-dist={avg_intra_w:.3f})', 
                  fontsize=16, fontweight='bold')
    ax2.set_title(f'$U_N^{{hl}}$: Rare Class Embeddings (Avg Intra-dist={avg_intra_uw:.3f})', 
                  fontsize=16, fontweight='bold')

    plt.tight_layout()
    encoder_suffix = "_cnn" 
    plt.savefig(f'results/{title}{encoder_suffix}_rare_class_embeddings.pdf', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("RARE CLASS EMBEDDING QUALITY")
    print("="*60)
    print(f"Weighted model - Avg intra-class distance:   {avg_intra_w:.4f}")
    print(f"Unweighted model - Avg intra-class distance: {avg_intra_uw:.4f}")
    
    if avg_intra_w < avg_intra_uw:
        improvement = ((avg_intra_uw - avg_intra_w) / avg_intra_uw) * 100
        print(f"\n✓ Weighted model has {improvement:.2f}% more compact clusters!")
    else:
        print(f"\n✗ Unweighted model has better compactness")
    print("="*60)
    
    return avg_intra_w, avg_intra_uw

# -----------------------------------------------------
# Visualization comparison
# -----------------------------------------------------
def visualize_comparison(results_weighted, results_unweighted, class_sizes, title='mnist'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    _, loss_w, test_loss_w, final_test_w, rarest = results_weighted
    _, loss_uw, test_loss_uw, final_test_uw, _ = results_unweighted
    
    x_pos = np.arange(len(rarest))
    width = 0.35
    
    losses_w = [final_test_w[c] for c in rarest]
    losses_uw = [final_test_uw[c] for c in rarest]
    
    ax.bar(x_pos - width/2, losses_w, width, label='$U_N$', 
            color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, losses_uw, width, label='$U_N^\\mathrm{hl}$',
            color='red', alpha=0.7)
    ax.set_xlabel('Rare Class ID', fontsize=16)
    ax.set_ylabel('Final Test Loss', fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{c}' for c in rarest])
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(f'results/{title}_comparison_results.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main(config):
    print("="*60)
    print(f"{config.dataset.upper()} Dataset")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Number of negative samples: {config.k_negatives}")
    print(f"Number of sub-sampled tuples: {config.m_incomplete}")
    
    # Load dataset 
    print("\n" + "-"*60)
    print(f"LOADING {config.dataset.upper()} DATASET")
    print("-"*60)
    X_train_img, labels_train, X_test_img, labels_test, class_sizes = load_imbalanced_dataset(config)

    # Train WEIGHTED
    print("\n" + "="*60)
    print("EXPERIMENT 1: WEIGHTED U-STATISTICS")
    print("="*60)
    start_time = time.time()
    results_weighted = train_contrastive_model(
        X_train_img, labels_train, X_test_img, labels_test,
        config, n_epochs=EPOCHS, use_weighting=True, avoid_collision=False
    )
    weighted_time = time.time() - start_time
    encoder_weighted, _, _, _, rarest_classes = results_weighted
    print(f"\nWeighted training completed in {weighted_time:.2f}s")

    # Train classifier weighted
    print("\n--- Training classifier on WEIGHTED encoder ---")
    classifier_weighted = train_linear_classifier(
        encoder_weighted, X_train_img, labels_train,
        X_test_img, labels_test, config, device, n_epochs=CLF_EPOCHS
    )

    # Evaluate classifier - weighted 
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS - WEIGHTED ENCODER")
    print("="*60)
    clf_result_weighted = evaluate_classifier_rare_classes(
        classifier_weighted, encoder_weighted, X_test_img, labels_test,
        rarest_classes, config, device
    )
    
    # Train UNWEIGHTED
    print("\n" + "="*60)
    print("EXPERIMENT 2: UNWEIGHTED U-STATISTICS")
    print("="*60)
    start_time = time.time()
    results_unweighted = train_contrastive_model(
        X_train_img, labels_train, X_test_img, labels_test,
        config, n_epochs=EPOCHS, use_weighting=False, avoid_collision=True
    )
    unweighted_time = time.time() - start_time
    encoder_unweighted, _, _, _, rarest_classes = results_unweighted
    print(f"\nUnweighted training completed in {unweighted_time:.2f}s")

    # Train classifier unweighted
    print("\n--- Training classifier on UNWEIGHTED encoder ---")
    classifier_unweighted = train_linear_classifier(
        encoder_unweighted, X_train_img, labels_train,
        X_test_img, labels_test, config, device, n_epochs=CLF_EPOCHS
    )

    # Evaluate classifier - unweighted
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS - UNWEIGHTED ENCODER")
    print("="*60)
    clf_result_unweighted = evaluate_classifier_rare_classes(
        classifier_unweighted, encoder_unweighted, X_test_img, labels_test,
        rarest_classes, config, device
    )

    # Visualize comparison
    print("\n" + "="*60)
    print("GENERATING COMPARISON PLOTS")
    print("="*60)
    encoder_suffix = f"_cnn"
    visualize_comparison(results_weighted, results_unweighted, class_sizes, title=f"{config.dataset}{encoder_suffix}")
    
    output_filename = f"results/clf_result_{config.dataset}_k{config.k_negatives}_rhomax{config.rho_max}_cnn.json"
    with open(output_filename, "w") as f:
        json.dump({'weighted': clf_result_weighted, 'unweighted': clf_result_unweighted}, f)
    
    print(f"\nResults saved to: {output_filename}")

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*60)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=False, default='cifar10', 
                       choices=['mnist', 'fashion_mnist', 'cifar10'], help='Real Dataset')
    parser.add_argument('--rho_max', type=float, required=False, default=0.5, 
                       help='Probability of dominant class')
    parser.add_argument('--k', type=int, required=False, default=5, 
                       help='Number of negative samples')
    parser.add_argument('--M', type=int, required=False, default=3000, 
                       help='Number of sub-sampled tuples')
    args = vars(parser.parse_args())

    # Run main
    config = ContrastiveConfig(
        k_negatives=args['k'], 
        dataset=args['dataset'], 
        m_incomplete=args['M'], 
        rho_max=args['rho_max']
    )
    main(config)
