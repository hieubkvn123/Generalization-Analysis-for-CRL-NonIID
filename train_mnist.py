import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import time
import random
import matplotlib.pyplot as plt
from tsne import tsne_2d
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
@dataclass
class ContrastiveConfig:
    n_samples: int = 5000  # Total samples to use from MNIST
    n_features: int = 28 * 28 * 1
    n_classes: int = 10
    k_negatives: int = 5
    temperature: float = 0.5
    batch_size: int = 64  # Reduced for image processing
    m_incomplete: int = 3000  # sub-sampled tuples 
    test_size: int = 4000  # test samples

# -----------------------------------------------------
# Feature Extractor 
# -----------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------
# Simple encoder
# -----------------------------------------------------
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, in_channels=1, hidden_dim=128, output_dim=64):
        super().__init__()
        # self.features = CIFARFeatureExtractor(in_channels=in_channels) #resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) 
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.linear1.weight, std=0.01)
        nn.init.normal_(self.linear2.weight, std=0.01)
        nn.init.normal_(self.linear3.weight, std=0.01)

    def forward(self, x):
        # Extract Features
        x = x.view(x.size(0), -1)

        # Pass through MLP
        z = F.relu(self.linear1(x))
        z = self.bn1(z)
        z = F.relu(self.linear2(z))
        z = self.bn2(z)
        z = self.linear3(z)
        return F.normalize(z, dim=1)

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x)

# -----------------------------------------------------
# Load and subsample MNIST
# -----------------------------------------------------
def load_mnist_imbalanced(config, seed=42):
    """
    Load MNIST and create highly imbalanced subset
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    # Load full MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
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
    
    # Generate class distribution: first class gets 45%, rest distributed exponentially
    class_sizes = np.zeros(n_classes, dtype=int)
    class_sizes[0] = int(0.45 * n)  # Dominant class: 45%
    
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
# Extract features using ResNet18
# -----------------------------------------------------
def extract_features(images, feature_extractor, device, batch_size=128):
    """
    Extract features from images using pretrained ResNet18
    """
    feature_extractor.eval()
    features = []
    
    dataset = torch.utils.data.TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            feats = feature_extractor(batch)
            features.append(feats.cpu())
    
    return torch.cat(features, dim=0)


# -----------------------------------------------------
# Contrastive Tuple Dataset
# -----------------------------------------------------
class ContrastiveTupleDataset(Dataset):
    def __init__(self, X, labels, k_negatives, num_tuples, tau_hat, 
                 use_weighting=True, avoid_collision=False):
        self.X = X
        self.labels = labels
        self.k = k_negatives
        self.num_tuples = num_tuples
        self.tau_hat = tau_hat
        self.use_weighting = use_weighting
        self.avoid_collision = avoid_collision
        
        # Precompute class information
        n_classes = int(labels.max().item()) + 1
        class_counts = torch.bincount(labels, minlength=n_classes).float()
        self.class_probs = class_counts / class_counts.sum()
        
        # Precompute class indices for faster sampling
        self.class_indices = {}
        for c in range(n_classes):
            self.class_indices[c] = torch.where(labels == c)[0].tolist()

        # Precompute negative indices
        self.class_to_neg_indices = {c:[] for c in range(n_classes)}
        for c in range(n_classes):
            self.class_to_neg_indices[c] = [i for i in range(len(self.X)) if self.labels[i] != c]
    
    def __len__(self):
        return self.num_tuples
    
    def sample_tuple(self, class_r):
        class_indices = self.class_indices[class_r]
        
        if len(class_indices) < 2:
            return None
        
        anchor_idx, pos_idx = random.sample(class_indices, 2)
        
        if self.avoid_collision:
            available = self.class_to_neg_indices[class_r]
        else:
            available = [i for i in range(len(self.X)) 
                        if i not in (anchor_idx, pos_idx)]
        
        if len(available) < self.k:
            return None
        
        neg_indices = random.sample(available, self.k)
        
        return (anchor_idx, pos_idx, neg_indices)
    
    def compute_weight(self, tuple_indices):
        if not self.use_weighting:
            return 1.0
        
        anchor_idx, pos_idx, neg_indices = tuple_indices
        anchor_label = self.labels[anchor_idx].item()
        
        has_collision = any(self.labels[n].item() == anchor_label for n in neg_indices)
        
        if not has_collision:
            return 1.0 / (1.0 - self.tau_hat)
        else:
            return 0.5 * (self.tau_hat / (1.0 - self.tau_hat))
    
    def __getitem__(self, idx):
        class_r = int(torch.multinomial(self.class_probs, 1).item())
        
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
# Tuple sampler for evaluation
# -----------------------------------------------------
def sample_tuple(X, labels, k, class_r, avoid_collision=False):
    class_indices = torch.where(labels == class_r)[0]

    if len(class_indices) < 2:
        return None

    anchor_idx, pos_idx = random.sample(class_indices.tolist(), 2)

    if avoid_collision:
        available = [i for i in range(len(X)) 
                    if i not in (anchor_idx, pos_idx) and labels[i] != class_r]
    else:
        available = [i for i in range(len(X)) if i not in (anchor_idx, pos_idx)]
    
    if len(available) < k:
        return None
    
    neg_indices = random.sample(available, k)

    return (anchor_idx, pos_idx, neg_indices)

# -----------------------------------------------------
# Estimate collision probability
# -----------------------------------------------------
def estimate_collision_probability(labels, k):
    labels_np = labels.cpu().numpy()
    n = len(labels_np)
    classes, counts = np.unique(labels_np, return_counts=True)
    rho = counts / n
    tau = 1 - np.sum(rho * (1 - rho)**k)
    return float(tau)

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
def evaluate_on_classes(X, labels, encoder, config, device, target_classes):
    encoder.eval()
    
    losses_per_class = {c: [] for c in target_classes}
    n_samples_per_class = 50
    
    with torch.no_grad():
        for c in target_classes:
            for _ in range(n_samples_per_class):
                t = sample_tuple(X, labels, config.k_negatives, c)
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
    
    encoder = SimpleEncoder(config.n_features, hidden_dim=128, output_dim=64).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, amsgrad=True)
    
    tau_hat = estimate_collision_probability(labels_train, config.k_negatives)
    
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
        tau_hat,
        use_weighting=use_weighting,
        avoid_collision=avoid_collision
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
            test_losses = evaluate_on_classes(X_test, labels_test, encoder, config, device, rarest_classes)
            avg_test_loss = np.mean([v for v in test_losses.values() if not np.isnan(v)])
            test_loss_history.append(avg_test_loss)
            
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_epoch_loss:.4f} | "
                  f"Test Loss (rare): {avg_test_loss:.4f} | Time: {elapsed:.2f}s")
    
    # Load best model
    encoder.load_state_dict(best_model)
    
    final_test_losses = evaluate_on_classes(X_test, labels_test, encoder, config, device, rarest_classes)
    
    return encoder, loss_history, test_loss_history, final_test_losses, rarest_classes

def train_linear_classifier(encoder, X_train, labels_train, X_test, labels_test, 
                           config, device, n_epochs=100):
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
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
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

def evaluate_classifier_rare_classes(classifier, encoder, X_test, labels_test, 
                                     rarest_classes, config, device):
    """
    Evaluate precision and recall for rare classes (without sklearn)
    """
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
    precision = {}
    recall = {}
    f1 = {}
    support = {}
    
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
    
    print(f"\nAverage across rare classes:")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall:    {avg_recall:.4f}")
    print(f"  F1-Score:  {avg_f1:.4f}")
    print("="*60)
    
    return rare_metrics, overall_acc

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
                                   rarest_classes, config, device):
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
    plt.savefig('results/mnist_rare_class_embeddings.pdf', dpi=150, bbox_inches='tight')
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
def visualize_comparison(results_weighted, results_unweighted, class_sizes):
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
    
    plt.savefig('results/mnist_comparison_results.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    print("="*60)
    print("WEIGHTED vs UNWEIGHTED INCOMPLETE U-STATISTICS")
    print("MNIST Dataset")
    print("="*60)
    
    config = ContrastiveConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load MNIST
    print("\n" + "-"*60)
    print("LOADING MNIST DATASET")
    print("-"*60)
    X_train_img, labels_train, X_test_img, labels_test, class_sizes = load_mnist_imbalanced(config)
    
    # Train UNWEIGHTED
    print("\n" + "="*60)
    print("EXPERIMENT 2: UNWEIGHTED U-STATISTICS")
    print("="*60)
    start_time = time.time()
    results_unweighted = train_contrastive_model(
        X_train_img, labels_train, X_test_img, labels_test,
        config, n_epochs=300, use_weighting=False, avoid_collision=True
    )
    unweighted_time = time.time() - start_time
    encoder_unweighted, _, _, _, rarest_classes = results_unweighted
    print(f"\nUnweighted training completed in {unweighted_time:.2f}s")

    # Train classifier unweighted
    print("\n--- Training classifier on UNWEIGHTED encoder ---")
    classifier_unweighted = train_linear_classifier(
        encoder_unweighted, X_train_img, labels_train,
        X_test_img, labels_test, config, device, n_epochs=100
    )

    # Evaluate classifier - unweighted
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS - UNWEIGHTED ENCODER")
    print("="*60)
    metrics_unweighted, acc_unweighted = evaluate_classifier_rare_classes(
        classifier_unweighted, encoder_unweighted, X_test_img, labels_test,
        rarest_classes, config, device
    )

    # Train WEIGHTED
    print("\n" + "="*60)
    print("EXPERIMENT 1: WEIGHTED U-STATISTICS")
    print("="*60)
    start_time = time.time()
    results_weighted = train_contrastive_model(
        X_train_img, labels_train, X_test_img, labels_test,
        config, n_epochs=300, use_weighting=True, avoid_collision=False
    )
    weighted_time = time.time() - start_time
    encoder_weighted, _, _, _, rarest_classes = results_weighted
    print(f"\nWeighted training completed in {weighted_time:.2f}s")

    # Train classifier weighted
    print("\n--- Training classifier on WEIGHTED encoder ---")
    classifier_weighted = train_linear_classifier(
        encoder_weighted, X_train_img, labels_train,
        X_test_img, labels_test, config, device, n_epochs=100
    )

    # Evaluate classifier - weighted 
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS - WEIGHTED ENCODER")
    print("="*60)
    metrics_weighted, acc_weighted = evaluate_classifier_rare_classes(
        classifier_weighted, encoder_weighted, X_test_img, labels_test,
        rarest_classes, config, device
    )

    # Visualize comparison
    print("\n" + "="*60)
    print("GENERATING COMPARISON PLOTS")
    print("="*60)
    visualize_comparison(results_weighted, results_unweighted, class_sizes)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*60)


if __name__ == '__main__':
    main()
