import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
@dataclass
class ContrastiveConfig:
    n_samples: int = 5000
    n_features: int = 64
    n_classes: int = 20
    rho_max: float = 0.5
    k_negatives: int = 5
    temperature: float = 0.5
    batch_size: int = 128
    m_incomplete: int = 3000  # sub-sampled tuples 
    test_size: int = 3000  # test samples
    n_epochs: int = 300

# -----------------------------------------------------
# Create highly imbalanced dataset
# -----------------------------------------------------
def create_imbalanced_dataset(config, seed=42):
    """
    Create dataset where one class dominates (40-50% of data)
    and remaining classes have decreasing frequencies
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    n = config.n_samples
    n_classes = config.n_classes
    
    # Generate class distribution: first class gets 50%
    # The rest distributed exponentially
    class_sizes = np.zeros(n_classes, dtype=int)
    class_sizes[0] = int(config.rho_max * n)  # Dominant class
    
    remaining = n - class_sizes[0]
    
    # Exponentially decreasing for remaining classes
    weights = np.exp(-0.1 * np.arange(1, n_classes))
    weights = weights / weights.sum()
    
    for i in range(1, n_classes):
        class_sizes[i] = int(weights[i-1] * remaining)
    
    # Adjust to exactly match n_samples
    diff = n - class_sizes.sum()
    class_sizes[-1] += diff
    
    print("\nClass distribution:")
    print(f"  Dominant class 0: {class_sizes[0]} ({100*class_sizes[0]/n:.1f}%)")
    print(f"  Rarest 5 classes: {class_sizes[-5:]}")
    print(f"  Total samples: {class_sizes.sum()}")
    
    # Generate data for each class
    X_list = []
    labels_list = []
    centers_list = []
    
    for c in range(n_classes):
        if class_sizes[c] > 0:
            # Generate cluster center
            center = np.random.randn(config.n_features) * 5
            centers_list.append(center)
            
            # Generate samples around center
            X_c = np.random.randn(class_sizes[c], config.n_features) + center
            X_list.append(X_c)
            labels_list.append(np.full(class_sizes[c], c))
    
    X = np.vstack(X_list)
    labels = np.concatenate(labels_list)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    labels = labels[indices]
    
    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    return X, labels, centers_list, class_sizes

# -----------------------------------------------------
# Simple encoder (PyTorch)
# -----------------------------------------------------
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=8):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.linear1.weight, std=0.01)
        nn.init.normal_(self.linear2.weight, std=0.01)
        nn.init.normal_(self.linear3.weight, std=0.01)

    def forward(self, x):
        z = F.relu(self.linear1(x))
        z = self.bn1(z)
        z = F.relu(self.linear2(z))
        z = self.bn2(z)
        z = self.linear3(z)
        return F.normalize(z, dim=1)

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
                weights[(r, False)] = 1 / (1 - self.tau_hat)
                weights[(r, True)] = (1 / (1 - self.tau_hat)) - threshold * (1/((1-self.tau_hat) * (self.class_counts[r] - 2)))
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
# Collate function for batching
# -----------------------------------------------------
def collate_tuples(batch):
    """
    Custom collate function to handle tuples
    batch: list of (anchor, positive, negatives, weight)
    """
    anchors = torch.stack([item[0] for item in batch])
    positives = torch.stack([item[1] for item in batch])
    negatives = torch.stack([item[2] for item in batch])  # (batch, k, d)
    weights = torch.stack([item[3] for item in batch])
    
    return anchors, positives, negatives, weights

# -----------------------------------------------------
# Batched contrastive loss
# -----------------------------------------------------
def batched_contrastive_loss(anchors, positives, negatives, temperature):
    """
    Compute contrastive loss for a batch
    anchors: (batch, d)
    positives: (batch, d)
    negatives: (batch, k, d)
    Returns: (batch,) loss per sample
    """
    batch_size = anchors.shape[0]
    
    # Compute positive similarities: (batch,)
    pos_sim = (anchors * positives).sum(dim=1) / temperature
    
    # Compute negative similarities: (batch, k)
    neg_sims = torch.bmm(negatives, anchors.unsqueeze(2)).squeeze(2) / temperature
    
    # Compute v: (batch, k)
    v = pos_sim.unsqueeze(1) - neg_sims
    
    # Compute loss per sample: (batch,)
    losses = torch.log1p(torch.exp(-v).sum(dim=1))
    
    return losses

# -----------------------------------------------------
# Collision check
# -----------------------------------------------------
def check_collision(tuple_indices, labels):
    anchor_idx, pos_idx, neg_indices = tuple_indices
    anchor_label = labels[anchor_idx]
    num_collisions = sum(int(labels[n] == anchor_label) for n in neg_indices)
    return num_collisions > 0, num_collisions

# -----------------------------------------------------
# Tuple sampler (kept for evaluation)
# -----------------------------------------------------
def sample_tuple(X, labels, k, class_r, avoid_collision=False):
    class_indices = torch.where(labels == class_r)[0]

    if len(class_indices) < 2:
        return None

    anchor_idx, pos_idx = random.sample(class_indices.tolist(), 2)

    if avoid_collision:
        # Only sample negatives from classes other than class_r
        available = [i for i in range(len(X)) 
                    if i not in (anchor_idx, pos_idx) and labels[i] != class_r]
    else:
        # Original behavior: sample from all data except anchor and positive
        available = [i for i in range(len(X)) if i not in (anchor_idx, pos_idx)]
    
    # Check if we have enough negatives
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
# Single tuple loss (for evaluation)
# -----------------------------------------------------
def contrastive_loss(anchor, positive, negatives, temperature):
    """
    anchor: (d,)
    positive: (d,)
    negatives: (k, d)
    """
    pos_sim = (anchor * positive).sum() / temperature
    neg_sims = (negatives @ anchor) / temperature
    v = pos_sim - neg_sims
    loss = torch.log1p(torch.exp(-v).sum())
    return loss

# -----------------------------------------------------
# Evaluate on specific classes
# -----------------------------------------------------
def evaluate_on_classes(X, labels, encoder, config, device, target_classes):
    """
    Evaluate test loss on specific classes
    """
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
# Training loop with DataLoader
# -----------------------------------------------------
def train_contrastive_model(X_train_np, labels_train_np, X_test_np, labels_test_np, 
                           config, n_epochs=100, use_weighting=True, avoid_collision=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    labels_train = torch.tensor(labels_train_np, dtype=torch.long, device=device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32, device=device)
    labels_test = torch.tensor(labels_test_np, dtype=torch.long, device=device)
    
    encoder = SimpleEncoder(config.n_features, output_dim=16).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, amsgrad=True)
    
    # Estimate collision probability
    tau_hat = estimate_collision_probability(labels_train, config.k_negatives)
    
    loss_history = []
    test_loss_history = []
    
    # Identify 5 rarest classes
    class_counts = np.bincount(labels_train_np)
    rarest_classes = np.argsort(class_counts)[:5].tolist()
    
    method_name = "WEIGHTED" if use_weighting else "UNWEIGHTED"
    print(f"\n{'='*60}")
    print(f"Training with {method_name} incomplete U-statistics (BATCHED)")
    print(f"M={config.m_incomplete} tuples per epoch")
    print(f"Batch size={config.batch_size}")
    print(f"Rarest classes: {rarest_classes}")
    print(f"{'='*60}")

    # Create dataset 
    dataset = ContrastiveTupleDataset(
        X_train, labels_train, 
        config.k_negatives, 
        config.m_incomplete,
        use_weighting=use_weighting,
        avoid_collision=avoid_collision
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_tuples,
        num_workers=0  # Use 0 for CUDA tensors
    )
    
    best_model, best_loss = None, np.inf
    for epoch in range(n_epochs):
        start = time.time()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for anchors, positives, negatives, weights in dataloader:
            optimizer.zero_grad()
            
            # Get embeddings
            weights = weights.to(device)
            z_anchors = encoder(anchors)
            z_positives = encoder(positives)
            
            # Reshape negatives: (batch, k, d) -> (batch*k, d)
            batch_size, k, d = negatives.shape
            z_negatives = encoder(negatives.view(batch_size * k, -1))
            z_negatives = z_negatives.view(batch_size, k, -1)
            
            # Compute batched loss: (batch,)
            losses = batched_contrastive_loss(
                z_anchors, z_positives, z_negatives, 
                config.temperature
            )
            
            # Apply importance weights
            weighted_losses = losses * weights
            
            # Average loss for this batch
            batch_loss = weighted_losses.sum() / weights.sum()
            
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)
        elapsed = time.time() - start
        if avg_epoch_loss < best_loss:
            best_model = encoder
            best_loss = avg_epoch_loss
            print(f' - Update model at epoch {epoch}, new best = {best_loss:.5f}')
        
        # Evaluate on test set (rarest classes) every 20 epochs
        if (epoch + 1) % 20 == 0:
            test_losses = evaluate_on_classes(X_test, labels_test, encoder, config, device, rarest_classes)
            avg_test_loss = np.mean([v for v in test_losses.values() if not np.isnan(v)])
            test_loss_history.append(avg_test_loss)
            
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_epoch_loss:.4f} | "
                  f"Test Loss (rare): {avg_test_loss:.4f} | Time: {elapsed:.2f}s")
    
    # Final evaluation on rarest classes
    final_test_losses = evaluate_on_classes(X_test, labels_test, encoder, config, device, rarest_classes)
    
    return best_model, loss_history, test_loss_history, final_test_losses, rarest_classes

# -----------------------------------------------------
# PCA Visualization of Rare Classes
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

def visualize_rare_class_embeddings(centers, rarest_classes, class_sizes, 
    encoder_weighted, encoder_unweighted, config, device, samples_per_class=100):
    """
    Visualize PCA-reduced embeddings for the 5 rarest classes
    comparing weighted vs unweighted models
    """
    # Generate data for each class
    X_list = []
    labels_list = []
    
    for c in rarest_classes:
        # Get the rare class center
        center = centers[c]
        
        # Generate samples around center
        X_c = np.random.randn(samples_per_class, config.n_features) + center
        X_list.append(X_c)
        labels_list.append(np.full(samples_per_class, c))
    X_rare = np.vstack(X_list)
    labels_rare = np.concatenate(labels_list)
    print(f"\nVisualizing {len(X_rare)} samples from 5 rarest classes...")
    
    # Convert to torch
    X_rare_torch = torch.tensor(X_rare, dtype=torch.float32, device=device)
    
    # Get embeddings from both models
    encoder_weighted.eval()
    encoder_unweighted.eval()
    
    with torch.no_grad():
        Z_weighted = encoder_weighted(X_rare_torch).cpu().numpy()
        Z_unweighted = encoder_unweighted(X_rare_torch).cpu().numpy()
    
    # Apply PCA to reduce to 2D
    Z_weighted_2d = pca_2d(Z_weighted)
    Z_unweighted_2d = pca_2d(Z_unweighted) 
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color map for classes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    class_colors = {c: colors[i % 10] for i, c in enumerate(rarest_classes)}
    
    # Plot weighted model embeddings
    ax1 = axes[0]
    for c in rarest_classes:
        mask_c = labels_rare == c
        if mask_c.sum() > 0:
            ax1.scatter(Z_weighted_2d[mask_c, 0], Z_weighted_2d[mask_c, 1],
                       c=[class_colors[c]], label=f'Id={c}',
                       alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
            
            # Compute and plot class centroid
            centroid = Z_weighted_2d[mask_c].mean(axis=0)
            ax1.scatter(centroid[0], centroid[1], c=[class_colors[c]], 
                       marker='X', s=300, edgecolors='black', linewidth=2)
    
    ax1.set_xlabel('PC1', fontsize=16, fontweight='bold')
    ax1.set_ylabel('PC2', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Compute intra-class compactness for weighted
    intra_dist_w = []
    for c in rarest_classes:
        mask_c = labels_rare == c
        if mask_c.sum() > 1:
            points = Z_weighted_2d[mask_c]
            centroid = points.mean(axis=0)
            dists = np.linalg.norm(points - centroid, axis=1)
            intra_dist_w.append(dists.mean())
    avg_intra_w = np.mean(intra_dist_w) if intra_dist_w else 0
    
    # Plot unweighted model embeddings
    ax2 = axes[1]
    for c in rarest_classes:
        mask_c = labels_rare == c
        if mask_c.sum() > 0:
            ax2.scatter(Z_unweighted_2d[mask_c, 0], Z_unweighted_2d[mask_c, 1],
                       c=[class_colors[c]], label=f'$n={class_sizes[c]}$',
                       alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
            
            # Compute and plot class centroid
            centroid = Z_unweighted_2d[mask_c].mean(axis=0)
            ax2.scatter(centroid[0], centroid[1], c=[class_colors[c]], 
                       marker='X', s=300, edgecolors='black', linewidth=2)
    
    ax2.set_xlabel('PC1', fontsize=16, fontweight='bold')
    ax2.set_ylabel('PC2', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Show legends
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='center left',
        title='Class Idx',
        bbox_to_anchor=(1.0, 0.67),  # slightly outside the right edge
        fontsize=16,
    )

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='center left',
        title='Train Size',
        bbox_to_anchor=(1.0, 0.33),  # slightly outside the right edge
        fontsize=16,
    )
    
    # Compute intra-class compactness for unweighted
    intra_dist_uw = []
    for c in rarest_classes:
        mask_c = labels_rare == c
        if mask_c.sum() > 1:
            points = Z_unweighted_2d[mask_c]
            centroid = points.mean(axis=0)
            dists = np.linalg.norm(points - centroid, axis=1)
            intra_dist_uw.append(dists.mean())
    avg_intra_uw = np.mean(intra_dist_uw) if intra_dist_uw else 0
    
    # Set title 
    hl = 'hl'
    ax1.set_title(f'$U_N$: Rare Class Embeddings (Avg Intra-dist={avg_intra_w:.3f})', 
                  fontsize=16, fontweight='bold')
    ax2.set_title(f'$U_N^{hl}$: Rare Class Embeddings (Avg Intra-dist={avg_intra_uw:.3f})', 
                  fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/gaussian_rare_class_embeddings.pdf', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print comparison
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
# Visualization (updated for comparison)
# -----------------------------------------------------
def visualize_comparison(results_weighted, results_unweighted, class_sizes):
    """
    Compare weighted vs unweighted approaches
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract results
    _, loss_w, test_loss_w, final_test_w, rarest = results_weighted
    _, loss_uw, test_loss_uw, final_test_uw, _ = results_unweighted
    
    # Final test loss per rare class - bar plot
    x_pos = np.arange(len(rarest))
    width = 0.35
    
    losses_w = [final_test_w[c] for c in rarest]
    losses_uw = [final_test_uw[c] for c in rarest]
    
    ax.bar(x_pos - width/2, losses_w, width, label='$U_N$', 
            color='tab:blue', alpha=0.7)
    ax.bar(x_pos + width/2, losses_uw, width, label='$U_N^\mathrm{hl}$',
            color='tab:red', alpha=0.7)
    ax.set_xlabel('Rare Class ID', fontsize=16)
    ax.set_ylabel('Final Test Loss', fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{c}' for c in rarest])
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.savefig('results/gaussian_comparison_results.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main(config):
    print("="*60)
    print("WEIGHTED vs UNWEIGHTED INCOMPLETE U-STATISTICS")
    print("With Batched DataLoader Processing")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Tuples per epoch: {config.m_incomplete}")
    
    # Generate imbalanced dataset
    print("\n" + "-"*60)
    print("GENERATING DATASET")
    print("-"*60)
    X, labels, centers, class_sizes = create_imbalanced_dataset(config)
    
    # Split into train and test
    n_train = len(X) - config.test_size
    indices = np.random.permutation(len(X))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train, labels_train = X[train_idx], labels[train_idx]
    X_test, labels_test = X[test_idx], labels[test_idx]
    
    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Number of batches per epoch: {config.m_incomplete // config.batch_size}")
    
    # Train with UNWEIGHTED approach
    print("\n" + "="*60)
    print("EXPERIMENT 2: UNWEIGHTED U-STATISTICS (with DataLoader)")
    print("="*60)
    start_time = time.time()
    results_unweighted = train_contrastive_model(
        X_train, labels_train, X_test, labels_test,
        config, n_epochs=config.n_epochs, use_weighting=False, avoid_collision=True
    )
    unweighted_time = time.time() - start_time
    encoder_unweighted, _, _, _, _ = results_unweighted
    
    print(f"\nUnweighted training completed in {unweighted_time:.2f}s")

    # Train with WEIGHTED approach
    print("\n" + "="*60)
    print("EXPERIMENT 1: WEIGHTED U-STATISTICS (with DataLoader)")
    print("="*60)
    start_time = time.time()
    results_weighted = train_contrastive_model(
        X_train, labels_train, X_test, labels_test,
        config, n_epochs=config.n_epochs, use_weighting=True, avoid_collision=False
    )
    weighted_time = time.time() - start_time
    encoder_weighted, _, _, _, rarest_classes = results_weighted
    
    print(f"\nWeighted training completed in {weighted_time:.2f}s")
    
    # Compare training times
    print("\n" + "="*60)
    print("TRAINING TIME COMPARISON")
    print("="*60)
    print(f"Weighted:   {weighted_time:.2f}s ({weighted_time/60:.2f} min)")
    print(f"Unweighted: {unweighted_time:.2f}s ({unweighted_time/60:.2f} min)")
    print(f"Speedup from batching: ~{max(weighted_time, unweighted_time) / min(weighted_time, unweighted_time):.2f}x")
    
    # Visualize comparison
    print("\n" + "="*60)
    print("GENERATING COMPARISON PLOTS")
    print("="*60)
    visualize_comparison(results_weighted, results_unweighted, class_sizes)
    
    # Visualize rare class embeddings with PCA
    print("\n" + "="*60)
    print("VISUALIZING RARE CLASS EMBEDDINGS (PCA)")
    print("="*60)
    visualize_rare_class_embeddings(
        centers, rarest_classes, class_sizes,
        encoder_weighted, encoder_unweighted,
        config, device
    )
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--rho_max', type=float, required=False, default=0.5, help='Probability of dominant class')
    parser.add_argument('--k', type=int, required=False, default=5, help='Number of negative samples')
    parser.add_argument('--M', type=int, required=False, default=3000, help='Number of sub-sampled tuples')
    parser.add_argument('--R', type=int, required=False, default=20, help='Number of classes')
    args = vars(parser.parse_args())

    # Run main
    config = ContrastiveConfig(
        rho_max=args['rho_max'],
        k_negatives=args['k'],
        m_incomplete=args['M'],
        n_classes=args['R']
    )
    main(config)
