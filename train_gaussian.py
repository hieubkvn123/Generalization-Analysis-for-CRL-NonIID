import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import time
import random
import matplotlib.pyplot as plt

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
@dataclass
class ContrastiveConfig:
    n_samples: int = 2500
    n_features: int = 64
    n_classes: int = 20
    k_negatives: int = 5
    temperature: float = 0.5
    batch_size: int = 128
    m_incomplete: int = 2000  # sub-sampled tuples 
    test_size: int = 2000  # test samples

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
    
    for c in range(n_classes):
        if class_sizes[c] > 0:
            # Generate cluster center
            center = np.random.randn(config.n_features) * 5
            
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
    
    return X, labels, class_sizes

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
    """
    Dataset that generates tuples on-the-fly
    """
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
        """Sample a single tuple for class r"""
        class_indices = self.class_indices[class_r]
        
        if len(class_indices) < 2:
            return None
        
        anchor_idx, pos_idx = random.sample(class_indices, 2)
        
        if self.avoid_collision:
            # Only sample negatives from classes other than class_r
            available = self.class_to_neg_indices[class_r]
        else:
            # Sample from all data except anchor and positive
            available = [i for i in range(len(self.X)) 
                        if i not in (anchor_idx, pos_idx)]
        
        if len(available) < self.k:
            return None
        
        neg_indices = random.sample(available, self.k)
        
        return (anchor_idx, pos_idx, neg_indices)
    
    def compute_weight(self, tuple_indices):
        """Compute importance weight for a tuple"""
        if not self.use_weighting:
            return 1.0
        
        anchor_idx, pos_idx, neg_indices = tuple_indices
        anchor_label = self.labels[anchor_idx].item()
        
        # Check for collision
        has_collision = any(self.labels[n].item() == anchor_label for n in neg_indices)
        
        if not has_collision:
            return 1.0 / (1.0 - self.tau_hat)
        else:
            return 0.5 * (self.tau_hat / (1.0 - self.tau_hat))
    
    def __getitem__(self, idx):
        """
        Returns: (anchor, positive, negatives, weight)
        """
        # Sample a class proportional to its frequency
        class_r = int(torch.multinomial(self.class_probs, 1).item())
        
        # Keep trying until we get a valid tuple
        max_attempts = 10
        for _ in range(max_attempts):
            tuple_indices = self.sample_tuple(class_r)
            if tuple_indices is not None:
                break
        else:
            # Fallback: return dummy tuple with zero weight
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
        tau_hat,
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
        
        # Evaluate on test set (rarest classes) every 20 epochs
        if (epoch + 1) % 20 == 0:
            test_losses = evaluate_on_classes(X_test, labels_test, encoder, config, device, rarest_classes)
            avg_test_loss = np.mean([v for v in test_losses.values() if not np.isnan(v)])
            test_loss_history.append(avg_test_loss)
            
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_epoch_loss:.4f} | "
                  f"Test Loss (rare): {avg_test_loss:.4f} | Time: {elapsed:.2f}s")
    
    # Final evaluation on rarest classes
    final_test_losses = evaluate_on_classes(X_test, labels_test, encoder, config, device, rarest_classes)
    
    return encoder, loss_history, test_loss_history, final_test_losses, rarest_classes

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

def visualize_rare_class_embeddings(X_test, labels_test, encoder_weighted, encoder_unweighted, 
                                   rarest_classes, config, device):
    """
    Visualize PCA-reduced embeddings for the 5 rarest classes
    comparing weighted vs unweighted models
    """
    # Filter test data to only include rare classes
    mask = np.isin(labels_test, rarest_classes)
    X_rare = X_test[mask]
    labels_rare = labels_test[mask]
    
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
                       c=[class_colors[c]], label=f'Class {c} (n={mask_c.sum()})',
                       alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
            
            # Compute and plot class centroid
            centroid = Z_weighted_2d[mask_c].mean(axis=0)
            ax1.scatter(centroid[0], centroid[1], c=[class_colors[c]], 
                       marker='X', s=300, edgecolors='black', linewidth=2)
    
    ax1.set_xlabel('PC1 (Weighted)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PC2 (Weighted)', fontsize=12, fontweight='bold')
    ax1.set_title('Weighted U-Statistics: Rare Class Embeddings', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
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
                       c=[class_colors[c]], label=f'Class {c} (n={mask_c.sum()})',
                       alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
            
            # Compute and plot class centroid
            centroid = Z_unweighted_2d[mask_c].mean(axis=0)
            ax2.scatter(centroid[0], centroid[1], c=[class_colors[c]], 
                       marker='X', s=300, edgecolors='black', linewidth=2)
    
    ax2.set_xlabel('PC1 (Unweighted)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PC2 (Unweighted)', fontsize=12, fontweight='bold')
    ax2.set_title('Unweighted U-Statistics: Rare Class Embeddings', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
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
    
    # Add text with metrics
    metrics_text_w = f'Avg intra-class dist: {avg_intra_w:.3f}'
    metrics_text_uw = f'Avg intra-class dist: {avg_intra_uw:.3f}'
    
    ax1.text(0.02, 0.98, metrics_text_w, transform=ax1.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.text(0.02, 0.98, metrics_text_uw, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('PCA Visualization: Rare Class Embeddings Comparison\n' +
                 f'(X markers = class centroids)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('rare_class_embeddings.pdf', dpi=150, bbox_inches='tight')
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
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Extract results
    _, loss_w, test_loss_w, final_test_w, rarest = results_weighted
    _, loss_uw, test_loss_uw, final_test_uw, _ = results_unweighted
    
    # 1. Training loss comparison
    ax1 = fig.add_subplot(gs[0, :])
    epochs = np.arange(len(loss_w))
    ax1.plot(epochs, loss_w, 'b-', linewidth=2, label='Weighted', alpha=0.7)
    ax1.plot(epochs, loss_uw, 'r-', linewidth=2, label='Unweighted', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison (Batched Processing)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Test loss on rare classes over time
    ax2 = fig.add_subplot(gs[1, :])
    test_epochs = np.arange(20, len(loss_w)+1, 20)
    ax2.plot(test_epochs, test_loss_w, 'bo-', linewidth=2, markersize=8, 
             label='Weighted', alpha=0.7)
    ax2.plot(test_epochs, test_loss_uw, 'ro-', linewidth=2, markersize=8,
             label='Unweighted', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Loss (5 Rarest Classes)', fontsize=12)
    ax2.set_title('Test Performance on Rare Classes', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Final test loss per rare class - bar plot
    ax3 = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(len(rarest))
    width = 0.35
    
    losses_w = [final_test_w[c] for c in rarest]
    losses_uw = [final_test_uw[c] for c in rarest]
    
    ax3.bar(x_pos - width/2, losses_w, width, label='Weighted', 
            color='blue', alpha=0.7)
    ax3.bar(x_pos + width/2, losses_uw, width, label='Unweighted',
            color='red', alpha=0.7)
    ax3.set_xlabel('Class ID', fontsize=11)
    ax3.set_ylabel('Final Test Loss', fontsize=11)
    ax3.set_title('Loss per Rare Class', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'C{c}' for c in rarest])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Class distribution
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.bar(range(len(class_sizes)), class_sizes, color='steelblue', alpha=0.7)
    ax4.set_xlabel('Class ID', fontsize=11)
    ax4.set_ylabel('Number of Samples', fontsize=11)
    ax4.set_title('Class Distribution (Imbalanced)', fontsize=12, fontweight='bold')
    ax4.axhline(y=class_sizes[0], color='red', linestyle='--', 
                linewidth=2, label=f'Dominant: {class_sizes[0]}')
    for c in rarest:
        ax4.axvline(x=c, color='orange', linestyle=':', alpha=0.5)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Summary statistics
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    avg_final_w = np.mean(losses_w)
    avg_final_uw = np.mean(losses_uw)
    improvement = ((avg_final_uw - avg_final_w) / avg_final_uw) * 100
    
    summary_text = f"""
    SUMMARY STATISTICS
    {'='*35}
    
    Training:
      • Weighted final loss: {loss_w[-1]:.4f}
      • Unweighted final loss: {loss_uw[-1]:.4f}
    
    Test (5 Rarest Classes):
      • Weighted avg: {avg_final_w:.4f}
      • Unweighted avg: {avg_final_uw:.4f}
      • Improvement: {improvement:+.2f}%
    
    Dataset:
      • Dominant class size: {class_sizes[0]}
      • Rarest class size: {class_sizes[rarest].min()}
      • Imbalance ratio: {class_sizes[0]/class_sizes[rarest].min():.1f}x
    
    Winner: {'WEIGHTED ✓' if avg_final_w < avg_final_uw else 'UNWEIGHTED ✓'}
    """
    
    ax5.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Weighted vs Unweighted Incomplete U-Statistics (Batched)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('comparison_results_batched.pdf', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS (BATCHED)")
    print("="*60)
    print(f"\nTest Loss on 5 Rarest Classes:")
    print(f"  Weighted:   {avg_final_w:.4f}")
    print(f"  Unweighted: {avg_final_uw:.4f}")
    print(f"  Improvement: {improvement:+.2f}%")
    print(f"\nWinner: {'WEIGHTED ✓' if avg_final_w < avg_final_uw else 'UNWEIGHTED ✓'}")
    print("="*60)

# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    print("="*60)
    print("WEIGHTED vs UNWEIGHTED INCOMPLETE U-STATISTICS")
    print("With Batched DataLoader Processing")
    print("="*60)
    
    config = ContrastiveConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Tuples per epoch: {config.m_incomplete}")
    
    # Generate imbalanced dataset
    print("\n" + "-"*60)
    print("GENERATING DATASET")
    print("-"*60)
    X, labels, class_sizes = create_imbalanced_dataset(config)
    
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
        config, n_epochs=300, use_weighting=False, avoid_collision=True
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
        config, n_epochs=300, use_weighting=True
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
        X_test, labels_test,
        encoder_weighted, encoder_unweighted,
        rarest_classes, config, device
    )
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*60)
    print("\nGenerated files:")
    print("  • comparison_results_batched.pdf")
    print("  • rare_class_embeddings.pdf")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
