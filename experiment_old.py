import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import time
import random
import matplotlib.pyplot as plt

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
@dataclass
class ContrastiveConfig:
    n_samples: int = 2000
    n_features: int = 64
    n_classes: int = 20
    k_negatives: int = 5
    temperature: float = 0.5
    m_incomplete: int = 2000  # tuples per epoch
    test_size: int = 500  # test samples

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
    weights = np.exp(-0.3 * np.arange(1, n_classes))
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
    def __init__(self, input_dim, output_dim=8):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.normal_(self.linear.weight, std=0.1)

    def forward(self, x):
        z = self.linear(x)
        return F.normalize(z, dim=1)

# -----------------------------------------------------
# Contrastive loss (same formula as before)
# -----------------------------------------------------
def contrastive_loss(anchor, positive, negatives, temperature):
    """
    anchor: (d,)
    positive: (d,)
    negatives: (k, d)
    """

    # similarities
    pos_sim = (anchor * positive).sum() / temperature
    neg_sims = (negatives @ anchor) / temperature

    v = pos_sim - neg_sims  # shape (k,)
    loss = torch.log1p(torch.exp(-v).sum())
    return loss

# -----------------------------------------------------
# Collision check
# -----------------------------------------------------
def check_collision(tuple_indices, labels):
    anchor_idx, pos_idx, neg_indices = tuple_indices
    anchor_label = labels[anchor_idx]
    num_collisions = sum(int(labels[n] == anchor_label) for n in neg_indices)
    return num_collisions > 0, num_collisions

# -----------------------------------------------------
# Tuple sampler
# -----------------------------------------------------
def sample_tuple(X, labels, k, class_r):
    class_indices = torch.where(labels == class_r)[0]

    if len(class_indices) < 2:
        return None

    anchor_idx, pos_idx = random.sample(class_indices.tolist(), 2)

    available = [i for i in range(len(X)) if i not in (anchor_idx, pos_idx)]
    neg_indices = random.sample(available, k)

    return (anchor_idx, pos_idx, neg_indices)

# -----------------------------------------------------
# Importance weights
# -----------------------------------------------------
def compute_weights(tuple_indices, labels, tau_hat):
    has_collision, _ = check_collision(tuple_indices, labels.cpu().numpy())
    if not has_collision:
        return 1.0 / (1.0 - tau_hat)
    else:
        return 0.5 * (1.0 / (1.0 - tau_hat))

# -----------------------------------------------------
# Estimate collision probability (same as original)
# -----------------------------------------------------
def estimate_collision_probability(labels, k):
    labels_np = labels.cpu().numpy()
    n = len(labels_np)
    classes, counts = np.unique(labels_np, return_counts=True)
    rho = counts / n
    tau = 1 - np.sum(rho * (1 - rho)**k)
    return float(tau)

# -----------------------------------------------------
# Incomplete U-statistics (WEIGHTED)
# -----------------------------------------------------
def incomplete_u_statistics_weighted(X, labels, encoder, config, device):
    n_classes = int(labels.max().item()) + 1

    class_counts = torch.bincount(labels, minlength=n_classes).float()
    class_probs = class_counts / class_counts.sum()

    tau_hat = estimate_collision_probability(labels, config.k_negatives)

    total_loss = 0.0
    m = config.m_incomplete

    for _ in range(m):
        r = int(torch.multinomial(class_probs, 1))

        t = sample_tuple(X, labels, config.k_negatives, r)
        if t is None:
            continue

        anchor_idx, pos_idx, neg_indices = t

        z_anchor = encoder(X[anchor_idx:anchor_idx+1])[0]
        z_pos = encoder(X[pos_idx:pos_idx+1])[0]
        z_negs = encoder(X[neg_indices])

        loss = contrastive_loss(z_anchor, z_pos, z_negs, config.temperature)
        weight = compute_weights(t, labels, tau_hat)

        total_loss += weight * loss

    return total_loss / m

# -----------------------------------------------------
# Incomplete U-statistics (UNWEIGHTED - baseline)
# -----------------------------------------------------
def incomplete_u_statistics_unweighted(X, labels, encoder, config, device):
    """Unweighted version - no importance sampling correction"""
    n_classes = int(labels.max().item()) + 1

    class_counts = torch.bincount(labels, minlength=n_classes).float()
    class_probs = class_counts / class_counts.sum()

    total_loss = 0.0
    m = config.m_incomplete

    for _ in range(m):
        r = int(torch.multinomial(class_probs, 1))

        t = sample_tuple(X, labels, config.k_negatives, r)
        if t is None:
            continue

        anchor_idx, pos_idx, neg_indices = t

        z_anchor = encoder(X[anchor_idx:anchor_idx+1])[0]
        z_pos = encoder(X[pos_idx:pos_idx+1])[0]
        z_negs = encoder(X[neg_indices])

        loss = contrastive_loss(z_anchor, z_pos, z_negs, config.temperature)
        
        # NO WEIGHTING - just accumulate raw loss
        total_loss += loss

    return total_loss / m

# -----------------------------------------------------
# Evaluate on specific classes
# -----------------------------------------------------
def evaluate_on_classes(X, labels, encoder, config, device, target_classes):
    """
    Evaluate test loss on specific classes
    """
    encoder.eval()
    
    losses_per_class = {c: [] for c in target_classes}
    n_samples_per_class = 50  # Sample 50 tuples per class for evaluation
    
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
    
    # Compute average loss per class
    avg_losses = {}
    for c in target_classes:
        if len(losses_per_class[c]) > 0:
            avg_losses[c] = np.mean(losses_per_class[c])
        else:
            avg_losses[c] = float('nan')
    
    encoder.train()
    return avg_losses

# -----------------------------------------------------
# Training loop (modified for comparison)
# -----------------------------------------------------
def train_contrastive_model(X_train_np, labels_train_np, X_test_np, labels_test_np, 
                           config, n_epochs=100, use_weighting=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    labels_train = torch.tensor(labels_train_np, dtype=torch.long, device=device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32, device=device)
    labels_test = torch.tensor(labels_test_np, dtype=torch.long, device=device)
    
    encoder = SimpleEncoder(config.n_features, output_dim=16).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    
    loss_history = []
    test_loss_history = []
    
    # Identify 5 rarest classes
    class_counts = np.bincount(labels_train_np)
    rarest_classes = np.argsort(class_counts)[:5].tolist()
    
    method_name = "WEIGHTED" if use_weighting else "UNWEIGHTED"
    print(f"\n{'='*60}")
    print(f"Training with {method_name} incomplete U-statistics")
    print(f"M={config.m_incomplete} tuples per epoch")
    print(f"Rarest classes: {rarest_classes}")
    print(f"{'='*60}")
    
    for epoch in range(n_epochs):
        start = time.time()
        optimizer.zero_grad()
        
        if use_weighting:
            loss = incomplete_u_statistics_weighted(X_train, labels_train, encoder, config, device)
        else:
            loss = incomplete_u_statistics_unweighted(X_train, labels_train, encoder, config, device)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(float(loss))
        elapsed = time.time() - start
        
        # Evaluate on test set (rarest classes) every 20 epochs
        if (epoch + 1) % 20 == 0:
            test_losses = evaluate_on_classes(X_test, labels_test, encoder, config, device, rarest_classes)
            avg_test_loss = np.mean([v for v in test_losses.values() if not np.isnan(v)])
            test_loss_history.append(avg_test_loss)
            
            print(f"Epoch {epoch+1:3d} | Train Loss: {loss.item():.4f} | "
                  f"Test Loss (rare): {avg_test_loss:.4f} | Time: {elapsed:.2f}s")
    
    # Final evaluation on rarest classes
    final_test_losses = evaluate_on_classes(X_test, labels_test, encoder, config, device, rarest_classes)
    
    return encoder, loss_history, test_loss_history, final_test_losses, rarest_classes

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
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
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
    
    plt.suptitle('Weighted vs Unweighted Incomplete U-Statistics Comparison',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('comparison_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
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
    print("Testing on Highly Imbalanced Dataset")
    print("="*60)
    
    config = ContrastiveConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Generate imbalanced dataset
    print("\nGenerating highly imbalanced dataset...")
    X, labels, class_sizes = create_imbalanced_dataset(config)
    
    # Split into train and test
    n_train = len(X) - config.test_size
    indices = np.random.permutation(len(X))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train, labels_train = X[train_idx], labels[train_idx]
    X_test, labels_test = X[test_idx], labels[test_idx]
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train with WEIGHTED approach
    print("\n" + "="*60)
    print("EXPERIMENT 1: WEIGHTED U-STATISTICS")
    print("="*60)
    results_weighted = train_contrastive_model(
        X_train, labels_train, X_test, labels_test,
        config, n_epochs=100, use_weighting=True
    )
    
    # Train with UNWEIGHTED approach
    print("\n" + "="*60)
    print("EXPERIMENT 2: UNWEIGHTED U-STATISTICS (Baseline)")
    print("="*60)
    results_unweighted = train_contrastive_model(
        X_train, labels_train, X_test, labels_test,
        config, n_epochs=100, use_weighting=False
    )
    

if __name__ == "__main__":
    main()
