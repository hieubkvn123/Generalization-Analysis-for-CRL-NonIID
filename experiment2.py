import time
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
@dataclass
class ContrastiveConfig:
    n_samples: int = 1000
    n_features: int = 64
    n_classes: int = 100
    k_negatives: int = 5
    temperature: float = 0.5
    m_incomplete: int = 100  # tuples per epoch
    test_size: int = 1000  # test samples

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------
# Create highly imbalanced dataset
# -----------------------------------------------------
def create_imbalanced_dataset(config, seed=42):
    # Get class sizes
    dominant_class_size = int(0.8 * config.n_samples)
    class_sizes = [dominant_class_size] + [(config.n_samples - dominant_class_size) // (config.n_classes - 1)] * (config.n_classes - 1)
    print(class_sizes)
    
    # Generate data for each class
    X_list = []
    labels_list = []
    centers_list = []
    
    for c in range(config.n_classes):
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
    
    return X, labels, class_sizes, centers_list

# -----------------------------------------------------
# Simple encoder (PyTorch)
# -----------------------------------------------------
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=8):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim, bias=False)
        nn.init.normal_(self.fc1.weight, std=0.1)

        self.fc2  = nn.Linear(hidden_dim, output_dim, bias=False)
        nn.init.normal_(self.fc2.weight, std=0.1)


    def forward(self, x):
        z = F.relu(self.fc1(x))
        z = self.fc2(z)
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

    v = pos_sim - neg_sims + 1e-4  # shape (k,)
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
# Importance weights
# -----------------------------------------------------
def compute_weights(tuple_indices, labels, tau_hat, rho_hat, k):
    has_collision, _ = check_collision(tuple_indices, labels.cpu().numpy())
    if not has_collision:
        return 1.0 / (1.0 - tau_hat)
    else:
        return (3/(k * rho_hat)) * (tau_hat / (1.0 - tau_hat))

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
def incomplete_u_statistics_weighted(X, labels, class_probs, encoder, config, device):
    tau_hat = estimate_collision_probability(labels, config.k_negatives)
    total_loss, total_weight = 0.0, 0.0
    m = config.m_incomplete

    for _ in range(m):
        r = int(torch.multinomial(class_probs, 1))

        t = sample_tuple(X, labels, config.k_negatives, r, avoid_collision=False)
        if t is None:
            continue

        anchor_idx, pos_idx, neg_indices = t

        z_anchor = encoder(X[anchor_idx:anchor_idx+1])[0]
        z_pos = encoder(X[pos_idx:pos_idx+1])[0]
        z_negs = encoder(X[neg_indices])

        loss = contrastive_loss(z_anchor, z_pos, z_negs, config.temperature)
        weight = compute_weights(t, labels, tau_hat, class_probs[r], config.k_negatives)
        print(r, weight)

        total_loss += weight * loss
        total_weight += abs(weight)

    return total_loss / m # total_weight

# -----------------------------------------------------
# Incomplete U-statistics (UNWEIGHTED - baseline)
# -----------------------------------------------------
def incomplete_u_statistics_unweighted(X, labels, class_probs, encoder, config, device):
    """Unweighted version - no importance sampling correction"""
    total_loss = 0.0
    m = config.m_incomplete

    for _ in range(m):
        r = int(torch.multinomial(class_probs, 1))

        t = sample_tuple(X, labels, config.k_negatives, r, avoid_collision=False)
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

def evaluate_on_class(encoder, centers, class_probs, tuples_per_class=30, k_negatives=10, temperature=0.5, device=None):
    if device is None:
        device = next(encoder.parameters()).device
    n_classes = len(centers)
    feature_dim = centers[0].shape[0]

    # Convert centers to tensor
    centers_tensor = [torch.tensor(c, dtype=torch.float32, device=device) for c in centers]

    class_risks = np.zeros(n_classes)

    for r in range(n_classes):
        class_loss = 0.0

        for _ in range(tuples_per_class):
            # Anchor and positive from class r
            z_anchor = centers_tensor[r] + torch.randn(feature_dim, device=device)
            z_pos = centers_tensor[r] + torch.randn(feature_dim, device=device)

            # Normalize
            z_anchor = F.normalize(z_anchor.unsqueeze(0), dim=1)[0]
            z_pos = F.normalize(z_pos.unsqueeze(0), dim=1)[0]

            # k negatives from other classes
            negs = []
            while len(negs) < k_negatives:
                neg_class = np.random.choice(n_classes)
                if neg_class == r:
                    continue
                neg_sample = centers_tensor[neg_class] + torch.randn(feature_dim, device=device)
                neg_sample = F.normalize(neg_sample.unsqueeze(0), dim=1)[0]
                negs.append(neg_sample)
            z_negs = torch.stack(negs)

            # Encode
            with torch.no_grad():
                z_anchor_enc = encoder(z_anchor.unsqueeze(0))[0]
                z_pos_enc = encoder(z_pos.unsqueeze(0))[0]
                z_negs_enc = encoder(z_negs)

            # Contrastive loss
            pos_sim = (z_anchor_enc * z_pos_enc).sum() / temperature
            neg_sims = (z_negs_enc @ z_anchor_enc) / temperature
            v = pos_sim - neg_sims
            loss = torch.log1p(torch.exp(-v).sum())

            class_loss += float(loss)

        # Average loss for this class
        class_risks[r] = class_loss / tuples_per_class

    # Weighted population risk
    population_risk = np.sum(class_risks * class_probs.cpu().numpy())
    return population_risk, class_risks

# -----------------------------------------------------
# Training loop (modified for comparison)
# -----------------------------------------------------
def train_contrastive_model(X_train_np, labels_train_np, centers, 
                           config, n_epochs=100, use_weighting=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    labels_train = torch.tensor(labels_train_np, dtype=torch.long, device=device)

    class_counts = torch.bincount(labels_train, minlength=config.n_classes).float()
    class_probs = class_counts / class_counts.sum()

    set_seed()
    encoder = SimpleEncoder(config.n_features, output_dim=16).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    
    loss_history = []
    test_loss_history = []
    
    # Identify 5 rarest classes
    class_counts = np.bincount(labels_train_np)
    rarest_classes = np.argsort(class_counts)[:10].tolist()
    
    method_name = "WEIGHTED" if use_weighting else "UNWEIGHTED"
    print(f"\n{'='*60}")
    print(f"Training with {method_name} incomplete U-statistics")
    print(f"M={config.m_incomplete} tuples per epoch")
    print(f"Rarest classes: {rarest_classes}")
    print(f"{'='*60}")
    
    for epoch in range(n_epochs):
        encoder.train()
        start = time.time()
        optimizer.zero_grad()
        
        if use_weighting:
            loss = incomplete_u_statistics_weighted(X_train, labels_train, class_probs, encoder, config, device)
        else:
            loss = incomplete_u_statistics_unweighted(X_train, labels_train, class_probs, encoder, config, device)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(float(loss))
        elapsed = time.time() - start
        
        # Evaluate on test set (rarest classes) every 20 epochs
        if (epoch + 1) % 10 == 0:
            # test_losses = evaluate_on_classes(X_test, labels_test, encoder, config, device, rarest_classes)
            # avg_test_loss = np.mean([v for v in test_losses.values() if not np.isnan(v)])
            avg_test_loss, _ = evaluate_on_class(encoder, centers, class_probs) 
            test_loss_history.append(avg_test_loss)
            
            print(f"Epoch {epoch+1:3d} | Train Loss: {loss.item():.4f} | "
                  f"Test Loss (rare): {avg_test_loss:.4f} | Time: {elapsed:.2f}s")
    
    return encoder, loss_history, test_loss_history, rarest_classes

# -----------------------------------------------------
# Main
# -----------------------------------------------------
print("="*60)
print("WEIGHTED vs UNWEIGHTED INCOMPLETE U-STATISTICS")
print("Testing on Highly Imbalanced Dataset")
print("="*60)

config = ContrastiveConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Generate imbalanced dataset
print("\nGenerating highly imbalanced dataset...")
X_train, labels_train, class_sizes, centers = create_imbalanced_dataset(config)

# Train with WEIGHTED approach
print("\n" + "="*60)
print("EXPERIMENT 1: WEIGHTED U-STATISTICS")
print("="*60)
results_weighted = train_contrastive_model(
    X_train, labels_train, centers,
    config, n_epochs=200, use_weighting=True
)

# Train with UNWEIGHTED approach
print("\n" + "="*60)
print("EXPERIMENT 2: UNWEIGHTED U-STATISTICS (Baseline)")
print("="*60)
results_unweighted = train_contrastive_model(
    X_train, labels_train, centers, 
    config, n_epochs=200, use_weighting=False
)

def pca_numpy(X, n_components=2):
    """
    Pure NumPy PCA using SVD.
    X: (n_samples, n_features)
    Returns: (n_samples, n_components)
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T

def visualize_pca_generated_rare(
    encoder, rare_classes, centers, config, title, device
):
    """
    Generate 30 synthetic samples per rare class from their stored centers.
    Run them through the encoder.
    Reduce to 2D using pure NumPy PCA.
    Plot the clusters.
    """

    encoder.eval()

    samples = []
    labels = []

    # Generate fresh synthetic Gaussian samples
    for c in rare_classes:
        center = centers[c]
        Xc = np.random.randn(30, config.n_features) + center
        samples.append(Xc)
        labels.append(np.full(30, c))

    X_gen = np.vstack(samples)
    y_gen = np.concatenate(labels)

    # Encode to latent space
    with torch.no_grad():
        Z = encoder(
            torch.tensor(X_gen, dtype=torch.float32, device=device)
        ).cpu().numpy()

    # PCA via pure NumPy
    Z2 = pca_numpy(Z, n_components=2)

    # Plot the embeddings
    plt.figure(figsize=(7, 6))
    for c in rare_classes:
        mask = (y_gen == c)
        plt.scatter(
            Z2[mask, 0],
            Z2[mask, 1],
            s=45,
            alpha=0.75,
            label=f"class {c}"
        )

    plt.title(f"PCA of Generated Rare-Class Samples ({title})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f'result_{title}.png')

encoder_weighted, _, _, rare_classes = results_weighted
encoder_unweighted, _, _, _ = results_unweighted

print("\nVisualizing PCA on *generated* rare-class samples...")
visualize_pca_generated_rare(
    encoder_weighted,
    rare_classes,
    centers,
    config,
    title="WEIGHTED",
    device=device
)

visualize_pca_generated_rare(
    encoder_unweighted,
    rare_classes,
    centers,
    config,
    title="UNWEIGHTED",
    device=device
)

