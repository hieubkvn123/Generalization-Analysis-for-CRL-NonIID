import warnings
import numpy as np

def tsne_2d(X, perplexity=30, n_iter=1000, learning_rate=200, momentum=0.8):
    """
    Reduce a high-dimensional matrix to 2D using t-SNE from scratch.
    
    Parameters:
    -----------
    X : numpy.ndarray, shape (B, d)
        Input matrix where B is number of samples, d is number of features
    perplexity : float, default=30
        Perplexity parameter (related to number of nearest neighbors)
    n_iter : int, default=1000
        Number of gradient descent iterations
    learning_rate : float, default=200
        Learning rate for gradient descent
    momentum : float, default=0.8
        Momentum for gradient descent
    
    Returns:
    --------
    Z : numpy.ndarray, shape (B, 2)
        2D embedding of the input data
    """
    warnings.filterwarnings('ignore')
    B, d = X.shape
    
    # Step 1: Compute pairwise squared Euclidean distances in high-dimensional space
    sum_X = np.sum(X**2, axis=1)
    D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * X @ X.T
    D = np.maximum(D, 0)  # Numerical stability
    
    # Step 2: Compute Gaussian kernel probabilities P_ij with binary search for sigma
    P = compute_joint_probabilities(D, perplexity)
    
    # Step 3: Initialize low-dimensional embedding with PCA or random
    Z = initialize_embedding(X, n_components=2)
    
    # Step 4: Perform gradient descent with momentum
    velocity = np.zeros_like(Z)
    
    for iteration in range(n_iter):
        # Compute pairwise squared Euclidean distances in low-dimensional space
        sum_Z = np.sum(Z**2, axis=1)
        D_low = sum_Z[:, np.newaxis] + sum_Z[np.newaxis, :] - 2 * Z @ Z.T
        D_low = np.maximum(D_low, 0)
        
        # Compute Q distribution (Student t-distribution with df=1)
        Q = 1 / (1 + D_low)
        np.fill_diagonal(Q, 0)
        Q = Q / np.sum(Q)
        Q = np.maximum(Q, 1e-12)  # Numerical stability
        
        # Compute gradient
        PQ_diff = P - Q
        grad = np.zeros_like(Z)
        
        for i in range(B):
            grad[i] = 4 * np.sum(
                (PQ_diff[i, :, np.newaxis] * (Z[i] - Z) * Q[i, :, np.newaxis]),
                axis=0
            )
        
        # Update with momentum
        velocity = momentum * velocity - learning_rate * grad
        Z = Z + velocity
        
        # Early exaggeration: multiply P by 4 for first 250 iterations
        if iteration == 250:
            P = P / 4
        
        # Print progress
        if (iteration + 1) % 100 == 0:
            C = np.sum(P * np.log(P / Q))  # KL divergence
            print(f"Iteration {iteration + 1}: KL divergence = {C:.4f}")
    
    return Z


def compute_joint_probabilities(D, perplexity, tol=1e-5):
    """
    Compute joint probability matrix P from distances using Gaussian kernel.
    Uses binary search to find appropriate sigma for each point.
    """
    B = D.shape[0]
    target_entropy = np.log(perplexity)
    P = np.zeros((B, B))
    
    for i in range(B):
        # Binary search for sigma_i
        beta_min, beta_max = -np.inf, np.inf
        beta = 1.0  # beta = 1/(2*sigma^2)
        
        for _ in range(50):  # Max iterations for binary search
            # Compute conditional probabilities P_j|i
            exp_D = np.exp(-D[i] * beta)
            exp_D[i] = 0  # Set P_i|i = 0
            sum_exp_D = np.sum(exp_D)
            
            if sum_exp_D == 0:
                P_i = np.zeros(B)
            else:
                P_i = exp_D / sum_exp_D
            
            # Compute Shannon entropy
            P_i_nonzero = P_i[P_i > 1e-12]
            if len(P_i_nonzero) > 0:
                entropy = -np.sum(P_i_nonzero * np.log2(P_i_nonzero))
            else:
                entropy = 0
            
            # Check if entropy matches target
            entropy_diff = entropy - target_entropy
            if np.abs(entropy_diff) < tol:
                break
            
            # Adjust beta
            if entropy_diff > 0:
                beta_min = beta
                beta = (beta + beta_max) / 2 if beta_max != np.inf else beta * 2
            else:
                beta_max = beta
                beta = (beta + beta_min) / 2 if beta_min != -np.inf else beta / 2
        
        P[i] = P_i
    
    # Symmetrize and normalize
    P = (P + P.T) / (2 * B)
    P = np.maximum(P, 1e-12)
    
    # Early exaggeration
    P = P * 4
    
    return P


def initialize_embedding(X, n_components=2):
    """
    Initialize low-dimensional embedding using PCA.
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov = X_centered.T @ X_centered / X.shape[0]
    
    # Compute eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Project onto top n_components
    Z = X_centered @ eigenvectors[:, :n_components]
    
    # Scale for numerical stability
    Z = Z * 1e-4
    
    return Z


# Example usage
if __name__ == "__main__":
    # Create sample data: 100 samples with 50 features
    np.random.seed(42)
    B, d = 100, 50
    X = np.random.randn(B, d)
    
    # Reduce to 2D
    print("Running t-SNE...")
    Z = tsne_2d(X, perplexity=30, n_iter=1000, learning_rate=200)
    
    print(f"\nInput shape: {X.shape}")
    print(f"Output shape: {Z.shape}")
    print(f"Output sample:\n{Z[:5]}")
    
    # Optional: Visualize the result
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 6))
        plt.scatter(Z[:, 0], Z[:, 1], alpha=0.6)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE 2D Projection (From Scratch)')
        plt.grid(True, alpha=0.3)
        plt.show()
    except ImportError:
        print("Matplotlib not available for visualization")
