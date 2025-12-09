import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_tau(rho, k):
    """
    Compute tau = 1 - sum(rho_r * (1 - rho_r)^k)
    """
    return 1 - np.sum(rho * (1 - rho)**k)

def generate_random_distribution(R, alpha=0.5):
    """
    Generate a random discrete probability distribution with R categories.
    Uses Dirichlet distribution for uniform sampling over the probability simplex.
    """
    alpha_vec = np.ones(R) * alpha
    rho = np.random.dirichlet(alpha_vec)
    return rho

def estimate_probability(R, k, alpha, n_samples=10000):
    """
    Estimate P(k / (1-tau)^2 <= R) via Monte Carlo simulation
    
    Parameters:
    - R: number of categories in the distribution
    - k: exponent parameter
    - alpha: Dirichlet concentration parameter
    - n_samples: number of Monte Carlo samples
    
    Returns:
    - Estimated probability
    """
    count = 0
    
    for _ in range(n_samples):
        # Generate random probability distribution
        rho = generate_random_distribution(R, alpha=alpha)
        
        # Compute tau
        tau = compute_tau(rho, k)
        
        # Check if the condition holds
        if tau < 1:  # Ensure (1-tau) is positive
            ratio = k / (1 - tau)**2
            if ratio <= R:
                count += 1
    
    return count / n_samples

# Parameters
R = 100
alphas_plot1 = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
alphas_plot2 = np.arange(0.1, 1.6, 0.1)
k_values = list(range(1, 51))
n_samples = 10000
delta_values = [1, 2, 3]

# Store results
results_plot1 = {alpha: [] for alpha in alphas_plot1}
results_plot2 = {alpha: [] for alpha in alphas_plot2}
max_k_results = {10 ** (-e_delta): {} for e_delta in delta_values}

# Run Monte Carlo simulations for Plot 1
print("="*70)
print("PART 1: Computing probability curves for selected alphas")
print("="*70)
for alpha in alphas_plot1:
    print(f"\nProcessing alpha = {alpha}")
    for k in tqdm(k_values):
        prob = estimate_probability(R, k, alpha, n_samples)
        results_plot1[alpha].append(prob)

# Run Monte Carlo simulations for Plot 2
print("\n" + "="*70)
print("PART 2: Computing max k for all alphas and delta values")
print("="*70)
for alpha in alphas_plot2:
    print(f"\nProcessing alpha = {alpha:.2f}")
    for k in tqdm(k_values):
        prob = estimate_probability(R, k, alpha, n_samples)
        results_plot2[alpha].append(prob)
    
    # For each delta, find max k such that P(k/(1-tau)^2 <= R) >= 1-delta
    for e_delta in delta_values:
        delta = 10 ** (-e_delta)
        threshold = 1 - delta
        max_k = None
        
        # Find the largest k satisfying the threshold
        for i, k in enumerate(k_values):
            if results_plot2[alpha][i] >= threshold:
                max_k = k
        
        max_k_results[delta][alpha] = max_k if max_k is not None else 0

# Create side-by-side visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Probability curves for selected alphas
for alpha in alphas_plot1:
    ax1.plot(k_values, results_plot1[alpha], marker='o', markersize=3, 
             label=f'α = {alpha}', linewidth=2)

ax1.set_xlabel('k', fontsize=12)
ax1.set_ylabel('$P(k \\leq R(1-\\tau)^2)$', fontsize=12)
ax1.set_title(f'Probability vs k for Different α (R={R})', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(10, 51)
ax1.set_ylim(-0.05, 1.05)

# Plot 2: Max k as a function of alpha for different delta values
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']

for i, e_delta in enumerate(delta_values):
    delta = 10 ** (-e_delta)
    alpha_list = sorted(max_k_results[delta].keys())
    max_k_list = [max_k_results[delta][alpha] for alpha in alpha_list]
    
    ax2.plot(alpha_list, max_k_list, marker=markers[i], markersize=6, linewidth=2, 
             label=f'δ = 1e-{e_delta}', color=colors[i])

ax2.set_xlabel('α (Dirichlet concentration parameter)', fontsize=12)
ax2.set_ylabel('Max k where $P(k \\leq R(1-\\tau)^2) \\geq 1-\\delta$', fontsize=12)
ax2.set_title(f'Maximum k vs α for Different Confidence Levels (R={R})', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('combined_analysis.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS - PART 1")
print("="*70)
for alpha in alphas_plot1:
    print(f"\nα = {alpha}:")
    print(f"  Min probability: {min(results_plot1[alpha]):.4f}")
    print(f"  Max probability: {max(results_plot1[alpha]):.4f}")
    print(f"  Mean probability: {np.mean(results_plot1[alpha]):.4f}")

print("\n" + "="*70)
print("SUMMARY STATISTICS - PART 2")
print("="*70)
print(f"\n{'α':<10}", end='')
for delta in delta_values:
    print(f"δ={delta} (P≥{1-delta}){' '*3}", end='')
print("\n" + "-"*70)

for alpha in sorted(alphas_plot2):
    print(f"{alpha:<10.2f}", end='')
    for e_delta in delta_values:
        delta = 10 ** (-e_delta)
        print(f"{max_k_results[delta][alpha]:<20}", end='')
    print()
