import numpy as np
from dataloader.gaussian import generate_gaussian_clusters
from dataloader.common import UnsupervisedDatasetWrapper

# Some constants
k = 100
R = 1000
N = 100000
M = 10000
rho_min = 1e-3

def generate_class_probs(R, p_min):
    if p_min * R > 1:
        raise ValueError("Minimum probability too large — cannot sum to 1.")
    
    # remaining probability mass after assigning the minimum
    remaining = 1 - R * p_min
    
    # generate R random positive numbers that sum to 1
    rand = np.random.rand(R)
    rand /= rand.sum()
    
    # scale the random proportions into the remaining mass
    probs = p_min + remaining * rand
    return probs

if __name__ == '__main__':
    # Load Gaussian dataset
    class_probs = generate_class_probs(R, rho_min)
    train_data, test_data = generate_gaussian_clusters(N, class_probs=class_probs)
    train_data = UnsupervisedDatasetWrapper(train_data, k=k, M=M, regime='weighted_subsample').get_dataset()
    test_data  = UnsupervisedDatasetWrapper(test_data,  k=k, M=M, regime='weighted_subsample').get_dataset()

    # Check instances
    for i in range(50):
        anchor, positive, negatives, weight = next(iter(train_data))
        print(weight)
