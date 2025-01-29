import re
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataloader.common import UnsupervisedDatasetWrapper
from dataloader.common import get_default_device
from dataloader.gaussian import generate_gaussian_clusters

import os
import pickle
import pathlib
import numpy as np
from collections import defaultdict

# Constants
N_TEST_TUPLES = 1000

# Data loader
default_transform = transforms.Compose([transforms.ToTensor()])
def get_dataset(name='cifar100', k=3, n=1000, regime='subsample'):
    # Get raw dataset
    train_data, test_data = None, None
    if name == 'cifar100':
        train_data_raw = torchvision.datasets.CIFAR100('./data/cifar', train=True, download=True, transform=default_transform)
        test_data_raw  = torchvision.datasets.CIFAR100('./data/cifar', train=False, download=True, transform=default_transform)
    elif name == 'mnist':
        train_data_raw = torchvision.datasets.MNIST('./data/mnist', train=True, download=True, transform=default_transform)
        test_data_raw  = torchvision.datasets.MNIST('./data/mnist', train=False, download=True, transform=default_transform)
    elif name.startswith('gaussian'):
        match = re.match(r"([a-zA-Z]+)(\d+)", name)
        N = int(match.group(2))
        train_data_raw, test_data_raw = generate_gaussian_clusters(N*2, './data/gaussian') 

    # Wrap them in custom dataset definition
    train_data = UnsupervisedDatasetWrapper(train_data_raw, k, n, regime=regime, indices_file='ind_indices.txt').get_dataset()
    test_data  = UnsupervisedDatasetWrapper(test_data_raw,  k, N_TEST_TUPLES, regime='subsample').get_dataset()
    print(len(test_data))

    return train_data, test_data, train_data_raw, test_data_raw

def get_dataloader(name='cifar100', save_path='cache', regime='subsample', save_loader=True, batch_size=64, num_batches=1000, sample_ratio=1.0, k=3, indices=None):
    # Get loader directly if saved
    loader_dir = os.path.join(save_path, name, f'{name}-m{batch_size*num_batches}-k{k}-{regime}')
    train_path = os.path.join(loader_dir, 'train.pth')
    test_path  = os.path.join(loader_dir, 'test.pth')
    train_sup_path  = os.path.join(loader_dir, 'train_sup.pth')
    test_sup_path  = os.path.join(loader_dir, 'test_sup.pth')
    if os.path.exists(train_path) and os.path.exists(test_path):
        print('[INFO] Loaders already exist, loading from', loader_dir)
        train_dataloader = torch.load(train_path)
        test_dataloader = torch.load(test_path)
        train_dataloader_sup = torch.load(train_sup_path)
        test_dataloader_sup = torch.load(test_sup_path)
        return train_dataloader, test_dataloader, train_dataloader_sup, test_dataloader_sup

    # Get dataset
    train_data, test_data, train_data_raw, test_data_raw = get_dataset(name=name, k=k, n=num_batches*batch_size, regime=regime)


    # Sample fewer data samples
    train_sampler = SubsetRandomSampler(
        indices=torch.arange(int(len(train_data) * sample_ratio))
    )
    test_sampler = SubsetRandomSampler(
        indices=torch.arange(int(len(test_data) * sample_ratio))
    )

    # Create custom dataloaders
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, shuffle=False)
        
    
    # Create supervised dataloader
    train_dataloader_sup = DataLoader(train_data_raw, batch_size=batch_size, shuffle=True)
    test_dataloader_sup = DataLoader(test_data_raw, batch_size=batch_size, shuffle=False)

    # Save loader if requested
    pathlib.Path(loader_dir).mkdir(parents=True, exist_ok=True)
    if save_loader:
        torch.save(train_dataloader, os.path.join(loader_dir, 'train.pth'))
        torch.save(test_dataloader, os.path.join(loader_dir, 'test.pth'))
        torch.save(train_dataloader_sup, os.path.join(loader_dir, 'train_sup.pth'))
        torch.save(test_dataloader_sup, os.path.join(loader_dir, 'test_sup.pth'))
    return train_dataloader, test_dataloader, train_dataloader_sup, test_dataloader_sup
