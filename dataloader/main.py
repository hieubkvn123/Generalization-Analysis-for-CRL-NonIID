import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataloader.common import UnsupervisedDataset
from dataloader.common import get_default_device
from dataloader.gaussian import generate_gaussian_clusters

import os
import pathlib
import numpy as np
from collections import defaultdict

# Data loader
default_transform = transforms.Compose([transforms.ToTensor()])
def get_dataset(name='cifar100', k=3, n=1000):
    # Get raw dataset
    train_data, test_data = None, None
    if name == 'cifar100':
        train_data = torchvision.datasets.CIFAR100('./data/cifar', train=True, download=True, transform=default_transform)
        test_data  = torchvision.datasets.CIFAR100('./data/cifar', train=False, download=True, transform=default_transform)
    elif name == 'mnist':
        train_data = torchvision.datasets.MNIST('./data/mnist', train=True, download=True, transform=default_transform)
        test_data  = torchvision.datasets.MNIST('./data/mnist', train=False, download=True, transform=default_transform)
    elif name == 'gaussian':
        train_data, test_data = generate_gaussian_clusters('./data/gaussian') 

    # Wrap them in custom dataset definition
    train_data = UnsupervisedDataset(train_data, k=k, n=n)
    test_data  = UnsupervisedDataset(test_data, k=k, n=n//3)
    return train_data, test_data

def get_dataloader(name='cifar100', save_path='cache', save_loader=True, batch_size=64, num_batches=1000, sample_ratio=1.0, k=3):
    # Get loader directly if saved
    loader_dir = os.path.join(save_path, name, f'n{num_batches}-k{k}')
    train_path = os.path.join(loader_dir, 'train.pth')
    test_path  = os.path.join(loader_dir, 'test.pth')
    if os.path.exists(train_path) and os.path.exists(test_path):
        print('[INFO] Loaders already exist, loading from', loader_dir)
        train_dataloader = torch.load(train_path)
        test_dataloader = torch.load(test_path)
        return train_dataloader, test_dataloader

    # Get dataset
    train_data, test_data = get_dataset(name=name, k=k, n=num_batches*batch_size)

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

    # Save loader if requested
    pathlib.Path(loader_dir).mkdir(parents=True, exist_ok=True)
    if save_loader:
        torch.save(train_dataloader, os.path.join(loader_dir, 'train.pth'))
        torch.save(test_dataloader, os.path.join(loader_dir, 'test.pth'))
    return train_dataloader, test_dataloader
