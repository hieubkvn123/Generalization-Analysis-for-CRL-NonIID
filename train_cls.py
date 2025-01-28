import os
import time
import tqdm
import torch
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from dataloader.main import get_dataloader
from dataloader.common import apply_model_to_batch, save_json_dict, get_default_device
from models import get_model, logistic_loss

# Visualization configs
fontconfig = {
    'family' : 'normal',
    'size' : 15
}
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['text.usetex'] = True

# Constants for ablation study
DEVICE = get_default_device()
DATASET_TO_INDIM = {'mnist' : 784, 'cifar100': 3072, 'gaussian' : 128}
DATASET_TO_NUMCLS = {'mnist': 10, 'cifar100': 100}

# Saving experiment results
def save_experiment_result(args, results, outfile):
    # Create folder if not exists
    dirname = os.path.dirname(outfile)
    pathlib.Path(dirname).mkdir(exist_ok=True, parents=True)

    # Store both configs + results
    data = {**args, **results}

    # Create dataframe
    if os.path.exists(outfile):
        df = pd.read_csv(outfile)
        df.loc[-1] = data
    else:
        df = pd.DataFrame([data])

    # Save the file
    df.to_csv(outfile, index=False)
    return df

# Training function
def train_classifier(args, model, train_loader, val_loader, num_epochs=50, patience=3):
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()  # Set the model to training mode

    # For early stopping
    best_val_acc = 0.0
    epochs_since_improvement = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        print(f'[*] Epoch #[{epoch+1}/{num_epochs}]:')
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            for inputs, targets in train_loader:
                # Move data to the GPU if available
                inputs = inputs.view(-1, DATASET_TO_INDIM[args['dataset']])
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Track statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Update progress bar
                pbar.update(1)

        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total

        # Validation phase
        val_loss, val_accuracy = validate_cls(args, model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy:.2f}%, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Early Stopping Check
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(f"Early stopping triggered. Validation accuracy did not improve for {patience} epochs.")
            break

    return best_val_acc

# Validation function
def validate_cls(args, model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients during validation
        for inputs, targets in val_loader:
            inputs = inputs.view(-1, DATASET_TO_INDIM[args['dataset']])
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # Track accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    return avg_val_loss, val_accuracy

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, required=False, default=1000, help='Number of training iterations')
    parser.add_argument('--dataset', type=str, required=False, default='gaussian', help='The dataset to experiment on')
    parser.add_argument('--d_dim', type=int, required=False, default=64, help='Output dimensionality')
    parser.add_argument('--hidden_dim', type=int, required=False, default=128, help='Hidden dimensionality')
    parser.add_argument('--num_seeds', type=int, required=False, default=5, help='Number of experiments to repeat')
    parser.add_argument('--num_train', type=int, required=False, default=30000, help='Number of supervised data points to train on')
    parser.add_argument('--batch_size', type=int, required=False, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, required=False, default=0.0009, help='learning rate')
    parser.add_argument('--regime', type=str, required=False, default='subsample', help='Sampling regime - all tuples or only a subset')
    parser.add_argument('--train_loss_thresh', type=float, required=False, default=0.01, help='Target training loss to reach before calculating generalization gap')
    parser.add_argument('--outfile', type=str, required=False, default=None, help='Output file for experiment results')
    args = vars(parser.parse_args())

    # Get datasets
    default_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST('./data/mnist', train=True, download=True, transform=default_transform)
    test_data  = torchvision.datasets.MNIST('./data/mnist', train=False, download=True, transform=default_transform)
    train_data = torch.utils.data.Subset(train_data, list(range(args['num_train'])))

    # Get dataloaders
    train_dataloader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False)

    for j in range(args['num_seeds']):
        # Update the seeds
        print(f'[INFO] Training seed #{j+1}')
        args['seed'] = j + 1

        # Get base model
        cls = nn.Linear(DATASET_TO_INDIM[args['dataset']], DATASET_TO_NUMCLS[args['dataset']]).to(DEVICE)
        best_val_acc_sup = train_classifier(args, cls, train_dataloader, test_dataloader)

        # Save experiment result
        if args['outfile']:
            save_experiment_result(args, {'val_acc': best_val_acc_sup}, args['outfile']) 
