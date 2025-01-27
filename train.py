import os
import time
import tqdm
import torch
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
def train_classifier(args, model, rep_model, train_loader, val_loader, num_epochs=50, patience=3):
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()  # Set the model to training mode
    rep_model.eval()

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
                inputs, targets = inputs.to(rep_model.device), targets.to(rep_model.device)
                inputs = inputs.view(-1, DATASET_TO_INDIM[args['dataset']])

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = rep_model(inputs)
                outputs = model(outputs)
                
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
        val_loss, val_accuracy = validate_cls(args, model, rep_model, val_loader, criterion)

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
def validate_cls(args, model,rep_model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    rep_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients during validation
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(rep_model.device), targets.to(rep_model.device)
            inputs = inputs.view(-1, DATASET_TO_INDIM[args['dataset']])

            # Forward pass
            outputs = rep_model(inputs)
            outputs = model(outputs)
            
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


def train(args):
    # Get dataset 
    num_batches = args['M'] // args['batch_size']
    train_dataloader, test_dataloader, train_dataloader_sup, test_dataloader_sup = get_dataloader(name=args['dataset'], k=args['k'], 
            batch_size=args['batch_size'], regime=args['regime'], num_batches=num_batches)
    num_train_batches = len(train_dataloader)
    num_test_batches = len(test_dataloader)

    # Load model
    model = get_model(in_dim=DATASET_TO_INDIM[args['dataset']], out_dim=args['d_dim'], hidden_dim=args['hidden_dim'], L=args['L'])
    model = model.to(model.device)
    print(f'Model is training on {model.device}')

    # Optimization algorithm
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.0009, 
        amsgrad=True)

    # To be stored as final result
    final_average_train_loss, final_average_test_loss = 0, 0

    # Train the representation model
    print('[INFO] Training the representation model...')
    model.train()
    for epoch in range(args['epochs']):
        print(f'[*] Epoch #[{epoch+1}/{args["epochs"]}]:')
        with tqdm.tqdm(total=len(train_dataloader)) as pbar:
            total_loss = 0.0
            for i, batch in enumerate(train_dataloader):
                # Calculate loss
                y1, y2, y3 = apply_model_to_batch(model, batch, device=model.device)
                loss = logistic_loss(y1, y2, y3)
                total_loss_batchwise = torch.sum(loss) 
                    
                # Back propagation
                total_loss_batchwise.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Update loss for this epoch
                total_loss += total_loss_batchwise.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'train_loss' : f'{total_loss_batchwise.item():.5f}',
                    'batch' : f'#[{i+1}/{num_train_batches}]'
                })
                pbar.update(1)
            time.sleep(0.1)
            final_average_train_loss = total_loss / (num_train_batches * args['batch_size'])
            print(f'\nAverage train loss : {final_average_train_loss:.4f}\n------\n')

        if final_average_train_loss <= args['train_loss_thresh']:
            print('[INFO] Train loss target reached, early stopping...')
            break

    # Evaluate the model
    model.eval()
    print('------\nEvaluation:')
    with tqdm.tqdm(total=len(test_dataloader)) as pbar:
        total_loss = 0.0
        for i, batch in enumerate(test_dataloader):
            # Calculate loss
            y1, y2, y3 = apply_model_to_batch(model, batch, device=model.device)
            loss = logistic_loss(y1, y2, y3)
            total_loss_batchwise = torch.sum(loss) 
            
            # Update loss
            total_loss += total_loss_batchwise.item()

            # Update progress bar
            pbar.set_postfix({
                'test_loss' : f'{total_loss_batchwise.item():.5f}',
                'batch' : f'#[{i+1}/{num_test_batches}]'
            })
            pbar.update(1)
        time.sleep(0.1)
        final_average_test_loss = total_loss / (num_test_batches * args['batch_size'])
        print(f'Average test loss : {final_average_test_loss}')

    # Train the classifier
    print('[INFO] Training the classifier...')
    cls = nn.Sequential(
        nn.Linear(args['d_dim'], DATASET_TO_NUMCLS[args['dataset']])
    )
    cls = cls.to(model.device)
    best_val_acc_sup = train_classifier(args, cls, model, train_dataloader_sup, test_dataloader_sup)
    

    # Save result
    if args['outfile']:
        results = {
            'train_loss': final_average_train_loss,
            'test_loss' : final_average_test_loss,
            'gen_gap'   : final_average_test_loss - final_average_train_loss,
            'test_acc'  : best_val_acc_sup
        }
        save_experiment_result(args, results, args['outfile'])

    # Save the model
    torch.save(model.state_dict(), f'weights/model_M{args["M"]}_{args["regime"]}.pth')

    return final_average_train_loss, final_average_test_loss

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, required=False, default=1000, help='Number of training iterations')
    parser.add_argument('--dataset', type=str, required=False, default='gaussian', help='The dataset to experiment on')
    parser.add_argument('--d_dim', type=int, required=False, default=64, help='Output dimensionality')
    parser.add_argument('--hidden_dim', type=int, required=False, default=128, help='Hidden dimensionality')
    parser.add_argument('--k', type=int, required=False, default=3, help='Number of negative samples')
    parser.add_argument('--L', type=int, required=False, default=2, help='Number of layers')
    parser.add_argument('--M', type=int, required=False, default=10000, help='Number of tuples to sub-sample')
    parser.add_argument('--num_seeds', type=int, required=False, default=5, help='Number of experiments to repeat')
    parser.add_argument('--batch_size', type=int, required=False, default=64, help='Batch size')
    parser.add_argument('--regime', type=str, required=False, default='subsample', help='Sampling regime - all tuples or only a subset')
    parser.add_argument('--train_loss_thresh', type=float, required=False, default=0.01, help='Target training loss to reach before calculating generalization gap')
    parser.add_argument('--outfile', type=str, required=False, default=None, help='Output file for experiment results')
    args = vars(parser.parse_args())

    for j in range(args['num_seeds']):
        print(f'[INFO] Training seed #{j+1}')
        args['seed'] = j + 1
        train(args)

