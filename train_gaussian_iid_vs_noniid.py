import re
import os
import time
import tqdm
import torch
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from models import get_model, logistic_loss
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataloader.common import UnsupervisedDatasetWrapper 
from dataloader.common import apply_model_to_batch, save_json_dict
from dataloader.gaussian import generate_gaussian_clusters
from dataloader.gaussian import IndependentTuplesGaussianDataset

# Visualization configs
fontconfig = {
    'family' : 'normal',
    'size' : 15
}
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['text.usetex'] = True

# Constants for training
MAX_EPOCHS = 1000
BATCH_SIZE = 64
TRAIN_LOSS_THRESHOLD = 1e-4

# Constants for ablation study
DATASET_TO_INDIM = {'mnist' : 784, 'gaussian' : 128}

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

def get_dataloader(name='gaussian100', regime='subsample', k=3, batch_size=64, n_tuples=64000, n_test=400, n_test_tuples=160000):
    # Get dataset
    N = int(re.match(r"([a-zA-Z]+)(\d+)", name).group(2))
    train_data, test_data, centers = generate_gaussian_clusters(N, './data/gaussian/train') 

    # Wrap them in custom dataset definition
    # number of tuples to subset = num_batches * batch_size
    train_data = UnsupervisedDatasetWrapper(train_data, k, n_tuples, regime=regime).get_dataset()
    test_data  = UnsupervisedDatasetWrapper(test_data, k, n_test_tuples, regime=regime).get_dataset() 
    train_data_iid = IndependentTuplesGaussianDataset(centers, n_tuples)

    # Sample fewer data samples
    train_sampler = SubsetRandomSampler(
        indices=torch.arange(len(train_data))
    )
    test_sampler = SubsetRandomSampler(
        indices=torch.arange(len(test_data))
    )

    # Create custom dataloaders
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, shuffle=False)
    train_iid_dataloader = DataLoader(train_data_iid, sampler=None, batch_size=batch_size, shuffle=False)

    return train_dataloader, train_iid_dataloader, test_dataloader

def train(args, train_dataloader, test_dataloader):
    # Get number of train and test batches
    num_train_batches = len(train_dataloader)
    num_test_batches = len(test_dataloader)

    # In-case I am using Gaussian dataset
    match = re.match(r"([a-zA-Z]+)(\d+)", args['dataset'])
    args['dataset'] = match.group(1)
    
    # Load model
    model = get_model(in_dim=DATASET_TO_INDIM[args['dataset']], out_dim=args['d_dim'], hidden_dim=args['hidden_dim'], L=args['L'])
    model = model.to(model.device)

    # Optimization algorithm
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.0009, 
        amsgrad=True)

    # To be stored as final result
    final_average_train_loss, final_average_test_loss = 0, 0

    # Train model
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

        if final_average_train_loss <= TRAIN_LOSS_THRESHOLD:
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

    # Save result
    if args['outfile']:
        results = {
            'train_loss': final_average_train_loss,
            'test_loss' : final_average_test_loss,
            'gen_gap'   : final_average_test_loss - final_average_train_loss
        }
        save_experiment_result(args, results, args['outfile'])

    return final_average_train_loss, final_average_test_loss

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, required=False, default=1000, help='Number of training iterations')
    parser.add_argument('--dataset', type=str, required=False, default='gaussian', help='The dataset to experiment on')
    parser.add_argument('--d_dim', type=int, required=False, default=64, help='Output dimensionality')
    parser.add_argument('--hidden_dim', type=int, required=False, default=128, help='Hidden dimensionality')
    parser.add_argument('--k', type=int, required=False, default=3, help='Number of negative samples')
    parser.add_argument('--L', type=int, required=False, default=1, help='Number of layers')
    parser.add_argument('--batch_size', type=int, required=False, default=64, help='Batch size')
    parser.add_argument('--num_tuples', type=int, required=False, default=64000, help='Number of batches')
    parser.add_argument('--seeds', type=int, required=False, default=5, help='Number of repititions for experiments')
    parser.add_argument('--regime', type=str, required=False, default='subsample', help='Sampling regime - all tuples or only a subset')
    parser.add_argument('--outfile', type=str, required=False, default=None, help='Output file for experiment results')
    args = vars(parser.parse_args())

    # Initialize train and test loaders
    train_loader, train_iid_loader, test_loader = get_dataloader(name=args['dataset'], k=args['k'], 
            batch_size=args['batch_size'], regime=args['regime'], n_tuples=args['num_tuples'])

    # Start training - NonIID Case
    for seed in range(1, args['seeds']+1):
        print(f'Running seed #{seed}')
        args['setting'] = 'NonIID'
        train(args, train_loader, test_loader)

        # Start training - IID Case
        args['setting'] = 'IID'
        train(args, train_loader_iid, test_loader)

