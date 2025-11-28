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
from torch.utils.data import DataLoader
from dataloader.gaussian import generate_gaussian_clusters
from dataloader.main import get_dataloader, default_transform
from dataloader.common import apply_model_to_batch, save_json_dict
from dataloader.common import UnsupervisedDatasetWrapper
from dataloader.gaussian import GaussianTestDataset
from models import get_model, npair_loss

# Visualization configs
fontconfig = {
    'family' : 'normal',
    'size' : 15
}
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['text.usetex'] = True

# Constants for training
TRAIN_LOSS_THRESHOLD = 0.01

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


def train(args):
    # Check if GPU is available
    if torch.cuda.is_available():
        print('GPU is available')

    # Generate training dataset 
    class_probs = generate_class_probs(args['R'], args['rho_min'])
    train_data, configs = generate_gaussian_clusters(args['N'], class_probs=class_probs)
    train_data = UnsupervisedDatasetWrapper(train_data, k=args['k'], M=args['M'], regime=args['regime']).get_dataset()
    train_dataloader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=False)

    # Generate testing dataset
    test_data = GaussianTestDataset(configs['probs'], configs['mu'], configs['sigma'], args['num_test_per_class'], args['k'])
    test_dataloader = DataLoader(test_data, batch_size=args['batch_size'],shuffle=False)

    # Get data length
    num_train_batches = len(train_dataloader)
    num_test_batches = len(test_dataloader)

    # Load model
    model = get_model(in_dim=INPUT_DIM, out_dim=args['d_dim'], hidden_dim=args['hidden_dim'], L=args['L'])
    model = model.to(model.device)
    print(model)

    # Optimization algorithm
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.0009, 
        amsgrad=True)

    # To be stored as final result
    for epoch in range(1, args['epochs'] + 1):
        # Training phase
        model.train()

        # --- #
        print(f'[*] Epoch #[{epoch}/{args["epochs"]}]:')
        with tqdm.tqdm(total=num_train_batches) as pbar:
            total_loss = 0.0
            for i, batch in enumerate(train_dataloader):
                # Calculate loss
                weights = batch[3].to(model.device) # / torch.sum(torch.abs(batch[3]))
                y1, y2, y3 = apply_model_to_batch(model, batch, device=model.device)
                loss = npair_loss(y1, y2, y3) * weights
                mean_batchwise_loss = torch.sum(loss) / args['batch_size'] 
                    
                # Back propagation
                mean_batchwise_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Update progress bar
                pbar.set_postfix({
                    'mean_batchwise_loss' : f'{mean_batchwise_loss.item():.5f}',
                    'batch' : f'#[{i+1}/{num_train_batches}]'
                })
                pbar.update(1)
            time.sleep(0.1)

        if epoch % args['eval_every'] == 0:
            # Evaluation phase
            model.eval()

            # --- #
            print('------\nEvaluating population risk')
            population_risk, total_test_loss = 0.0, 0.0
            with tqdm.tqdm(total=num_test_batches) as pbar:
                for i, batch in enumerate(test_dataloader):
                    weights = batch[3].to(model.device)
                    y1, y2, y3 = apply_model_to_batch(model, batch, device=model.device)
                    loss = npair_loss(y1, y2, y3) * weights
                    total_test_loss += torch.sum(loss).item()
                    pbar.update(1)
                population_risk = (1/args['num_test_per_class']) * total_test_loss 
            print(f'Population risk estimated: {population_risk:.4f}')

            # --- #
            print('------\nEvaluating population risk')
            empirical_risk, total_train_loss = 0.0, 0.0
            with tqdm.tqdm(total=len(train_dataloader)) as pbar:
                for i, batch in enumerate(train_dataloader):
                    weights = batch[3].to(model.device)
                    y1, y2, y3 = apply_model_to_batch(model, batch, device=model.device)
                    loss = npair_loss(y1, y2, y3) * weights
                    total_train_loss += torch.sum(loss).item()
                    pbar.update(1)
                empirical_risk = (1/args['M']) * total_train_loss 
            print(f'Empirical risk estimated: {empirical_risk:.4f}')

            # --- #
            if empirical_risk <= TRAIN_LOSS_THRESHOLD:
                print('[INFO] Train loss target reached, early stopping...')
                break
            print('------\n')

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
    parser.add_argument('--d_dim', type=int, required=False, default=128, help='Output dimensionality')
    parser.add_argument('--hidden_dim', type=int, required=False, default=256, help='Hidden dimensionality')
    parser.add_argument('--k', type=int, required=False, default=3, help='Number of negative samples')
    parser.add_argument('--L', type=int, required=False, default=2, help='Number of layers')
    parser.add_argument('--N', type=int, required=False, default=100000, help='Number of labeled data points')
    parser.add_argument('--M', type=int, required=False, default=10000, help='Number of subsampled tuples')
    parser.add_argument('--R', type=int, required=False, default=1000, help='Number of classes')
    parser.add_argument('--rho_min', type=float, required=False, default=0.0001, help='Minimum class probabilities')

    parser.add_argument('--batch_size', type=int, required=False, default=64, help='Batch size')
    parser.add_argument('--num_batches', type=int, required=False, default=1000, help='Number of batches')
    parser.add_argument('--num_test_per_class', type=int, required=False, default=100, help='Number of testing data points per class')
    parser.add_argument('--eval_every', type=int, required=False, default=5, help='Evaluation interval')
    parser.add_argument('--regime', type=str, required=False, default='subsample', help='Sampling regime - all tuples or only a subset')
    parser.add_argument('--outfile', type=str, required=False, default=None, help='Output file for experiment results')
    args = vars(parser.parse_args())

    train(args)

