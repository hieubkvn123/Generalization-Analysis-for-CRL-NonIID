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
from dataloader.main import get_dataloader
from dataloader.gaussian import generate_gaussian_clusters
from dataloader.common import apply_model_to_batch, save_json_dict
from dataloader.common import UnsupervisedDatasetWrapper
from models import get_model, logistic_loss

# Visualization configs
fontconfig = {
    'family' : 'normal',
    'size' : 15
}
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['text.usetex'] = True

# Constants for training
BATCH_SIZE = 64
INPUT_DIM  = 128
MAX_EPOCHS = 1000
TRAIN_LOSS_THRESHOLD = 1e-3

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
    # Load Gaussian dataset
    class_probs = generate_class_probs(args['R'], args['rho_min'])
    train_data, test_data, configs = generate_gaussian_clusters(args['N'], class_probs=class_probs)
    train_data = UnsupervisedDatasetWrapper(train_data, k=args['k'], M=args['M'], regime=args['regime']).get_dataset()
    test_data  = UnsupervisedDatasetWrapper(test_data,  k=args['k'], M=args['M'], regime=args['regime']).get_dataset()

    # Get necessary things
    probs, mus, sigmas = configs['probs'], configs['mu'], configs['sigma']

    # Create data loader
    train_dataloader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=False)
    test_dataloader  = DataLoader(test_data,  batch_size=args['batch_size'], shuffle=False)

    # Get data length
    num_train_batches = len(train_dataloader)
    num_test_batches = len(test_dataloader)

    # Load model
    model = get_model(in_dim=INPUT_DIM, out_dim=args['d_dim'], hidden_dim=args['hidden_dim'], L=args['L'])
    model = model.to(model.device)

    # Optimization algorithm
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.0009, 
        amsgrad=True)

    # To be stored as final result
    final_average_train_loss, final_average_test_loss = 0, 0
    for epoch in range(1, args['epochs'] + 1):
        # Set to train mode
        model.train()
        print(f'[*] Epoch #[{epoch+1}/{args["epochs"]}]:')
        with tqdm.tqdm(total=len(train_dataloader)) as pbar:
            total_loss = 0.0
            for i, batch in enumerate(train_dataloader):
                # Calculate loss
                weights = batch[3] / torch.sum(torch.abs(batch[3]))
                y1, y2, y3 = apply_model_to_batch(model, batch, device=model.device)
                loss = logistic_loss(y1, y2, y3) * weights
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
        if epoch % 10 == 0:
            model.eval()
            print('------\nEvaluation:')
            for i, (prob, mu, sigma) in enumerate(zip(configs['probs'], configs['mu'], configs['sigma'])):
                # Evaluate the model on some tuple corresponding to current class 
                neg_classes = np.delete(np.arange(len(probs)), idx)
                neg_probs = rhos[neg_classes]
                neg_probs = neg_probs / neg_probs.sum() # Re-normalize

                # Estimate the expected risk of this class

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
    parser.add_argument('--d_dim', type=int, required=False, default=64, help='Output dimensionality')
    parser.add_argument('--hidden_dim', type=int, required=False, default=128, help='Hidden dimensionality')
    parser.add_argument('--k', type=int, required=False, default=3, help='Number of negative samples')
    parser.add_argument('--L', type=int, required=False, default=2, help='Number of layers')
    parser.add_argument('--N', type=int, required=False, default=100000, help='Number of labeled data points')
    parser.add_argument('--M', type=int, required=False, default=100000, help='Number of subsampled tuples')
    parser.add_argument('--R', type=int, required=False, default=1000, help='Number of classes')
    parser.add_argument('--rho_min', type=float, required=False, default=0.0001, help='Minimum class probabilities')

    parser.add_argument('--batch_size', type=int, required=False, default=64, help='Batch size')
    parser.add_argument('--num_batches', type=int, required=False, default=1000, help='Number of batches')
    parser.add_argument('--regime', type=str, required=False, default='subsample', help='Sampling regime - all tuples or only a subset')
    parser.add_argument('--outfile', type=str, required=False, default=None, help='Output file for experiment results')
    args = vars(parser.parse_args())

    train(args)

