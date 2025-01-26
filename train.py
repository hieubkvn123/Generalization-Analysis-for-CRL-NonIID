import os
import time
import tqdm
import torch
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from dataloader.main import get_dataloader
from dataloader.common import apply_model_to_batch, save_json_dict
from models import get_model, logistic_loss

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
TRAIN_LOSS_THRESHOLD = 1e-3

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


def train(args):
    # Get dataset 
    num_batches = args['M'] // args['batch_size']
    train_dataloader, test_dataloader = get_dataloader(name=args['dataset'], k=args['k'], 
            batch_size=args['batch_size'], regime=args['regime'], num_batches=num_batches)
    num_train_batches = len(train_dataloader)
    num_test_batches = len(test_dataloader)

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
    parser.add_argument('--L', type=int, required=False, default=2, help='Number of layers')
    parser.add_argument('--M', type=int, required=False, default=10000, help='Number of tuples to sub-sample')
    parser.add_argument('--batch_size', type=int, required=False, default=64, help='Batch size')
    parser.add_argument('--regime', type=str, required=False, default='subsample', help='Sampling regime - all tuples or only a subset')
    parser.add_argument('--outfile', type=str, required=False, default=None, help='Output file for experiment results')
    args = vars(parser.parse_args())

    train(args)

