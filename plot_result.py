import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

RESULT_FILE1 = 'results/ablation_study_iid_vs_noniid.csv'

def plot_result1(output, figsize=(12, 7)):
    if not os.path.exists(RESULT_FILE1):
        print(f'File {RESULT_FILE1} does not exist...')
        return

    # Read the data file
    df = pd.read_csv(RESULT_FILE1)
    df['N'] = np.sqrt(df['num_tuples']).astype(int)

    # Split dataset
    df_iid = df[df['setting'] == 'IID']
    df_niid = df[df['setting'] == 'NonIID']

    # Initialize figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the IID case
    mean  = df_iid.groupby('k').mean()['gen_gap']
    std   = df_iid.groupby('k').std()['gen_gap']
    xaxis = mean.index
    ax.errorbar(xaxis, mean, std, linestyle='-.', marker='o', label=f'IID setting')

    # Plot the Non-IID case
    mean  = df_niid.groupby('k').mean()['gen_gap']
    std   = df_niid.groupby('k').std()['gen_gap']
    xaxis = mean.index
    ax.errorbar(xaxis, mean, std, linestyle='-.', marker='o', label=f'Non-IID setting')

    # Axes labels
    ax.set_xlabel('Number of negative samples ($k$)')
    ax.set_ylabel("Generalization gap")

    # Plot 
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output)
    print(f'Output saved to {output}.')
         

if __name__ == '__main__':
    plot_result1(output='result.png')
