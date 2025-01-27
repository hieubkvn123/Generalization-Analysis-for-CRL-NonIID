import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

RESULT_FILE1 = 'results/ablation_study_iid_vs_noniid_mnist_old.csv'

def plot_result1(output, figsize=(12, 7)):
    if not os.path.exists(RESULT_FILE1):
        print(f'File {RESULT_FILE1} does not exist...')
        return

    # Read the data file
    df = pd.read_csv(RESULT_FILE1)
    tmp = df[df['regime'] == 'subsample']
    baseline = df[df['regime'] == 'independent']['gen_gap']

    # Initialize plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the baseline
    ax.hlines(y=baseline, xmin=min(tmp['M']), xmax=max(df['M']), linestyle='-.', color='r', label='Baseline ($i.i.d.$ case)')

    # Axes labels
    ax.plot(tmp['M'], tmp['gen_gap'], label='Non-$i.i.d.$ case')
    ax.set_xlabel('Number of tuples used for training ($\mathrm{M}$)')
    ax.set_ylabel("Generalization gap")

    # Plot 
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output)
    print(f'Output saved to {output}.')
         

if __name__ == '__main__':
    plot_result1(output='result.png')
