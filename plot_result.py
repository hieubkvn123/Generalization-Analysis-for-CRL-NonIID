import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Set fonts
plt.rcParams.update({'font.size': 18})

# Specify result files
RESULT_FILE1 = 'results/ablation_study_values_of_k_small_C_old.csv'
RESULT_FILE2 = 'results/ablation_study_values_of_C_small_k.csv'

def plot_result1(output, figsize=(12, 7)):
    if not os.path.exists(RESULT_FILE1) or not os.path.exists(RESULT_FILE2):
        print(f'File {RESULT_FILE1} does not exist...')
        return

    # Initialize figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].tick_params(axis='both', which='major', labelsize=15)
    axes[1].tick_params(axis='both', which='major', labelsize=15)

    # Plot result 1
    ## Read the data file
    df = pd.read_csv(RESULT_FILE1)
    mean = df.groupby('k')['Nmin'].mean()
    std = df.groupby('k')['Nmin'].std()
    axes[0].errorbar(mean.index, mean, yerr=std, linestyle='-.', marker='o')

    ## Axes labels
    axes[0].set_xlabel('Number of negative samples $k$')
    axes[0].set_ylabel('Sample complexity $\mathrm{N}_\epsilon$ ($\epsilon=0.5$)')
    axes[0].grid()

    # Plot result 2
    ## Read the data file
    df = pd.read_csv(RESULT_FILE2)
    mean = df.groupby('C')['Nmin'].mean()
    std = df.groupby('C')['Nmin'].std()
    axes[1].errorbar(mean.index, mean, yerr=std, linestyle='-.', marker='o')

    ## Axes labels
    axes[1].set_xlabel('Number of classes $|\mathcal{C}|$')
    axes[1].grid()

    # Plot 
    plt.tight_layout()
    plt.savefig(output)
    print(f'Output saved to {output}.')
         

if __name__ == '__main__':
    plot_result1(output='result.pdf')
