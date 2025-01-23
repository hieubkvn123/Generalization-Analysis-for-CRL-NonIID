import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

RESULT_FILE1 = 'results/ablation_study_values_of_k.csv'

def plot_result1(output, figsize=(12, 7)):
    if not os.path.exists(RESULT_FILE1):
        print(f'File {RESULT_FILE1} does not exist...')
        return

    # Read the data file
    df = pd.read_csv(RESULT_FILE1)
    df['N'] = np.sqrt(df['num_tuples']).astype(int)

    # Initialize figure
    fig, ax = plt.subplots(figsize=figsize)
    for N in np.unique(df['N']):
        tmp = df[df['N'] == N]
        ax.plot(tmp['k'], tmp['gen_gap'], linestyle='-.', marker='o', label=f'$N=${N}')

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
