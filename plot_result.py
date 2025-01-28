import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

RESULT_FILE1 = 'results/ablation_study_iid_vs_noniid_mnist.csv'
RESULT_FILE2 = 'results/ablation_study_iid_vs_noniid_mnist_only_cls.csv'

def plot_errbar(ax, df, baseline, metric):
    tmp = df[df['regime'] == 'subsample']
    
    # Plot the baseline
    mean_independent = baseline[metric].mean()
    std_independent = baseline[metric].std()
    CI95 = 1.96 * std_independent / np.sqrt(len(baseline))
    ax.axhline(y=mean_independent, color='tab:red', label='Baseline ($i.i.d.$ case)')
    ax.axhline(y=mean_independent - CI95, color='tab:red', linestyle='-.', label='Baseline CI95 ($i.i.d.$ case)')
    ax.axhline(y=mean_independent + CI95, color='tab:red', linestyle='-.')

    # Plot the sub-sample regime
    mean = tmp.groupby('M')[metric].mean()
    std = tmp.groupby('M')[metric].std()
    ax.errorbar(mean.index, mean, yerr=std, label='Sub-sample (Non-$i.i.d.$ case)')

    return ax 

def plot_result1(output, figsize=(12, 7)):
    if not os.path.exists(RESULT_FILE1):
        print(f'File {RESULT_FILE1} does not exist...')
        return

    # Read the data file
    df = pd.read_csv(RESULT_FILE1)
    df_cls = pd.read_csv(RESULT_FILE2)
    baseline = df[df['regime'] == 'independent']

    # Initialize plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot errorbars for gen gap
    axes[0] = plot_errbar(axes[0], df, baseline, metric='gen_gap')
    axes[0].set_xlabel('Number of tuples used for training ($\mathrm{M}$)')
    axes[0].set_ylabel("Generalization gap")
    axes[0].legend()
    axes[0].grid()    

    # Plot errorbars for gen gap
    axes[1] = plot_errbar(axes[1], df, baseline, metric='test_acc')
    axes[1].set_xlabel('Number of tuples used for training ($\mathrm{M}$)')
    axes[1].set_ylabel("Classifier Accuracy")

    # Plot the range for cls without CRL
    mean_acc = df_cls['val_acc'].mean()
    std_acc = df_cls['val_acc'].std()
    CI95 = 1.96 * std_acc / np.sqrt(len(df_cls))
    axes[1].axhline(y=mean_acc, color='tab:orange', label='Baseline (without CRL)')
    axes[1].axhline(y=mean_acc - CI95, color='tab:orange', linestyle='-.', label='Baseline CI95 (without CRL)')
    axes[1].axhline(y=mean_acc + CI95, color='tab:orange', linestyle='-.')
    axes[1].legend()
    axes[1].grid()    

    # Plot 
    plt.tight_layout()
    plt.savefig(output)
    print(f'Output saved to {output}.')
         

if __name__ == '__main__':
    plot_result1(output='result.png')
    
