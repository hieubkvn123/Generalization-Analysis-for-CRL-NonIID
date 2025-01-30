import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Set fonts
plt.rcParams.update({'font.size': 18})
RESULT_FILE1 = 'results/ablation_study_iid_vs_noniid_mnist.csv'
RESULT_FILE2 = 'results/ablation_study_iid_vs_noniid_mnist_only_cls.csv'

def plot_errbar(ax, df, baseline1, baseline2, metric, label=True):
    tmp = df[df['regime'] == 'subsample']
    
    # Plot the baseline (iid case)
    mean_independent = baseline1[metric].mean()
    std_independent = baseline1[metric].std()
    CI95 = 1.96 * std_independent / np.sqrt(len(baseline1))
    ax.axhline(y=mean_independent, color='tab:red', label='$i.i.d.$ case' if label else None)
    ax.axhline(y=mean_independent - CI95, color='tab:red', linestyle='-.', label='$i.i.d.$ case ($95\%$ CI)' if label else None)
    ax.axhline(y=mean_independent + CI95, color='tab:red', linestyle='-.')

    # Plot the baseline (all-tuples)
    mean_independent = baseline2[metric].mean()
    std_independent = baseline2[metric].std()
    CI95 = 1.96 * std_independent / np.sqrt(len(baseline2))
    ax.axhline(y=mean_independent, color='tab:green', label='All-tuples' if label else None)
    ax.axhline(y=mean_independent - CI95, color='tab:green', linestyle='-.', label='All-tuples ($95\%$ CI)' if label else None)
    ax.axhline(y=mean_independent + CI95, color='tab:green', linestyle='-.')

    # Plot the sub-sample regime
    mean = tmp.groupby('M')[metric].mean()
    std = tmp.groupby('M')[metric].std()
    ax.errorbar(mean.index, mean, yerr=std, label='Non-$i.i.d.$ case' if label else None)

    return ax 

def plot_result1(output, figsize=(12, 7)):
    if not os.path.exists(RESULT_FILE1):
        print(f'File {RESULT_FILE1} does not exist...')
        return

    # Read the data file
    df = pd.read_csv(RESULT_FILE1)
    df_cls = pd.read_csv(RESULT_FILE2)
    baseline1 = df[df['regime'] == 'independent']
    baseline2 = df[df['regime'] == 'all']

    # Initialize plot
    fig, axes = plt.subplot_mosaic([['left', 'right']], layout='constrained', figsize=figsize)
    axes['left'].tick_params(axis='both', which='major', labelsize=12)
    axes['right'].tick_params(axis='both', which='major', labelsize=12)

    # Plot the range for cls without CRL
    mean_acc = df_cls['val_acc'].mean()
    std_acc = df_cls['val_acc'].std()
    CI95 = 1.96 * std_acc / np.sqrt(len(df_cls))
    axes['right'].axhline(y=mean_acc, color='tab:orange', label='No-CRL')
    axes['right'].axhline(y=mean_acc - CI95, color='tab:orange', linestyle='-.', label='No-CRL ($95\%$ CI)')
    axes['right'].axhline(y=mean_acc + CI95, color='tab:orange', linestyle='-.')
    axes['right'].grid()    

    # Plot errorbars for gen gap
    axes['left'] = plot_errbar(axes['left'], df, baseline1, baseline2, metric='gen_gap', label=False)
    axes['left'].set_xlabel('Number of tuples ($\mathrm{M}$)', fontsize=20)
    axes['left'].set_ylabel("Generalization gap", fontsize=20)
    axes['left'].grid()    

    # Plot errorbars for gen gap
    axes['right'] = plot_errbar(axes['right'], df, baseline1, baseline2, metric='test_acc')
    axes['right'].set_xlabel('Number of tuples ($\mathrm{M}$)', fontsize=20)
    axes['right'].set_ylabel("Classifier Accuracy", fontsize=20)

    # Plot 
    fig.legend(loc='outside upper left', mode='expand', ncols=4, fontsize=17)
    plt.savefig(output)
    print(f'Output saved to {output}.')
         

if __name__ == '__main__':
    plot_result1(output='result.pdf')
    
