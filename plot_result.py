import os
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def check_unique_xaxis(x):
    return len(x) == len(set(x))

def plot_simple_line(x, y, outfile, figsize=(15, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y)

    # Adding labels and title
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Error bar plot')

    # Show grid for better readability
    plt.grid(True)

    # Legend + save
    plt.legend()
    plt.savefig(outfile, bbox_inches='tight')

def plot_errbar(x, y_mean, y_std, outfile, figsize=(15, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(x, y_mean, yerr=y_std, fmt='-o', capsize=3, color='b', label='Data with error bars', markersize=8)
    
    # Adding labels and title
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Error bar plot')

    # Show grid for better readability
    plt.grid(True)

    # Legend + save
    plt.legend()
    plt.savefig(outfile, bbox_inches='tight')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--outfile', type=str, required=True, help='Path to experiment result file')
    parser.add_argument('--resultfile', type=str, required=False, default='result.png', help='Path to visualization')
    parser.add_argument('--xaxis', type=str, required=False, default='k', help='Data at the x-axis')
    parser.add_argument('--yaxis', type=str, required=False, default='gen_gap', help='Data at the y-axis')
    args = vars(parser.parse_args())

    if os.path.exists(args['outfile']):
        df = pd.read_csv(args['outfile'])
        xaxis, yaxis = args['xaxis'], args['yaxis']

        # Plot simple line
        if check_unique_xaxis(df[xaxis].to_list()):
            plot_simple_line(df[xaxis], df[yaxis], args['resultfile'])
        else: # Plot error bar
            mean = df.groupby(xaxis).mean()[yaxis]
            std  = df.groupby(xaxis).std()[yaxis].fillna(0)
            x    = mean.index
            plot_errbar(x, mean, std, args['resultfile'])
    else:
        print(f'File args["outfile"] does not exist')
