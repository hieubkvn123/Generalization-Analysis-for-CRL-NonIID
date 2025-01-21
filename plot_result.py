import os
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def plot_simple_line(x, y, outfile, figsize=(15, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y)
    plt.savefig(outfile)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--outfile', type=str, required=True, help='Path to experiment result file')
    parser.add_argument('--resultfile', type=str, required=False, default='result.png', help='Path to visualization')
    parser.add_argument('--xaxis', type=str, required=False, default='k', help='Data at the x-axis')
    parser.add_argument('--yaxis', type=str, required=False, default='gen_gap', help='Data at the y-axis')
    args = vars(parser.parse_args())

    if os.path.exists(args['outfile']):
        df = pd.read_csv(args['outfile'])
        plot_simple_line(df[args['xaxis']], df[args['yaxis']], args['resultfile'])
    else:
        print(f'File args["outfile"] does not exist')
