import os

# Command template
EXP_TEMPLATE = 'python3 train_gaussian.py --dataset gaussian{N} --k {k} --outfile results/{fname}.csv --batch_size 64 --num_tuples {n_tuples}\n'

# Hyper-parameters
ks = list(range(42, 201, 2))
Ns = [500, 600, 700, 800]
exp_file = 'bash/additional_experiments.sh'
fname = 'results/additional_experiments.csv'

if __name__ == '__main__':
    with open(exp_file, 'w') as f:
        for N in Ns:
            for k in ks:
                cmd = EXP_TEMPLATE.format(N=N, k=k, fname=fname, n_tuples=N**2)
                f.write(cmd)

