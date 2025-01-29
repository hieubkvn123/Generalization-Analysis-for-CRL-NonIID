#!/bin/bash
python3 train_cls.py --dataset mnist --outfile results/ablation_study_iid_vs_noniid_mnist_only_cls.csv --batch_size 10 --num_train 300
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 100 --regime independent --train_loss_thresh 0.001
#python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --regime all --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 100 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 200 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 300 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 400 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 500 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 1000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 1500 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 2000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 2500 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 3000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 3500 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 4000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 5000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 5500 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 6000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 6500 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 7000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 7500 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 8000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 8500 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 9000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 9500 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 10000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 20000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 30000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 40000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 50000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 60000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 70000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 80000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 90000 --regime subsample --train_loss_thresh 0.001
python3 train.py --dataset mnist --k 1 --outfile results/ablation_study_iid_vs_noniid_mnist.csv --batch_size 10 --M 100000 --regime subsample --train_loss_thresh 0.001
