#!/bin/bash
python3 train_gaussian.py --k 5 --outfile test.csv --batch_size 64 --C 10 --outfile results/ablation_study_values_of_k_small_C.csv
python3 train_gaussian.py --k 10 --outfile test.csv --batch_size 64 --C 10 --outfile results/ablation_study_values_of_k_small_C.csv
python3 train_gaussian.py --k 15 --outfile test.csv --batch_size 64 --C 10 --outfile results/ablation_study_values_of_k_small_C.csv
python3 train_gaussian.py --k 20 --outfile test.csv --batch_size 64 --C 10 --outfile results/ablation_study_values_of_k_small_C.csv
python3 train_gaussian.py --k 25 --outfile test.csv --batch_size 64 --C 10 --outfile results/ablation_study_values_of_k_small_C.csv
python3 train_gaussian.py --k 30 --outfile test.csv --batch_size 64 --C 10 --outfile results/ablation_study_values_of_k_small_C.csv
python3 train_gaussian.py --k 35 --outfile test.csv --batch_size 64 --C 10 --outfile results/ablation_study_values_of_k_small_C.csv
python3 train_gaussian.py --k 40 --outfile test.csv --batch_size 64 --C 10 --outfile results/ablation_study_values_of_k_small_C.csv
python3 train_gaussian.py --k 45 --outfile test.csv --batch_size 64 --C 10 --outfile results/ablation_study_values_of_k_small_C.csv
python3 train_gaussian.py --k 50 --outfile test.csv --batch_size 64 --C 10 --outfile results/ablation_study_values_of_k_small_C.csv
