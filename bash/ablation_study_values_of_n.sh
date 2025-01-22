# batch_size * num_batches = N**2
python train.py --dataset gaussian100 --k 5 --outfile results/ablation_study_values_of_n.csv --batch_size 10 --num_batches 1000
python train.py --dataset gaussian200 --k 5 --outfile results/ablation_study_values_of_n.csv --batch_size 10 --num_batches 4000
python train.py --dataset gaussian300 --k 5 --outfile results/ablation_study_values_of_n.csv --batch_size 10 --num_batches 9000
python train.py --dataset gaussian400 --k 5 --outfile results/ablation_study_values_of_n.csv --batch_size 10 --num_batches 16000
python train.py --dataset gaussian500 --k 5 --outfile results/ablation_study_values_of_n.csv --batch_size 10 --num_batches 25000
python train.py --dataset gaussian600 --k 5 --outfile results/ablation_study_values_of_n.csv --batch_size 10 --num_batches 36000
python train.py --dataset gaussian700 --k 5 --outfile results/ablation_study_values_of_n.csv --batch_size 10 --num_batches 49000
python train.py --dataset gaussian800 --k 5 --outfile results/ablation_study_values_of_n.csv --batch_size 10 --num_batches 64000
python train.py --dataset gaussian900 --k 5 --outfile results/ablation_study_values_of_n.csv --batch_size 10 --num_batches 81000
python train.py --dataset gaussian1000 --k 5 --outfile results/ablation_study_values_of_n.csv --batch_size 10 --num_batches 100000
