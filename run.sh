# MNIST 
python3 train_cnn.py  --dataset mnist --k 3 --N 10000 --M 10000 --model cnn
python3 train_cnn.py  --dataset mnist --k 5 --N 10000 --M 10000 --model cnn
python3 train_cnn.py  --dataset mnist --k 7 --N 10000 --M 10000 --model cnn

# Fashion MNIST
python3 train_cnn.py  --dataset fashion_mnist --k 3 --N 10000 --M 10000 --model cnn
python3 train_cnn.py  --dataset fashion_mnist --k 5 --N 10000 --M 10000 --model cnn
python3 train_cnn.py  --dataset fashion_mnist --k 7 --N 10000 --M 10000 --model cnn

# CIFAR10 + CIFAR100
python3 train_cnn.py --dataset cifar10 --k 3 --N 10000 --M 20000 --model cnn && python3 train_cnn.py  --dataset cifar10 --k 5 --N 10000 --M 20000 --model cnn && python3 train_cnn.py  --dataset cifar10 --k 7 --N 10000 --M 20000 --model cnn
python3 train_cnn.py --dataset cifar100 --k 3 --N 10000 --M 20000 --model cnn && python3 train_cnn.py  --dataset cifar100 --k 5 --N 10000 --M 20000 --model cnn && python3 train_cnn.py  --dataset cifar100 --k 7 --N 10000 --M 20000 --model cnn

