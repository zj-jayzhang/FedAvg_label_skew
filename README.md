# FedAvg_label_skew

"""
# mnist, 100 clients, 10 clients for each round
python3 fl.py --dataset=mnist --model=mnist_cnn --local_ep=5 --local_bs=64 --epoch=200 --num_users=100 --frac=0.1

# mnist, 5 clients, 5 clients for each round
python3 fl.py --dataset=mnist --model=mnist_cnn --local_ep=5 --local_bs=128 --epoch=200 --num_users=5 --frac=1

# fmnist, 100 clients, 10 clients for each round
python3 fl.py --dataset=fmnist --model=fmnist_cnn --local_ep=5 --local_bs=64 --epoch=200 --num_users=100 --frac=0.1

# fmnist, 5 clients, 5 clients for each round
python3 fl.py --dataset=fmnist --model=fmnist_cnn --local_ep=5 --local_bs=128 --epoch=200 --num_users=5 --frac=1

# cifar10, 100 clients, 10 clients for each round
python3 fl.py --dataset=cifar10 --model=cifar10_res --local_ep=5 --local_bs=64 --epoch=200 --num_users=100 --frac=0.1

# cifar10, 5 clients, 5 clients for each round
python3 fl.py --dataset=cifar10 --model=cifar10_res --local_ep=5 --local_bs=128 --epoch=200 --num_users=5 --frac=1

# svhn_cnn, 100 clients, 10 clients for each round
python3 fl.py --dataset=svhn --model=svhn_res --local_ep=5 --local_bs=64 --epoch=200 --num_users=100 --frac=0.1
# svhn_cnn, 5 clients, 5 clients for each round
python3 fl.py --dataset=svhn --model=svhn_res --local_ep=5 --local_bs=128 --epoch=200 --num_users=5 --frac=1


# cifar100, 100 clients, 10 clients for each round
python3 fl.py --dataset=cifar100 --model=cifar100_res --local_ep=5 --local_bs=64 --epoch=200 --num_users=100 --frac=0.1

# svhn_cnn, 5 clients, 5 clients for each round
python3 fl.py --dataset=cifar100 --model=cifar100_res --local_ep=5 --local_bs=128 --epoch=200 --num_users=5 --frac=1

"""
