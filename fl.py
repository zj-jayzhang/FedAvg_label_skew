#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import copy
import warnings
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from helpers.datasets import partition_data
from helpers.utils import average_weights, DatasetSplit, setup_seed, test
from models.nets import CNNCifar, CNNMnist
from models.resnet import resnet18
import numpy as np

warnings.filterwarnings('ignore')

criterion = nn.CrossEntropyLoss().cuda()


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.train_loader = DataLoader(DatasetSplit(dataset, idxs),
                                       batch_size=self.args.local_bs, shuffle=True, num_workers=4)

    def update_weights(self, model):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=0.9)
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.cuda(), labels.cuda()
                model.zero_grad()
                # ---------------------------------------
                output = model(images)
                loss = criterion(output, labels)
                # ---------------------------------------
                loss.backward()
                optimizer.step()
        return model.state_dict()


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epoch', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=100,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")

    parser.add_argument('--partition', default='dirichlet', type=str)
    parser.add_argument('--beta', default=0.5, type=float,
                        help=' If beta is set to a smaller value, '
                             'then the partition is more unbalanced')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')

    parser.add_argument('--seed', default=2021, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--loss', default="pretrain", type=str,
                        help='seed for initializing training.')
    parser.add_argument('--model', default="", type=str,
                        help='seed for initializing training.')

    args = parser.parse_args()
    return args


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


def get_model(args):
    if args.model == "mnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "fmnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "cifar10_res":
        global_model = resnet18(num_classes=10).cuda()
    elif args.model == "svhn_res":
        global_model = resnet18(num_classes=10).cuda()
    elif args.model == "cifar100_res":
        global_model = resnet18(num_classes=100).cuda()
    return global_model


def get_cls_num_list(traindata_cls_counts):
    cls_num_list = []
    for key, val in traindata_cls_counts.items():
        temp = [0] * 10
        for key_1, val_1 in val.items():
            temp[key_1] = val_1
        cls_num_list.append(temp)

    return cls_num_list


if __name__ == '__main__':

    args = args_parser()
    setup_seed(args.seed)

    train_dataset, test_dataset, user_groups, traindata_cls_counts = partition_data(
        args.dataset, args.partition, beta=args.beta, num_users=args.num_users)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                              shuffle=False, num_workers=4)
    # BUILD MODEL
    global_model = get_model(args)
    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    
    global_model.train()
    cls_num_list = get_cls_num_list(traindata_cls_counts)
    print(cls_num_list)
    tf_writer = SummaryWriter(log_dir='/logs')
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    # ===============================================
    for com in tqdm(range(args.epoch)):
        local_weights = []
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w = local_model.update_weights(copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(w))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        test_acc, test_loss = test(global_model, test_loader)
        tf_writer.add_scalar('test_acc', test_acc, com)
        tf_writer.add_scalar('test_loss', test_loss, com)
        is_best = test_acc > bst_acc
        bst_acc = max(bst_acc, test_acc)

        print("The {}-th communication round, test acc:{}, best_acc={}".format(com, test_acc, bst_acc))
        save_checkpoint(global_model.state_dict(), is_best)
    # ===============================================


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
