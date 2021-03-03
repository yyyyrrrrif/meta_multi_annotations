# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
#import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pandas as pd
import sklearn.metrics as sm
import random
import numpy as np
from torch.nn.parameter import Parameter

from model import simple_net
from dataset import LabelMe_dataset

parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--iters', default=60000, type=int,
                    help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--meta_lr', '--meta_learning-rate', default=1e-3, type=float,
                    help='Meta learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--checkpoint', default='./ckpt/', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--matrix_dir', type=str, help='dir to save estimated matrix', default='matrix_meta')
parser.add_argument('--data_dir', type=str, help='data dir', default='../data')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
parser.set_defaults(augment=True)

#os.environ['CUD_DEVICE_ORDER'] = "1"
#ids = [1]
args = parser.parse_args()
use_cuda = True
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")


print()
print(args)

def adjust_learning_rate(base_lr, optimizer, epochs):
    lr = base_lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy, test_loss

def CE(train_loader, model, optimizer, epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0

    # best_acc = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}| Acc: {:.3f} ({}/{})'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), (train_loss / (batch_idx + 1)), 100. * correct / total,
                correct, total))

train_data = LabelMe_dataset('train', args.data_dir)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)

test_data = LabelMe_dataset('test', args.data_dir)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

model = simple_net(n_class=8)
model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def main():
    best_acc = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        adjust_learning_rate(0.01, optimizer, epoch)
        CE(train_loader, model, optimizer, epoch)
        test_acc, test_loss = test(model, test_loader)
        if test_acc >= best_acc:
            best_acc = test_acc

            if not os.path.exists(args.checkpoint):
                os.system('mkdir -p %s' % (args.checkpoint))
            torch.save(model.state_dict(), args.checkpoint + 'CE_labelme.pth')


if __name__ == '__main__':
    main()