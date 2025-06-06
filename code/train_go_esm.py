import os
import os.path as osp
import numpy as np
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, fps, global_max_pool, global_mean_pool, radius

from datasets import *
from models import *
from utils import fmax

# Set CUDA visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training function
def train(epoch, dataloader, loss_fn):
    model.train()
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        y = torch.from_numpy(np.stack(data.y, axis=0)).to(device)
        out = model(data)
        loss = loss_fn(out.sigmoid(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

# Evaluation function
def test(dataloader):
    model.eval()
    probs, labels = [], []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            prob = model(data).sigmoid().detach().cpu().numpy()
            y = np.stack(data.y, axis=0)
        probs.append(prob)
        labels.append(y)
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return fmax(probs, labels)

# Argument parser
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='GOTrain')
    parser.add_argument('--data-dir', default='/usr/data/protein/Gearnet/GeneOntology', type=str)
    parser.add_argument('--level', default='cc', type=str, help='mf, bp, cc')
    parser.add_argument('--geometric-radius', default=4.0, type=float)
    parser.add_argument('--sequential-kernel-size', default=21, type=int)
    parser.add_argument('--kernel-channels', default=[24], type=int)
    parser.add_argument('--base-width', default=32, type=float)
    parser.add_argument('--channels', nargs='+', default=[256, 512, 1024, 2048], type=int)
    parser.add_argument('--num-epochs', default=500, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr-milestones', nargs='+', default=[300, 400], type=int)
    parser.add_argument('--lr-gamma', default=0.1, type=float)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--ckpt-path', default='./ckpt/go/cc.pt', type=str)
    return parser.parse_args()

# Main script
if __name__ == '__main__':
    args = parse_args()
    print(args)
    print("torch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)

    # Seed and device setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_dataset = GODataset_ESM(root=args.data_dir, random_seed=args.seed, level=args.level, split='train')
    valid_dataset = GODataset_ESM(root=args.data_dir, random_seed=args.seed, level=args.level, split='valid')
    test_dataset_95 = GODataset_ESM(root=args.data_dir, random_seed=args.seed, level=args.level, percent=95, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader_95 = DataLoader(test_dataset_95, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Build model
    model = Model_ESM(
        geometric_radii=[2*args.geometric_radius, 3*args.geometric_radius, 4*args.geometric_radius, 5*args.geometric_radius],
        sequential_kernel_size=args.sequential_kernel_size,
        kernel_channels=args.kernel_channels,
        channels=args.channels,
        base_width=args.base_width,
        num_classes=train_dataset.num_classes
    ).to(device)

    # Optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCELoss(weight=torch.as_tensor(train_dataset.weights).to(device))

    # Learning rate schedule
    lr_weights = []
    for i, milestone in enumerate(args.lr_milestones):
        power = np.power(args.lr_gamma, i)
        prev_milestone = args.lr_milestones[i - 1] if i > 0 else 0
        lr_weights += [power] * (milestone - prev_milestone)
    if args.lr_milestones[-1] < args.num_epochs:
        lr_weights += [np.power(args.lr_gamma, len(args.lr_milestones))] * (args.num_epochs + 1 - args.lr_milestones[-1])
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_weights[epoch])

    # Training loop
    best_valid = best_test_95 = best_95 = 0.0
    best_epoch = 0
    for epoch in range(args.num_epochs):
        train(epoch, train_loader, loss_fn)
        lr_scheduler.step()
        valid_fmax = test(valid_loader)
        test_95 = test(test_loader_95)
        print(f'Epoch: {epoch+1:03d}, Validation: {valid_fmax:.4f}, Test: {test_95:.4f}')
        if valid_fmax >= best_valid:
            best_valid = valid_fmax
            best_95 = test_95
            best_epoch = epoch
            checkpoint = model.state_dict()
        best_test_95 = max(test_95, best_test_95)

    print(args)
    print(f'Best: {best_epoch+1:03d}, Validation: {best_valid:.4f}, Test: {best_test_95:.4f}, Valided Test: {best_95:.4f}')
    
    # Save checkpoint (optional)
    # torch.save(checkpoint, osp.join(args.ckpt_path))
