import os
import os.path as osp
import numpy as np
from typing import List

import torch
import torch.nn.functional as F
import torchvision
from torch_geometric.loader import DataLoader

import torch_geometric.transforms as T
from datasets import *
from models import *

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Config:
    # 通用配置
    data_dir = '/usr/data/protein/Gearnet/ProtFunct'
    geometric_radius = 4.0
    sequential_kernel_size = 21
    kernel_channels = [24]
    base_width = 32
    channels = [256, 512, 1024, 2048]
    num_epochs = 400
    batch_size = 8
    lr = 0.001
    weight_decay = 5e-4
    momentum = 0.9
    lr_milestones = [100, 300]
    lr_gamma = 0.1
    workers = 8
    seed = 0
    ckpt_path = './ckpt/func_2.pt'


def train(epoch, dataloader, model, optimizer, device):
    model.train()
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data).log_softmax(dim=-1), data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()


def test(dataloader, model, device):
    model.eval()
    correct = 0
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(dataloader.dataset)


if __name__ == '__main__':
    args = Config()

    print("torch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)
    print("Using config:", vars(args))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = FuncDataset_Struct(root=args.data_dir, random_seed=args.seed, split='training')
    valid_dataset = FuncDataset_Struct(root=args.data_dir, random_seed=args.seed, split='validation')
    test_dataset = FuncDataset_Struct(root=args.data_dir, random_seed=args.seed, split='testing')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = Model(
        geometric_radii=[
            2 * args.geometric_radius,
            3 * args.geometric_radius,
            4 * args.geometric_radius,
            5 * args.geometric_radius
        ],
        sequential_kernel_size=args.sequential_kernel_size,
        kernel_channels=args.kernel_channels,
        channels=args.channels,
        base_width=args.base_width,
        num_classes=train_dataset.num_classes
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        weight_decay=args.weight_decay,
        lr=args.lr,
        momentum=args.momentum
    )

    # 学习率调度器
    lr_weights = []
    for i, milestone in enumerate(args.lr_milestones):
        if i == 0:
            lr_weights += [np.power(args.lr_gamma, i)] * milestone
        else:
            lr_weights += [np.power(args.lr_gamma, i)] * (milestone - args.lr_milestones[i - 1])
    if args.lr_milestones[-1] < args.num_epochs:
        lr_weights += [np.power(args.lr_gamma, len(args.lr_milestones))] * (args.num_epochs + 1 - args.lr_milestones[-1])
    lambda_lr = lambda epoch: lr_weights[epoch]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    best_valid_acc = best_test_acc = best_acc = 0.0
    best_epoch = 0
    checkpoint = None

    for epoch in range(args.num_epochs):
        train(epoch, train_loader, model, optimizer, device)
        lr_scheduler.step()
        valid_acc = test(valid_loader, model, device)
        test_acc = test(test_loader, model, device)
        print(f'Epoch: {epoch + 1:03d}, Validation: {valid_acc:.4f}, Test: {test_acc:.4f}')

        if valid_acc >= best_valid_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_valid_acc = valid_acc
            checkpoint = model.state_dict()
        best_test_acc = max(test_acc, best_test_acc)

    print(f'Best Epoch: {best_epoch + 1:03d}, Validation: {best_valid_acc:.4f}, '
          f'Test: {best_test_acc:.4f}, Validated Test: {best_acc:.4f}')

    if args.ckpt_path:
        os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
        torch.save(checkpoint, args.ckpt_path)
