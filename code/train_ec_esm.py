import os
import os.path as osp
import argparse
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from datasets import *
from models import *
from utils import fmax

# ------------------------- 训练函数 -------------------------
def train(epoch, dataloader, loss_fn, model, optimizer, device):
    model.train()
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        y = torch.from_numpy(np.stack(data.y, axis=0)).to(device)
        pred = model(data).sigmoid()
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

# ------------------------- 验证 / 测试函数 -------------------------
def evaluate(dataloader, model, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            prob = model(data).sigmoid().detach().cpu().numpy()
            y = np.stack(data.y, axis=0)
            probs.append(prob)
            labels.append(y)
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return fmax(probs, labels)

# ------------------------- 参数解析器 -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='ECTrain')
    parser.add_argument('--data-dir', default='/usr/data/protein/Gearnet/EnzymeCommission/', type=str)
    parser.add_argument('--geometric-radius', default=4.0, type=float)
    parser.add_argument('--sequential-kernel-size', default=21, type=int)
    parser.add_argument('--kernel-channels', nargs='+', default=[32], type=int)
    parser.add_argument('--base-width', default=16, type=float)
    parser.add_argument('--channels', nargs='+', default=[256, 512, 1024, 2048], type=int)
    parser.add_argument('--num-epochs', default=500, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr-milestones', nargs='+', default=[300, 400], type=int)
    parser.add_argument('--lr-gamma', default=0.1, type=float)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ckpt-path', default='./ckpt/ec/ec.pt', type=str)
    parser.add_argument('--save-model', action='store_true', help='Whether to save the best model')
    return parser.parse_args()

# ------------------------- 主函数 -------------------------
if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print(args)

    # 环境设置
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    train_dataset = ECDataset_ESM(root=args.data_dir, random_seed=args.seed, split='train')
    valid_dataset = ECDataset_ESM(root=args.data_dir, random_seed=args.seed, split='valid')
    test_dataset_95 = ECDataset_ESM(root=args.data_dir, random_seed=args.seed, percent=95, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader_95 = DataLoader(test_dataset_95, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # 模型构建
    geometric_radii = [k * args.geometric_radius for k in (2, 3, 4, 5)]
    model = Model_ESM(
        geometric_radii=geometric_radii,
        sequential_kernel_size=args.sequential_kernel_size,
        kernel_channels=args.kernel_channels,
        channels=args.channels,
        base_width=args.base_width,
        num_classes=train_dataset.num_classes
    ).to(device)

    # 优化器与损失
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    loss_fn = nn.BCELoss(weight=torch.tensor(train_dataset.weights).to(device))

    # 学习率调度器
    lr_weights = []
    for i, milestone in enumerate(args.lr_milestones):
        start = 0 if i == 0 else args.lr_milestones[i - 1]
        lr_weights += [args.lr_gamma ** i] * (milestone - start)
    if args.lr_milestones[-1] < args.num_epochs:
        lr_weights += [args.lr_gamma ** len(args.lr_milestones)] * (args.num_epochs - args.lr_milestones[-1] + 1)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_weights[epoch])

    # 训练过程
    best_valid = best_test_95 = best_95 = 0.0
    best_epoch = 0
    checkpoint = None

    for epoch in range(args.num_epochs):
        train(epoch, train_loader, loss_fn, model, optimizer, device)
        lr_scheduler.step()
        valid_fmax = evaluate(valid_loader, model, device)
        test_95 = evaluate(test_loader_95, model, device)
        print(f'Epoch: {epoch+1:03d}, Validation Fmax: {valid_fmax:.4f}, Test@95: {test_95:.4f}')

        if valid_fmax >= best_valid:
            best_valid = valid_fmax
            best_95 = test_95
            best_epoch = epoch
            checkpoint = model.state_dict()
        best_test_95 = max(best_test_95, test_95)

    print(f'\nBest Epoch: {best_epoch+1:03d} | Validation: {best_valid:.4f} | Test: {best_test_95:.4f} | Validated Test: {best_95:.4f}')

    # 保存模型
    if args.save_model and checkpoint is not None:
        os.makedirs(osp.dirname(args.ckpt_path), exist_ok=True)
        torch.save(checkpoint, args.ckpt_path)
        print(f'Model checkpoint saved to {args.ckpt_path}')
