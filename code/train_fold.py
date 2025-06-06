import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch_geometric.transforms as T
from functools import partial
from typing import List
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, fps, global_max_pool, global_mean_pool, radius

from datasets import *
from models import *

# 设置使用的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ----------------------------
# 参数配置类
# ----------------------------
class Config:
    data_dir = '/usr/data/protein/Gearnet/HomologyTAPE'
    geometric_radius = 4.0
    sequential_kernel_size = 5
    kernel_channels = [24]
    base_width = 64
    channels = [256, 512, 1024, 2048]
    num_epochs = 400
    batch_size = 4
    lr = 0.001
    weight_decay = 5e-4
    momentum = 0.9
    lr_milestones = [100, 300]
    lr_gamma = 0.1
    workers = 4
    seed = 0
    ckpt_path = './ckpt/fold/fold.pt'
    out_channels = 1195  # Number of classes

# ----------------------------
# 训练函数
# ----------------------------
def train(model, dataloader, optimizer, device):
    model.train()
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data).log_softmax(dim=-1), data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

# ----------------------------
# 测试函数
# ----------------------------
def test(model, dataloader, device):
    model.eval()
    correct = 0
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(dataloader.dataset)

# ----------------------------
# 主函数入口
# ----------------------------
if __name__ == '__main__':
    cfg = Config()
    print(cfg.__dict__)
    print("torch version:", torch.__version__)
    print("torchvision version:", torchvision.__version__)

    # 设置随机种子和设备
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    train_dataset = FOLDdataset_Struct2(root=cfg.data_dir, random_seed=cfg.seed, split='training')
    valid_dataset = FOLDdataset_Struct2(root=cfg.data_dir, random_seed=cfg.seed, split='validation')
    test_fold = FOLDdataset_Struct2(root=cfg.data_dir, random_seed=cfg.seed, split='test_fold')
    test_family = FOLDdataset_Struct2(root=cfg.data_dir, random_seed=cfg.seed, split='test_family')
    test_super = FOLDdataset_Struct2(root=cfg.data_dir, random_seed=cfg.seed, split='test_superfamily')

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)
    fold_loader = DataLoader(test_fold, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)
    family_loader = DataLoader(test_family, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)
    super_loader = DataLoader(test_super, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)

    # 构建模型
    model = Model(
        geometric_radii=[2*cfg.geometric_radius, 3*cfg.geometric_radius, 4*cfg.geometric_radius, 5*cfg.geometric_radius],
        sequential_kernel_size=cfg.sequential_kernel_size,
        kernel_channels=cfg.kernel_channels,
        channels=cfg.channels,
        base_width=cfg.base_width,
        num_classes=cfg.out_channels
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    # 学习率调度器
    lr_weights = []
    for i, milestone in enumerate(cfg.lr_milestones):
        if i == 0:
            lr_weights += [cfg.lr_gamma**i] * milestone
        else:
            lr_weights += [cfg.lr_gamma**i] * (milestone - cfg.lr_milestones[i - 1])
    if cfg.lr_milestones[-1] < cfg.num_epochs:
        lr_weights += [cfg.lr_gamma**len(cfg.lr_milestones)] * (cfg.num_epochs + 1 - cfg.lr_milestones[-1])
    lambda_lr = lambda epoch: lr_weights[epoch]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    # 训练与验证
    best_valid_acc = best_fold_acc = best_family_acc = best_super_acc = best_fold = best_family = best_super = 0.0
    best_epoch = 0

    for epoch in range(cfg.num_epochs):
        train(model, train_loader, optimizer, device)
        lr_scheduler.step()
        valid_acc = test(model, valid_loader, device)
        test_fold_acc = test(model, fold_loader, device)
        test_family_acc = test(model, family_loader, device)
        test_super_acc = test(model, super_loader, device)

        print(f'Epoch: {epoch+1:03d}, Validation: {valid_acc:.4f}, Fold: {test_fold_acc:.4f}, Family: {test_family_acc:.4f}, Super: {test_super_acc:.4f}')

        if valid_acc >= best_valid_acc:
            best_fold = test_fold_acc
            best_family = test_family_acc
            best_super = test_super_acc
            best_epoch = epoch
            best_valid_acc = valid_acc
            checkpoint = model.state_dict()

        best_fold_acc = max(best_fold_acc, test_fold_acc)
        best_family_acc = max(best_family_acc, test_family_acc)
        best_super_acc = max(best_super_acc, test_super_acc)

    # 打印结果
    print(cfg.__dict__)
    print(f'Best: {best_epoch+1:03d}, Validation: {best_valid_acc:.4f}, '
          f'Fold: {best_fold_acc:.4f}, Family: {best_family_acc:.4f}, Super: {best_super_acc:.4f}, '
          f'Validated Fold: {best_fold:.4f}, Validated Family: {best_family:.4f}, Validated Super: {best_super:.4f}')

    # 保存模型
    if cfg.ckpt_path:
        torch.save(checkpoint, cfg.ckpt_path)
