import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from utils import fmax
import Bio.PDB as pdb  # 可选导入

# 设置GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# 数据集定义
# -------------------------------
class ECData_ESM(Dataset):
    def __init__(self, root='/tmp/protein-data/ec', percent=30, random_seed=0, split='train', ca_path="/usr/data/protein/ec"):
        self.split = split
        self.root = root
        self.ca_path = ca_path
        self.random_state = np.random.RandomState(random_seed)

        npy_dir = os.path.join(ca_path, 'coordinates')
        fasta_file = os.path.join(ca_path, f'{split}.fasta')

        # 加载测试集筛选列表
        test_set = set()
        if split == "test":
            with open(os.path.join(ca_path, "nrPDB-EC_test.csv"), 'r') as f:
                for idx, line in enumerate(f):
                    if idx == 0:
                        continue
                    arr = line.rstrip().split(',')
                    if (
                        (percent == 30 and arr[1] == '1') or
                        (percent == 40 and arr[2] == '1') or
                        (percent == 50 and arr[3] == '1') or
                        (percent == 70 and arr[4] == '1') or
                        (percent == 95 and arr[5] == '1')
                    ):
                        test_set.add(arr[0])

        # 读取FASTA，构建蛋白质名称列表
        self.name = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    self.name.append(protein_name)

        # 构建标签和类别权重
        level_idx = 1
        ec_annotations = {}
        ec_num = {}
        self.labels = {}

        with open(os.path.join(ca_path, 'nrPDB-EC_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                arr = line.rstrip().split('\t')
                if idx == 1:
                    for ec in arr:
                        ec_annotations[ec] = len(ec_annotations)
                        ec_num[ec] = 0
                elif idx > 2:
                    protein_name = arr[0]
                    protein_ecs = arr[level_idx].split(',') if len(arr) > level_idx else []
                    protein_labels = [ec_annotations[ec] for ec in protein_ecs if ec]
                    for ec in protein_ecs:
                        if ec:
                            ec_num[ec] += 1
                    self.labels[protein_name] = np.array(protein_labels)

        self.num_classes = len(ec_annotations)
        self.weights = np.zeros(self.num_classes, dtype=np.float32)
        for ec, idx in ec_annotations.items():
            self.weights[idx] = len(self.labels) / ec_num[ec]

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        protein_name = self.name[idx]
        label = np.zeros((self.num_classes,), dtype=np.float32)
        if len(self.labels[protein_name]) > 0:
            label[self.labels[protein_name]] = 1.0

        esm_path = os.path.join(self.ca_path, 'esm2_emb', f'{protein_name}.npy')
        esm_emb = np.load(esm_path)[0][1:-1]  # 去掉[CLS]和[EOS] token

        return torch.from_numpy(esm_emb), label


# -------------------------------
# 模型定义
# -------------------------------
class PredESM(nn.Module):
    def __init__(self, device, layer_output, num_classes):
        super(PredESM, self).__init__()
        self.device = device
        self.dim = 1280
        self.layer_output = layer_output
        self.W_out = nn.ModuleList([nn.Linear(self.dim, self.dim) for _ in range(layer_output)])
        self.W_pred = nn.Linear(self.dim, num_classes)

    def forward(self, x):
        # 输入 shape: [batch_size, seq_len, 1280]
        x_mean = torch.mean(x, dim=1)  # mean pooling
        return self.W_pred(x_mean)


# -------------------------------
# 训练与测试函数
# -------------------------------
def train(epoch, dataloader, loss_fn):
    model.train()
    for data in dataloader:
        x, y = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        preds = model(x).sigmoid()
        loss = loss_fn(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

def test(dataloader):
    model.eval()
    probs, labels = [], []
    for data in dataloader:
        x, y = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            prob = model(x).sigmoid().cpu().numpy()
            y = y.cpu().numpy()
        probs.append(prob)
        labels.append(y)
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return fmax(probs, labels)


# -------------------------------
# 主程序入口
# -------------------------------
if __name__ == "__main__":
    SEED = 42
    batch_size = 1
    workers_num = 8
    num_epochs = 40
    lr = 0.001

    # 固定随机种子
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 加载数据集
    train_dataset = ECData_ESM(random_seed=SEED, split='train')
    valid_dataset = ECData_ESM(random_seed=SEED, split='valid')
    test_dataset_95 = ECData_ESM(random_seed=SEED, percent=95, split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers_num)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=workers_num)
    test_loader_95 = DataLoader(test_dataset_95, batch_size=batch_size, shuffle=False, num_workers=workers_num)

    # 初始化模型与优化器
    model = PredESM(device=device, layer_output=3, num_classes=train_dataset.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    loss_fn = nn.BCELoss(weight=torch.as_tensor(train_dataset.weights).to(device))

    best_valid = best_test_95 = best_95 = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        train(epoch, train_loader, loss_fn)
        valid_fmax = test(valid_loader)
        test_95 = test(test_loader_95)

        print(f'Epoch: {epoch+1:03d}, Validation: {valid_fmax:.4f}, Test: {test_95:.4f}')
        
        if valid_fmax >= best_valid:
            best_valid = valid_fmax
            best_95 = test_95
            best_epoch = epoch
            checkpoint = model.state_dict()

        best_test_95 = max(test_95, best_test_95)

    print(f'Best: {best_epoch+1:03d}, Validation: {best_valid:.4f}, Test: {best_test_95:.4f}, Valided Test: {best_95:.4f}')
