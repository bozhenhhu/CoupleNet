import numpy as np
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from datasets import GODataset1
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, fps, global_max_pool, global_mean_pool, radius

from models import Model1
from utils import fmax



if __name__ == '__main__':
    args = parse_args()
    


    test_dataset_95 = GODataset1(root=args.data_dir, random_seed=args.seed, level=args.level, percent=95, split='test')

    test_loader_95 = DataLoader(test_dataset_95, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = Model1().to(device)


    # model.load_state_dict(torch.load('./ckpt/go/go_cc.pt'))
    # print('model load successfully')
    # model.eval()

    probs = []
    labels = []
    lengths = []
    names = []
    with torch.no_grad():
        for data in test_loader_95:
            data = data.to(device)
            # prob = model(data).sigmoid().detach().cpu().numpy()
            # y = np.stack(data.y, axis=0)
            # probs.append(prob)
            # labels.append(y)
            l = data.x.size()
            lengths.append(l[0])
            names.append(data.name)
    # probs = np.concatenate(probs, axis=0)
    # labels = np.concatenate(labels, axis=0)

    mean = np.mean(lengths)

    print(mean)

    large_indices = np.where(lengths >= mean)
    prob = probs[large_indices]
    label = labels[large_indices]

    for i in range(10):
        tmp1 = prob[i, :]
        tmp2 = label[i, :]
        f_max = fmax(tmp1[np.newaxis, :], tmp2[np.newaxis, :])
        print(large_indices[0][i])
        print('large', f_max)
    print('large num', len(large_indices))
