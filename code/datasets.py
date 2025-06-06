import os
import math
import Bio.PDB as pdb

import numpy as np

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data

from utils import orientation

import os.path as osp
import h5py
import warnings
from tqdm import tqdm

import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch_geometric.data import InMemoryDataset

# AA Letter to id
aa = "ACDEFGHIKLMNPQRSTVWYX"
aa_to_id = {}
for i in range(0, 21):
    aa_to_id[aa[i]] = i


restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


# NB: restype_3to1 differs from Bio.PDB.protein_letters_3to1 by being a simple
# 1-to-1 mapping of 3 letter names to one letter names. The latter contains
# many more, and less common, three letter names as keys and maps many of these
# to the same one letter name (including 'X' and 'U' which we don't use here).
restype_3to1 = {v: k for k, v in restype_1to3.items()}


resname_to_pc7_dict = {'A': [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476],
                'C': [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
                'D': [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635],
                'E': [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582],
                'F': [ 0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
                'G': [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900],
                'H': [ 0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106],
                'I': [ 0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
                'K': [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
                'L': [ 0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
                'M': [ 0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
                'N': [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
                'P': [ 0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106],
                'Q': [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371],
                'R': [ 0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371],
                'S': [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
                'T': [ 0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
                'V': [ 0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900],
                'W': [ 0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529],
                'Y': [ 0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476],
                'X': [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]}


class FOLDdataset_Struct2(Dataset): 
    #add backbone dihedral angles and side-chain angles
    #add PC7 and interresidue geometries
    def __init__(self,
                 root,
                 random_seed=0, 
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 split='training',
                 ca_path='/usr/data/protein/fold',
                ):

        self.split = split
        self.random_state = np.random.RandomState(random_seed)
        self.root = root

        npy_dir = os.path.join(ca_path, 'coordinates', split)
        fasta_file = os.path.join(ca_path, split+'.fasta')

        # Load the fasta file.
        protein_seqs = []
        seq_pc7 = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    amino_ids = []
                    amino_pc7 = []
                    for amino in amino_chain:
                        amino_pc7.append(resname_to_pc7_dict[amino])
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))
                    seq_pc7.append(np.array(amino_pc7))

        fold_classes = {}
        with open(os.path.join(ca_path, 'class_map.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                fold_classes[arr[0]] = int(arr[1])

        protein_folds = {}
        with open(os.path.join(ca_path, split+'.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                protein_folds[arr[0]] = fold_classes[arr[-1]]

        self.seq_pc7 = seq_pc7
        self.data = []
        self.labels = []
        self.name = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            ori = orientation(pos)

            self.data.append((pos, ori, amino_ids.astype(int)))

            self.labels.append(protein_folds[protein_name])
            self.name.append(protein_name)

        self.num_classes = max(self.labels) + 1



    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
    # Load the dataset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pos, ori, amino = self.data[idx]
            label = self.labels[idx]
            pc7 = self.seq_pc7[idx]

            if self.split == "training":
                pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

            pos = pos.astype(dtype=np.float32)
            ori = ori.astype(dtype=np.float32)
            seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

            file_name = self.name[idx]
            file_path = self.root+"/"+self.split+"/"+file_name+".hdf5"
            side_chain_embs, bb_embs, amino_types_, pos_n, pos_cb = self.protein_to_graph(file_path, amino, pos)

            for i, n in enumerate(amino):
                if i < len(amino_types_):
                    if n != amino_types_[i]: 
                        side_chain_embs = torch.cat((side_chain_embs[:i], 
                                torch.zeros(1, side_chain_embs.shape[-1]),   # 新增加一行
                                side_chain_embs[i:]))
                        bb_embs = torch.cat((bb_embs[:i], torch.zeros(1, bb_embs.shape[-1]), bb_embs[i:]))
                        amino_types_ = np.insert(amino_types_, i, 20)
                        pos_n = torch.cat((pos_n[:i], torch.zeros(1, 3), pos_n[i:]))
                        pos_cb = torch.cat((pos_cb[:i], torch.zeros(1, 3), pos_cb[i:]))
                else:
                    side_chain_embs = torch.cat((side_chain_embs[:i], 
                                torch.zeros(1, side_chain_embs.shape[-1]),   # 新增加一行
                                side_chain_embs[i:]))
                    bb_embs = torch.cat((bb_embs[:i], torch.zeros(1, bb_embs.shape[-1]), bb_embs[i:]))
                    amino_types_ = np.insert(amino_types_, i, 20)
                    pos_n = torch.cat((pos_n[:i], torch.zeros(1, 3), pos_n[i:]))
                    pos_cb = torch.cat((pos_cb[:i], torch.zeros(1, 3), pos_cb[i:]))

            # assert bb_embs.shape[0] == amino.shape[0]
            data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                        edge_index = None,              # [2, num_edges]
                        edge_attr = None,               # [num_edges, num_edge_features]
                        y = label,
                        ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                        seq = torch.from_numpy(seq),    # [num_nodes, 1]
                        pos = torch.from_numpy(pos),
                        side_chain_embs = side_chain_embs,
                        bb_embs = bb_embs,
                        pc7 = torch.FloatTensor(pc7),
                        pos_n = pos_n,
                        pos_cb = pos_cb
                        )    # [num_nodes, num_dimensions]
        return data

    def _normalize(self,tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))
    
    def ceterize(self, pos):
        pos = np.nan_to_num(pos)
        center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
        pos = pos - center
        return pos

    def get_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names, b'N')
        mask_ca = np.char.equal(atom_names, b'CA')
        mask_c = np.char.equal(atom_names, b'C')
        mask_cb = np.char.equal(atom_names, b'CB')
        mask_g = np.char.equal(atom_names, b'CG') | np.char.equal(atom_names, b'SG') | np.char.equal(atom_names, b'OG') | np.char.equal(atom_names, b'CG1') | np.char.equal(atom_names, b'OG1')
        mask_d = np.char.equal(atom_names, b'CD') | np.char.equal(atom_names, b'SD') | np.char.equal(atom_names, b'CD1') | np.char.equal(atom_names, b'OD1') | np.char.equal(atom_names, b'ND1')
        mask_e = np.char.equal(atom_names, b'CE') | np.char.equal(atom_names, b'NE') | np.char.equal(atom_names, b'OE1')
        mask_z = np.char.equal(atom_names, b'CZ') | np.char.equal(atom_names, b'NZ')
        mask_h = np.char.equal(atom_names, b'NH1')

        pos_n = np.full((len(amino_types),3),np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        pos_n_ = self.ceterize(pos_n)
        pos_n_ = torch.FloatTensor(pos_n_)
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types),3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types),3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        # pos_c = self.ceterize(pos_c)
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types),3),np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        pos_cb_ = self.ceterize(pos_cb)
        pos_cb_ = torch.FloatTensor(pos_cb_)
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types),3),np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        # pos_g = self.ceterize(pos_g)
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types),3),np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        # pos_d = self.ceterize(pos_d)
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types),3),np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        # pos_e = self.ceterize(pos_e)
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types),3),np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        # pos_z = self.ceterize(pos_z)
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types),3),np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        # pos_h = self.ceterize(pos_h)
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h, pos_n_, pos_cb_


    def side_chain_embs(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(self.compute_diherals(v1, v2, v3),1)
        angle2 = torch.unsqueeze(self.compute_diherals(v2, v3, v4),1)
        angle3 = torch.unsqueeze(self.compute_diherals(v3, v4, v5),1)
        angle4 = torch.unsqueeze(self.compute_diherals(v4, v5, v6),1)
        angle5 = torch.unsqueeze(self.compute_diherals(v5, v6, v7),1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4),1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)),1)
        
        return side_chain_embs

    
    def bb_embs(self, X):   
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6 
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self.compute_diherals(u0, u1, u2)
        
        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2]) 
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features

    
    def compute_diherals(self, v1, v2, v3):
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion
    
    
    def protein_to_graph(self, pFilePath, amino, pos):
        h5File = h5py.File(pFilePath, "r")
        
        amino_types = h5File['amino_types'][()] # size: (n_amino,)

        mask = amino_types == -1
        if np.sum(mask) > 0:
            amino_types[mask] = 25 # for amino acid types, set the value of -1 to 25
        atom_amino_id = h5File['atom_amino_id'][()] # size: (n_atom,)
        atom_names = h5File['atom_names'][()] # size: (n_atom,)
        atom_pos = h5File['atom_pos'][()][0] #size: (n_atom,3)

        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h, pos_n_, pos_cb_ = self.get_atom_pos(amino_types, atom_names, atom_amino_id, atom_pos)

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        side_chain_embs = self.side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_embs[torch.isnan(side_chain_embs)] = 0

        # three backbone torsion angles
        bb_embs = self.bb_embs(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
        bb_embs[torch.isnan(bb_embs)] = 0

        atom_residue_names = h5File['atom_residue_names'][()]
        first_pos = {}
        amino_types_ = []
        for i, n in enumerate(atom_amino_id):
            if n not in first_pos:
                first_pos[n] = i
                residue3 = atom_residue_names[i].decode()
                if residue3 in restype_3to1.keys():
                    three2one = restype_3to1[residue3]
                    amino_types_.append(aa_to_id[three2one])
                else:
                    j = len(amino_types_)
                    side_chain_embs = torch.cat((side_chain_embs[:j], side_chain_embs[j+1:]))
                    bb_embs = torch.cat((bb_embs[:j], bb_embs[j+1:]))
                    pos_n_ = torch.cat((pos_n_[:j], pos_n_[j+1:]))
                    pos_cb_= torch.cat((pos_cb_[:j], pos_cb_[j+1:]))

        h5File.close()
        return side_chain_embs, bb_embs, amino_types_, pos_n_, pos_cb_

class FOLDdataset_Struct(Dataset): 
    #add backbone dihedral angles and side-chain angles
    def __init__(self,
                 root,
                 random_seed=0, 
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 split='training',
                 ca_path='/usr/data/protein/fold',
                ):

        self.split = split
        self.random_state = np.random.RandomState(random_seed)
        self.root = root

        npy_dir = os.path.join(ca_path, 'coordinates', split)
        fasta_file = os.path.join(ca_path, split+'.fasta')

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        fold_classes = {}
        with open(os.path.join(ca_path, 'class_map.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                fold_classes[arr[0]] = int(arr[1])

        protein_folds = {}
        with open(os.path.join(ca_path, split+'.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                protein_folds[arr[0]] = fold_classes[arr[-1]]

        self.data = []
        self.labels = []
        self.name = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            ori = orientation(pos)

            self.data.append((pos, ori, amino_ids.astype(int)))

            self.labels.append(protein_folds[protein_name])
            self.name.append(protein_name)

        self.num_classes = max(self.labels) + 1



    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
    # Load the dataset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pos, ori, amino = self.data[idx]
            label = self.labels[idx]

            if self.split == "training":
                pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

            pos = pos.astype(dtype=np.float32)
            ori = ori.astype(dtype=np.float32)
            seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)
            try:
                file_name = self.name[idx]
                file_path = self.root+"/"+self.split+"/"+file_name+".hdf5"
                side_chain_embs, bb_embs = self.protein_to_graph(file_path, amino, pos)
                assert bb_embs.shape[0] == amino.shape[0]
            except: 
                side_chain_embs = torch.zeros((len(amino), 8))
                bb_embs = torch.zeros((len(amino),6))
          
            data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                        edge_index = None,              # [2, num_edges]
                        edge_attr = None,               # [num_edges, num_edge_features]
                        y = label,
                        ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                        seq = torch.from_numpy(seq),    # [num_nodes, 1]
                        pos = torch.from_numpy(pos),
                        side_chain_embs = side_chain_embs,
                        bb_embs = bb_embs
                        )    # [num_nodes, num_dimensions]
        return data

    def _normalize(self,tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def get_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names, b'N')
        mask_ca = np.char.equal(atom_names, b'CA')
        mask_c = np.char.equal(atom_names, b'C')
        mask_cb = np.char.equal(atom_names, b'CB')
        mask_g = np.char.equal(atom_names, b'CG') | np.char.equal(atom_names, b'SG') | np.char.equal(atom_names, b'OG') | np.char.equal(atom_names, b'CG1') | np.char.equal(atom_names, b'OG1')
        mask_d = np.char.equal(atom_names, b'CD') | np.char.equal(atom_names, b'SD') | np.char.equal(atom_names, b'CD1') | np.char.equal(atom_names, b'OD1') | np.char.equal(atom_names, b'ND1')
        mask_e = np.char.equal(atom_names, b'CE') | np.char.equal(atom_names, b'NE') | np.char.equal(atom_names, b'OE1')
        mask_z = np.char.equal(atom_names, b'CZ') | np.char.equal(atom_names, b'NZ')
        mask_h = np.char.equal(atom_names, b'NH1')

        pos_n = np.full((len(amino_types),3),np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        # pos_n = self.ceterize(pos_n)
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types),3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types),3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        # pos_c = self.ceterize(pos_c)
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types),3),np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        # pos_cb = self.ceterize(pos_cb)
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types),3),np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        # pos_g = self.ceterize(pos_g)
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types),3),np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        # pos_d = self.ceterize(pos_d)
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types),3),np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        # pos_e = self.ceterize(pos_e)
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types),3),np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        # pos_z = self.ceterize(pos_z)
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types),3),np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        # pos_h = self.ceterize(pos_h)
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h


    def side_chain_embs(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(self.compute_diherals(v1, v2, v3),1)
        angle2 = torch.unsqueeze(self.compute_diherals(v2, v3, v4),1)
        angle3 = torch.unsqueeze(self.compute_diherals(v3, v4, v5),1)
        angle4 = torch.unsqueeze(self.compute_diherals(v4, v5, v6),1)
        angle5 = torch.unsqueeze(self.compute_diherals(v5, v6, v7),1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4),1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)),1)
        
        return side_chain_embs

    
    def bb_embs(self, X):   
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6 
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self.compute_diherals(u0, u1, u2)
        
        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2]) 
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features

    
    def compute_diherals(self, v1, v2, v3):
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion
    
    
    def protein_to_graph(self, pFilePath, amino, pos):
        h5File = h5py.File(pFilePath, "r")
        
        amino_types = h5File['amino_types'][()] # size: (n_amino,)

        mask = amino_types == -1
        if np.sum(mask) > 0:
            amino_types[mask] = 25 # for amino acid types, set the value of -1 to 25
        atom_amino_id = h5File['atom_amino_id'][()] # size: (n_atom,)
        atom_names = h5File['atom_names'][()] # size: (n_atom,)
        atom_pos = h5File['atom_pos'][()][0] #size: (n_atom,3)

        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = self.get_atom_pos(amino_types, atom_names, atom_amino_id, atom_pos)

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        side_chain_embs = self.side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_embs[torch.isnan(side_chain_embs)] = 0

        # three backbone torsion angles
        bb_embs = self.bb_embs(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
        bb_embs[torch.isnan(bb_embs)] = 0

        h5File.close()
        return side_chain_embs, bb_embs

class FoldDataset_Ca(Dataset):

    def __init__(self, root='/tmp/protein-data/fold', random_seed=0, split='training'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        npy_dir = os.path.join(root, 'coordinates', split)
        fasta_file = os.path.join(root, split+'.fasta')

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        fold_classes = {}
        with open(os.path.join(root, 'class_map.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                fold_classes[arr[0]] = int(arr[1])

        protein_folds = {}
        with open(os.path.join(root, split+'.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                protein_folds[arr[0]] = fold_classes[arr[-1]]

        self.data = []
        self.labels = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            ori = orientation(pos)

            self.data.append((pos, ori, amino_ids.astype(int)))

            self.labels.append(protein_folds[protein_name])

        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pos, ori, amino = self.data[idx]
        label = self.labels[idx]

        if self.split == "training":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos))    # [num_nodes, num_dimensions]

        return data

class FuncDataset_Struct(Dataset):

    def __init__(self, root='/tmp/protein-data/func', random_seed=0, split='training', ca_path='/usr/data/protein/func'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        self.root = root
        # Get the paths.
        npy_dir = os.path.join(os.path.join(ca_path, 'coordinates'), split)
        fasta_file = os.path.join(ca_path, 'chain_'+split+'.fasta')

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        protein_functions = {}
        with open(os.path.join(ca_path, 'chain_functions.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split(',')
                protein_functions[arr[0]] = int(arr[1])

        self.data = []
        self.labels = []
        self.names = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center

            ori = orientation(pos)
            self.data.append((pos, ori, amino_ids.astype(int)))
            self.labels.append(protein_functions[protein_name])
            self.names.append(protein_name)

        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pos, ori, amino = self.data[idx]
            label = self.labels[idx]

            if self.split == "training":
                pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

            pos = pos.astype(dtype=np.float32)
            ori = ori.astype(dtype=np.float32)
            seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

            file_name = self.names[idx]
            file_path = self.root+"/"+'data'+"/"+file_name+".hdf5"
            side_chain_embs, bb_embs, amino_types_= self.protein_to_graph(file_path, amino, pos)

            for i, n in enumerate(amino):
                if i < len(amino_types_):
                    if n != amino_types_[i]: 
                        side_chain_embs = torch.cat((side_chain_embs[:i], 
                                torch.zeros(1, side_chain_embs.shape[-1]),   # 新增加一行
                                side_chain_embs[i:]))
                        bb_embs = torch.cat((bb_embs[:i], torch.zeros(1, bb_embs.shape[-1]), bb_embs[i:]))
                        amino_types_ = np.insert(amino_types_, i, 20)
                else:
                    side_chain_embs = torch.cat((side_chain_embs[:i], 
                                torch.zeros(1, side_chain_embs.shape[-1]),   # 新增加一行
                                side_chain_embs[i:]))
                    bb_embs = torch.cat((bb_embs[:i], torch.zeros(1, bb_embs.shape[-1]), bb_embs[i:]))
                    amino_types_ = np.insert(amino_types_, i, 20)
            try:
                assert bb_embs.shape[0] == amino.shape[0]
            except:
                side_chain_embs = torch.full((len(amino), 8), torch.nan)
                bb_embs = torch.full((len(amino),6), torch.nan) 
                   
            data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                        edge_index = None,              # [2, num_edges]
                        edge_attr = None,               # [num_edges, num_edge_features]
                        y = label,
                        ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                        seq = torch.from_numpy(seq),    # [num_nodes, 1]
                        pos = torch.from_numpy(pos),
                        side_chain_embs = side_chain_embs,
                        bb_embs = bb_embs
                        )    # [num_nodes, num_dimensions]

        return data
    
    
    def _normalize(self,tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def get_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names, b'N')
        mask_ca = np.char.equal(atom_names, b'CA')
        mask_c = np.char.equal(atom_names, b'C')
        mask_cb = np.char.equal(atom_names, b'CB')
        mask_g = np.char.equal(atom_names, b'CG') | np.char.equal(atom_names, b'SG') | np.char.equal(atom_names, b'OG') | np.char.equal(atom_names, b'CG1') | np.char.equal(atom_names, b'OG1')
        mask_d = np.char.equal(atom_names, b'CD') | np.char.equal(atom_names, b'SD') | np.char.equal(atom_names, b'CD1') | np.char.equal(atom_names, b'OD1') | np.char.equal(atom_names, b'ND1')
        mask_e = np.char.equal(atom_names, b'CE') | np.char.equal(atom_names, b'NE') | np.char.equal(atom_names, b'OE1')
        mask_z = np.char.equal(atom_names, b'CZ') | np.char.equal(atom_names, b'NZ')
        mask_h = np.char.equal(atom_names, b'NH1')

        pos_n = np.full((len(amino_types),3),np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        # pos_n = self.ceterize(pos_n)
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types),3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types),3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        # pos_c = self.ceterize(pos_c)
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types),3),np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        # pos_cb = self.ceterize(pos_cb)
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types),3),np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        # pos_g = self.ceterize(pos_g)
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types),3),np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        # pos_d = self.ceterize(pos_d)
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types),3),np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        # pos_e = self.ceterize(pos_e)
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types),3),np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        # pos_z = self.ceterize(pos_z)
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types),3),np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        # pos_h = self.ceterize(pos_h)
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h


    def side_chain_embs(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(self.compute_diherals(v1, v2, v3),1)
        angle2 = torch.unsqueeze(self.compute_diherals(v2, v3, v4),1)
        angle3 = torch.unsqueeze(self.compute_diherals(v3, v4, v5),1)
        angle4 = torch.unsqueeze(self.compute_diherals(v4, v5, v6),1)
        angle5 = torch.unsqueeze(self.compute_diherals(v5, v6, v7),1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4),1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)),1)
        
        return side_chain_embs

    
    def bb_embs(self, X):   
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6 
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self.compute_diherals(u0, u1, u2)
        
        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2]) 
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features

    
    def compute_diherals(self, v1, v2, v3):
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion
    
    
    def protein_to_graph(self, pFilePath, amino, pos):
        h5File = h5py.File(pFilePath, "r")
        
        amino_types = h5File['amino_types'][()] # size: (n_amino,)

        mask = amino_types == -1
        if np.sum(mask) > 0:
            amino_types[mask] = 25 # for amino acid types, set the value of -1 to 25
        atom_amino_id = h5File['atom_amino_id'][()] # size: (n_atom,)
        atom_names = h5File['atom_names'][()] # size: (n_atom,)
        atom_pos = h5File['atom_pos'][()][0] #size: (n_atom,3)

        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = self.get_atom_pos(amino_types, atom_names, atom_amino_id, atom_pos)

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        side_chain_embs = self.side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_embs[torch.isnan(side_chain_embs)] = 0

        # three backbone torsion angles
        bb_embs = self.bb_embs(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
        bb_embs[torch.isnan(bb_embs)] = 0

        atom_residue_names = h5File['atom_residue_names'][()]
        first_pos = {}
        amino_types_ = []
        for i, n in enumerate(atom_amino_id):
            if n not in first_pos:
                first_pos[n] = i
                residue3 = atom_residue_names[i].decode()
                if residue3 in restype_3to1.keys():
                    three2one = restype_3to1[residue3]
                    amino_types_.append(aa_to_id[three2one])
                else:
                    j = len(amino_types_)
                    side_chain_embs = torch.cat((side_chain_embs[:j], side_chain_embs[j+1:]))
                    bb_embs = torch.cat((bb_embs[:j], bb_embs[j+1:]))

        h5File.close()
        return side_chain_embs, bb_embs, amino_types_

class FuncDataset_Ca(Dataset):

    def __init__(self, root='/tmp/protein-data/func', random_seed=0, split='training'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        # Get the paths.
        npy_dir = os.path.join(os.path.join(root, 'coordinates'), split)
        fasta_file = os.path.join(root, 'chain_'+split+'.fasta')

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        protein_functions = {}
        with open(os.path.join(root, 'chain_functions.txt'), 'r') as f:
            for line in f:
                arr = line.rstrip().split(',')
                protein_functions[arr[0]] = int(arr[1])

        self.data = []
        self.labels = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center

            ori = orientation(pos)
            self.data.append((pos, ori, amino_ids.astype(int)))
            self.labels.append(protein_functions[protein_name])

        self.num_classes = max(self.labels) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pos, ori, amino = self.data[idx]
        label = self.labels[idx]

        if self.split == "training":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos))    # [num_nodes, num_dimensions]

        return data

class GODataset_Struct(Dataset):

    def __init__(self, root='/tmp/protein-data/go', level='mf', percent=30, random_seed=0, split='train', 
                 ca_path='/usr/data/protein/go',):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split
        self.root = root

        # Get the paths.
        npy_dir = os.path.join(ca_path, 'coordinates')
        fasta_file = os.path.join(ca_path, split+'.fasta')

        # Mask test set.
        test_set = set()
        if split == "test":
            with open(os.path.join(ca_path, "nrPDB-GO_2019.06.18_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))


        self.data = []
        self.name = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center

            ori = orientation(pos)
            self.data.append((protein_name, pos, ori, amino_ids.astype(int)))
            self.name.append(protein_name)
        
        self.files_name = {}
        file_path = root + '/'+split + '/'
        files = os.listdir(file_path)
        for file in files:
            file = file.rstrip('.pdb')  
            tmp = file.split('_')
            self.files_name[tmp[0]] = tmp[1]

        level_idx = 0
        go_cnt = 0
        go_num = {}
        go_annotations = {}
        self.labels = {}
        with open(os.path.join(ca_path, 'nrPDB-GO_2019.06.18_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1 and level == "mf":
                    level_idx = 1
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 5 and level == "bp":
                    level_idx = 2
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 9 and level == "cc":
                    level_idx = 3
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx > 12:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_go_list = arr[level_idx]
                        protein_go_list = protein_go_list.split(',')
                        for go in protein_go_list:
                            if len(go) > 0:
                                protein_labels.append(go_annotations[go])
                                go_num[go] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(go_annotations)

        self.weights = np.zeros((go_cnt,), dtype=np.float32)
        for go, idx in go_annotations.items():
            self.weights[idx] = len(self.labels)/go_num[go]

        # self.go_annotations = go_annotations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            protein_name, pos, ori, amino = self.data[idx]
            label = np.zeros((self.num_classes,)).astype(np.float32)
            if len(self.labels[protein_name]) > 0:
                label[self.labels[protein_name]] = 1.0

            if self.split == "train":
                pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

            pos = pos.astype(dtype=np.float32)
            ori = ori.astype(dtype=np.float32)
            seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

            file_name = self.name[idx]
            try:
                id = self.files_name[file_name]
                file_path = self.root+"/"+self.split+"/"+file_name+'_'+id+".pdb"
                side_chain_embs, bb_embs = self.protein_to_graph(file_path, amino, pos)
            except: 
                side_chain_embs = torch.zeros((len(amino), 8))
                bb_embs = torch.zeros((len(amino),6))  
     
            data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                        edge_index = None,              # [2, num_edges]
                        edge_attr = None,               # [num_edges, num_edge_features]
                        y = label,
                        ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                        seq = torch.from_numpy(seq),    # [num_nodes, 1]
                        pos = torch.from_numpy(pos),
                        side_chain_embs = side_chain_embs,
                        bb_embs = bb_embs,
                        # name = file_name
                        )    # [num_nodes, num_dimensions]

        return data
    
    def _normalize(self,tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def get_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names, b'N')
        mask_ca = np.char.equal(atom_names, b'CA')
        mask_c = np.char.equal(atom_names, b'C')
        mask_cb = np.char.equal(atom_names, b'CB')
        mask_g = np.char.equal(atom_names, b'CG') | np.char.equal(atom_names, b'SG') | np.char.equal(atom_names, b'OG') | np.char.equal(atom_names, b'CG1') | np.char.equal(atom_names, b'OG1')
        mask_d = np.char.equal(atom_names, b'CD') | np.char.equal(atom_names, b'SD') | np.char.equal(atom_names, b'CD1') | np.char.equal(atom_names, b'OD1') | np.char.equal(atom_names, b'ND1')
        mask_e = np.char.equal(atom_names, b'CE') | np.char.equal(atom_names, b'NE') | np.char.equal(atom_names, b'OE1')
        mask_z = np.char.equal(atom_names, b'CZ') | np.char.equal(atom_names, b'NZ')
        mask_h = np.char.equal(atom_names, b'NH1')

        pos_n = np.full((len(amino_types),3),np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        # pos_n = self.ceterize(pos_n)
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types),3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types),3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        # pos_c = self.ceterize(pos_c)
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types),3),np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        # pos_cb = self.ceterize(pos_cb)
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types),3),np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        # pos_g = self.ceterize(pos_g)
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types),3),np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        # pos_d = self.ceterize(pos_d)
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types),3),np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        # pos_e = self.ceterize(pos_e)
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types),3),np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        # pos_z = self.ceterize(pos_z)
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types),3),np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        # pos_h = self.ceterize(pos_h)
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h


    def side_chain_embs(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(self.compute_diherals(v1, v2, v3),1)
        angle2 = torch.unsqueeze(self.compute_diherals(v2, v3, v4),1)
        angle3 = torch.unsqueeze(self.compute_diherals(v3, v4, v5),1)
        angle4 = torch.unsqueeze(self.compute_diherals(v4, v5, v6),1)
        angle5 = torch.unsqueeze(self.compute_diherals(v5, v6, v7),1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4),1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)),1)
        
        return side_chain_embs

    
    def bb_embs(self, X):   
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6 
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self.compute_diherals(u0, u1, u2)
        
        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2]) 
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features

    
    def compute_diherals(self, v1, v2, v3):
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion
    
    
    def protein_to_graph(self, pFilePath, amino, pos_ca_ori):
        with open(pFilePath) as f:
            pdb_structure = pdb.PDBParser().get_structure('protein', f)

    
        pos = np.full((len(amino),3),np.nan)
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = pos, pos, pos, pos, pos, pos, pos, pos, pos

        # 获得所有的原子坐标
        sequence = []
        for residue in pdb_structure.get_residues():
            tmp_residue = residue.get_resname()
            if tmp_residue in restype_3to1.keys():
                seq = aa_to_id[restype_3to1[tmp_residue]]
                sequence.append(seq)
                for atom in residue.get_atoms():
                    atom_name = atom.get_name()
                    coord = atom.get_coord()
                    if atom_name == 'N':
                        pos_n[i] = coord
                    elif atom_name == 'CA':
                        pos_ca[i] = coord
                    elif atom_name == 'C':
                        pos_c[i] = coord
                    elif atom_name == 'CB':
                        pos_cb[i] = coord
                    elif atom_name in ['CG', 'SG', 'OG', 'CG1', 'OG1']:
                        pos_g[i] = coord
                    elif atom_name in ['CD', 'SD', 'CD1', 'OD1', 'ND1']:
                        pos_d[i] = coord
                    elif atom_name in ['CE', 'NE','OE1']:
                        pos_e[i] = coord
                    elif atom_name in ['CZ', 'NZ']:
                        pos_z[i] = coord
                    elif atom_name == 'NH1':
                        pos_h[i] = coord

        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = torch.FloatTensor(pos_n), torch.FloatTensor(pos_ca),torch.FloatTensor(pos_c), torch.FloatTensor(pos_cb),torch.FloatTensor(pos_g), torch.FloatTensor(pos_d), torch.FloatTensor(pos_e), torch.FloatTensor(pos_z), torch.FloatTensor(pos_h)
        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        side_chain_embs = self.side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_embs[torch.isnan(side_chain_embs)] = 0

        # three backbone torsion angles
        bb_embs = self.bb_embs(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
        bb_embs[torch.isnan(bb_embs)] = 0

        return side_chain_embs, bb_embs

class GODataset_ESM(Dataset):

    def __init__(self, root='/tmp/protein-data/go', level='mf', percent=30, random_seed=0, split='train', 
                 ca_path='/usr/data/protein/go',):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split
        self.root = root

        # Get the paths.
        npy_dir = os.path.join(ca_path, 'coordinates')
        fasta_file = os.path.join(ca_path, split+'.fasta')
        self.ca_path = ca_path

        # Mask test set.
        test_set = set()
        if split == "test":
            with open(os.path.join(ca_path, "nrPDB-GO_2019.06.18_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))


        self.data = []
        self.name = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center

            ori = orientation(pos)
            self.data.append((protein_name, pos, ori, amino_ids.astype(int)))
            self.name.append(protein_name)
        
        self.files_name = {}
        file_path = root + '/'+split + '/'
        files = os.listdir(file_path)
        for file in files:
            file = file.rstrip('.pdb')  
            tmp = file.split('_')
            self.files_name[tmp[0]] = tmp[1]

        level_idx = 0
        go_cnt = 0
        go_num = {}
        go_annotations = {}
        self.labels = {}
        with open(os.path.join(ca_path, 'nrPDB-GO_2019.06.18_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1 and level == "mf":
                    level_idx = 1
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 5 and level == "bp":
                    level_idx = 2
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 9 and level == "cc":
                    level_idx = 3
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx > 12:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_go_list = arr[level_idx]
                        protein_go_list = protein_go_list.split(',')
                        for go in protein_go_list:
                            if len(go) > 0:
                                protein_labels.append(go_annotations[go])
                                go_num[go] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(go_annotations)

        file_path = "GO_frequences.txt"
        import json
        with open(file_path, "w") as file:
            json.dump(go_num, file)

        self.weights = np.zeros((go_cnt,), dtype=np.float32)
        for go, idx in go_annotations.items():
            self.weights[idx] = len(self.labels)/go_num[go]

        # self.go_annotations = go_annotations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            protein_name, pos, ori, amino = self.data[idx]
            label = np.zeros((self.num_classes,)).astype(np.float32)
            if len(self.labels[protein_name]) > 0:
                label[self.labels[protein_name]] = 1.0

            if self.split == "train":
                pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

            pos = pos.astype(dtype=np.float32)
            ori = ori.astype(dtype=np.float32)
            seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

            file_name = self.name[idx]
            try:
                id = self.files_name[file_name]
                file_path = self.root+"/"+self.split+"/"+file_name+'_'+id+".pdb"
                side_chain_embs, bb_embs = self.protein_to_graph(file_path, amino, pos)
            except: 
                side_chain_embs = torch.zeros((len(amino), 8))
                bb_embs = torch.zeros((len(amino),6)) 
            
            esm_path = self.ca_path+"/"+'esm2_emb'+"/"+protein_name+".npy"
            esm_emb = np.load(esm_path)
            esm_emb = esm_emb[0][1:-1] 
     
            data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                        edge_index = None,              # [2, num_edges]
                        edge_attr = None,               # [num_edges, num_edge_features]
                        y = label,
                        ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                        seq = torch.from_numpy(seq),    # [num_nodes, 1]
                        pos = torch.from_numpy(pos),
                        side_chain_embs = side_chain_embs,
                        bb_embs = bb_embs,
                        esm_emb = torch.from_numpy(esm_emb)
                        # name = file_name
                        )    # [num_nodes, num_dimensions]

        return data
    
    def _normalize(self,tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def get_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names, b'N')
        mask_ca = np.char.equal(atom_names, b'CA')
        mask_c = np.char.equal(atom_names, b'C')
        mask_cb = np.char.equal(atom_names, b'CB')
        mask_g = np.char.equal(atom_names, b'CG') | np.char.equal(atom_names, b'SG') | np.char.equal(atom_names, b'OG') | np.char.equal(atom_names, b'CG1') | np.char.equal(atom_names, b'OG1')
        mask_d = np.char.equal(atom_names, b'CD') | np.char.equal(atom_names, b'SD') | np.char.equal(atom_names, b'CD1') | np.char.equal(atom_names, b'OD1') | np.char.equal(atom_names, b'ND1')
        mask_e = np.char.equal(atom_names, b'CE') | np.char.equal(atom_names, b'NE') | np.char.equal(atom_names, b'OE1')
        mask_z = np.char.equal(atom_names, b'CZ') | np.char.equal(atom_names, b'NZ')
        mask_h = np.char.equal(atom_names, b'NH1')

        pos_n = np.full((len(amino_types),3),np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        # pos_n = self.ceterize(pos_n)
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types),3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types),3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        # pos_c = self.ceterize(pos_c)
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types),3),np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        # pos_cb = self.ceterize(pos_cb)
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types),3),np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        # pos_g = self.ceterize(pos_g)
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types),3),np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        # pos_d = self.ceterize(pos_d)
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types),3),np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        # pos_e = self.ceterize(pos_e)
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types),3),np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        # pos_z = self.ceterize(pos_z)
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types),3),np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        # pos_h = self.ceterize(pos_h)
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h


    def side_chain_embs(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(self.compute_diherals(v1, v2, v3),1)
        angle2 = torch.unsqueeze(self.compute_diherals(v2, v3, v4),1)
        angle3 = torch.unsqueeze(self.compute_diherals(v3, v4, v5),1)
        angle4 = torch.unsqueeze(self.compute_diherals(v4, v5, v6),1)
        angle5 = torch.unsqueeze(self.compute_diherals(v5, v6, v7),1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4),1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)),1)
        
        return side_chain_embs

    
    def bb_embs(self, X):   
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6 
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self.compute_diherals(u0, u1, u2)
        
        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2]) 
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features

    
    def compute_diherals(self, v1, v2, v3):
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion
    
    
    def protein_to_graph(self, pFilePath, amino, pos_ca_ori):
        with open(pFilePath) as f:
            pdb_structure = pdb.PDBParser().get_structure('protein', f)

    
        pos = np.full((len(amino),3),np.nan)
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = pos, pos, pos, pos, pos, pos, pos, pos, pos

        # 获得所有的原子坐标
        sequence = []
        for residue in pdb_structure.get_residues():
            tmp_residue = residue.get_resname()
            if tmp_residue in restype_3to1.keys():
                seq = aa_to_id[restype_3to1[tmp_residue]]
                sequence.append(seq)
                for atom in residue.get_atoms():
                    atom_name = atom.get_name()
                    coord = atom.get_coord()
                    if atom_name == 'N':
                        pos_n[i] = coord
                    elif atom_name == 'CA':
                        pos_ca[i] = coord
                    elif atom_name == 'C':
                        pos_c[i] = coord
                    elif atom_name == 'CB':
                        pos_cb[i] = coord
                    elif atom_name in ['CG', 'SG', 'OG', 'CG1', 'OG1']:
                        pos_g[i] = coord
                    elif atom_name in ['CD', 'SD', 'CD1', 'OD1', 'ND1']:
                        pos_d[i] = coord
                    elif atom_name in ['CE', 'NE','OE1']:
                        pos_e[i] = coord
                    elif atom_name in ['CZ', 'NZ']:
                        pos_z[i] = coord
                    elif atom_name == 'NH1':
                        pos_h[i] = coord

        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = torch.FloatTensor(pos_n), torch.FloatTensor(pos_ca),torch.FloatTensor(pos_c), torch.FloatTensor(pos_cb),torch.FloatTensor(pos_g), torch.FloatTensor(pos_d), torch.FloatTensor(pos_e), torch.FloatTensor(pos_z), torch.FloatTensor(pos_h)
        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        side_chain_embs = self.side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_embs[torch.isnan(side_chain_embs)] = 0

        # three backbone torsion angles
        bb_embs = self.bb_embs(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
        bb_embs[torch.isnan(bb_embs)] = 0

        return side_chain_embs, bb_embs
            
class GODataset_Ca(Dataset):

    def __init__(self, root='/tmp/protein-data/go', level='mf', percent=30, random_seed=0, split='train'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        # Get the paths.
        npy_dir = os.path.join(root, 'coordinates')
        fasta_file = os.path.join(root, split+'.fasta')

        # Mask test set.
        test_set = set()
        if split == "test":
            with open(os.path.join(root, "nrPDB-GO_2019.06.18_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))


        self.data = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))

            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center

            ori = orientation(pos)
            self.data.append((protein_name, pos, ori, amino_ids.astype(int)))

        level_idx = 0
        go_cnt = 0
        go_num = {}
        go_annotations = {}
        self.labels = {}
        with open(os.path.join(root, 'nrPDB-GO_2019.06.18_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1 and level == "mf":
                    level_idx = 1
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 5 and level == "bp":
                    level_idx = 2
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx == 9 and level == "cc":
                    level_idx = 3
                    arr = line.rstrip().split('\t')
                    for go in arr:
                        go_annotations[go] = go_cnt
                        go_num[go] = 0
                        go_cnt += 1
                elif idx > 12:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_go_list = arr[level_idx]
                        protein_go_list = protein_go_list.split(',')
                        for go in protein_go_list:
                            if len(go) > 0:
                                protein_labels.append(go_annotations[go])
                                go_num[go] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(go_annotations)

        self.weights = np.zeros((go_cnt,), dtype=np.float32)
        for go, idx in go_annotations.items():
            self.weights[idx] = len(self.labels)/go_num[go]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_name, pos, ori, amino = self.data[idx]
        label = np.zeros((self.num_classes,)).astype(np.float32)
        if len(self.labels[protein_name]) > 0:
            label[self.labels[protein_name]] = 1.0

        if self.split == "train":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos))    # [num_nodes, num_dimensions]

        return data

class ECDataset_Ca(Dataset):

    def __init__(self, root='/tmp/protein-data/ec', percent=30, random_seed=0, split='train'):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split

        # Get the paths.
        npy_dir = os.path.join(root, 'coordinates')
        fasta_file = os.path.join(root, split+'.fasta')

        # Mask test set.
        test_set = set()
        if split == "test":
            with open(os.path.join(root, "nrPDB-EC_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))


        self.data = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))
            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            ori = orientation(pos)
            self.data.append((protein_name, pos, ori, amino_ids.astype(int)))

        level_idx = 1
        ec_cnt = 0
        ec_num = {}
        ec_annotations = {}
        self.labels = {}

        with open(os.path.join(root, 'nrPDB-EC_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1:
                    arr = line.rstrip().split('\t')
                    for ec in arr:
                        ec_annotations[ec] = ec_cnt
                        ec_num[ec] = 0
                        ec_cnt += 1

                elif idx > 2:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_ec_list = arr[level_idx]
                        protein_ec_list = protein_ec_list.split(',')
                        for ec in protein_ec_list:
                            if len(ec) > 0:
                                protein_labels.append(ec_annotations[ec])
                                ec_num[ec] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(ec_annotations)
        self.weights = np.zeros((ec_cnt,), dtype=np.float32)
        for ec, idx in ec_annotations.items():
            self.weights[idx] = len(self.labels)/ec_num[ec]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_name, pos, ori, amino = self.data[idx]
        label = np.zeros((self.num_classes,)).astype(np.float32)
        if len(self.labels[protein_name]) > 0:
            label[self.labels[protein_name]] = 1.0

        if self.split == "train":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos))    # [num_nodes, num_dimensions]

        return data

class ECDataset_Struct(Dataset):

    def __init__(self, root='/tmp/protein-data/ec', percent=30, random_seed=0, split='train', ca_path="/usr/data/protein/ec"):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split
        self.root = root

        # Get the paths.
        npy_dir = os.path.join(ca_path, 'coordinates')
        fasta_file = os.path.join(ca_path, split+'.fasta')

        # Mask test set.
        test_set = set()
        if split == "test":
            with open(os.path.join(ca_path, "nrPDB-EC_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        self.files_name = {}
        file_path = root + '/'+split + '/'
        files = os.listdir(file_path)
        for file in files:
            file = file.rstrip('.pdb')  
            tmp = file.split('_')
            self.files_name[tmp[0]] = tmp[1]

        self.data = []
        self.name = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))
            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            ori = orientation(pos)
            self.data.append((protein_name, pos, ori, amino_ids.astype(int)))
            self.name.append(protein_name)

        level_idx = 1
        ec_cnt = 0
        ec_num = {}
        ec_annotations = {}
        self.labels = {}

        with open(os.path.join(ca_path, 'nrPDB-EC_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1:
                    arr = line.rstrip().split('\t')
                    for ec in arr:
                        ec_annotations[ec] = ec_cnt
                        ec_num[ec] = 0
                        ec_cnt += 1

                elif idx > 2:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_ec_list = arr[level_idx]
                        protein_ec_list = protein_ec_list.split(',')
                        for ec in protein_ec_list:
                            if len(ec) > 0:
                                protein_labels.append(ec_annotations[ec])
                                ec_num[ec] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(ec_annotations)
        self.weights = np.zeros((ec_cnt,), dtype=np.float32)
        for ec, idx in ec_annotations.items():
            self.weights[idx] = len(self.labels)/ec_num[ec]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_name, pos, ori, amino = self.data[idx]
        label = np.zeros((self.num_classes,)).astype(np.float32)
        if len(self.labels[protein_name]) > 0:
            label[self.labels[protein_name]] = 1.
            
        file_name = self.name[idx]
        try:
            id = self.files_name[file_name]
            file_path = self.root+"/"+self.split+"/"+file_name+'_'+id+".pdb"
            side_chain_embs, bb_embs = self.protein_to_graph(file_path, amino, pos)
        except: 
            side_chain_embs = torch.zeros((len(amino), 8))
            bb_embs = torch.zeros((len(amino),6))

        if self.split == "train":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        assert bb_embs.shape[0] == amino.shape[0] 
        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos),
                    side_chain_embs = side_chain_embs,
                    bb_embs = bb_embs
                    )    # [num_nodes, num_dimensions]

        return data
    
    def _normalize(self,tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def get_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names, b'N')
        mask_ca = np.char.equal(atom_names, b'CA')
        mask_c = np.char.equal(atom_names, b'C')
        mask_cb = np.char.equal(atom_names, b'CB')
        mask_g = np.char.equal(atom_names, b'CG') | np.char.equal(atom_names, b'SG') | np.char.equal(atom_names, b'OG') | np.char.equal(atom_names, b'CG1') | np.char.equal(atom_names, b'OG1')
        mask_d = np.char.equal(atom_names, b'CD') | np.char.equal(atom_names, b'SD') | np.char.equal(atom_names, b'CD1') | np.char.equal(atom_names, b'OD1') | np.char.equal(atom_names, b'ND1')
        mask_e = np.char.equal(atom_names, b'CE') | np.char.equal(atom_names, b'NE') | np.char.equal(atom_names, b'OE1')
        mask_z = np.char.equal(atom_names, b'CZ') | np.char.equal(atom_names, b'NZ')
        mask_h = np.char.equal(atom_names, b'NH1')

        pos_n = np.full((len(amino_types),3),np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        # pos_n = self.ceterize(pos_n)
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types),3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types),3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        # pos_c = self.ceterize(pos_c)
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types),3),np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        # pos_cb = self.ceterize(pos_cb)
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types),3),np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        # pos_g = self.ceterize(pos_g)
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types),3),np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        # pos_d = self.ceterize(pos_d)
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types),3),np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        # pos_e = self.ceterize(pos_e)
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types),3),np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        # pos_z = self.ceterize(pos_z)
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types),3),np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        # pos_h = self.ceterize(pos_h)
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h


    def side_chain_embs(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(self.compute_diherals(v1, v2, v3),1)
        angle2 = torch.unsqueeze(self.compute_diherals(v2, v3, v4),1)
        angle3 = torch.unsqueeze(self.compute_diherals(v3, v4, v5),1)
        angle4 = torch.unsqueeze(self.compute_diherals(v4, v5, v6),1)
        angle5 = torch.unsqueeze(self.compute_diherals(v5, v6, v7),1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4),1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)),1)
        
        return side_chain_embs

    
    def bb_embs(self, X):   
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6 
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self.compute_diherals(u0, u1, u2)
        
        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2]) 
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features

    
    def compute_diherals(self, v1, v2, v3):
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion
    
    
    def protein_to_graph(self, pFilePath, amino, pos_ca_ori):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with open(pFilePath) as f:
                pdb_structure = pdb.PDBParser().get_structure('protein', f)

            pos = np.full((len(amino),3),np.nan)
            pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = pos, pos, pos, pos, pos, pos, pos, pos, pos

            for residue in pdb_structure.get_residues():
                for atom in residue.get_atoms():
                    atom_name = atom.get_name()
                    coord = atom.get_coord()
                    if atom_name == 'N':
                        pos_n[i] = coord
                    elif atom_name == 'CA':
                        pos_ca[i] = coord
                    elif atom_name == 'C':
                        pos_c[i] = coord
                    elif atom_name == 'CB':
                        pos_cb[i] = coord
                    elif atom_name in ['CG', 'SG', 'OG', 'CG1', 'OG1']:
                        pos_g[i] = coord
                    elif atom_name in ['CD', 'SD', 'CD1', 'OD1', 'ND1']:
                        pos_d[i] = coord
                    elif atom_name in ['CE', 'NE','OE1']:
                        pos_e[i] = coord
                    elif atom_name in ['CZ', 'NZ']:
                        pos_z[i] = coord
                    elif atom_name == 'NH1':
                        pos_h[i] = coord
                    
            pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = torch.FloatTensor(pos_n), torch.FloatTensor(pos_ca),torch.FloatTensor(pos_c), torch.FloatTensor(pos_cb),torch.FloatTensor(pos_g), torch.FloatTensor(pos_d), torch.FloatTensor(pos_e), torch.FloatTensor(pos_z), torch.FloatTensor(pos_h)
            # if data only contain pos_ca, we set the position of C and N as the position of CA
            pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
            pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]


            side_chain_embs = self.side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
            side_chain_embs[torch.isnan(side_chain_embs)] = 0

            # three backbone torsion angles
            bb_embs = self.bb_embs(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
            bb_embs[torch.isnan(bb_embs)] = 0

            return side_chain_embs, bb_embs

class ECDataset_ESM(Dataset):

    def __init__(self, root='/tmp/protein-data/ec', percent=30, random_seed=0, split='train', ca_path="/usr/data/protein/ec"):

        self.random_state = np.random.RandomState(random_seed)
        self.split = split
        self.root = root

        # Get the paths.
        npy_dir = os.path.join(ca_path, 'coordinates')
        fasta_file = os.path.join(ca_path, split+'.fasta')
        self.ca_path = ca_path

        # Mask test set.
        test_set = set()
        if split == "test":
            with open(os.path.join(ca_path, "nrPDB-EC_test.csv"), 'r') as f:
                head = True
                for line in f:
                    if head:
                        head = False
                        continue
                    arr = line.rstrip().split(',')
                    if percent == 30 and arr[1] == '1':
                        test_set.add(arr[0])
                    elif percent == 40 and arr[2] == '1':
                        test_set.add(arr[0])
                    elif percent == 50 and arr[3] == '1':
                        test_set.add(arr[0])
                    elif percent == 70 and arr[4] == '1':
                        test_set.add(arr[0])
                    elif percent == 95 and arr[5] == '1':
                        test_set.add(arr[0])
                    else:
                        pass

        # Load the fasta file.
        protein_seqs = []
        with open(fasta_file, 'r') as f:
            protein_name = ''
            for line in f:
                if line.startswith('>'):
                    protein_name = line.rstrip()[1:]
                else:
                    if split == "test" and (protein_name not in test_set):
                        continue
                    amino_chain = line.rstrip()
                    amino_ids = []
                    for amino in amino_chain:
                        amino_ids.append(aa_to_id[amino])
                    protein_seqs.append((protein_name, np.array(amino_ids)))

        self.files_name = {}
        file_path = root + '/'+split + '/'
        files = os.listdir(file_path)
        for file in files:
            file = file.rstrip('.pdb')  
            tmp = file.split('_')
            self.files_name[tmp[0]] = tmp[1]

        self.data = []
        self.name = []
        for protein_name, amino_ids in protein_seqs:
            pos = np.load(os.path.join(npy_dir, protein_name+".npy"))
            center = np.sum(a=pos, axis=0, keepdims=True)/pos.shape[0]
            pos = pos - center
            ori = orientation(pos)
            self.data.append((protein_name, pos, ori, amino_ids.astype(int)))
            self.name.append(protein_name)

        level_idx = 1
        ec_cnt = 0
        ec_num = {}
        ec_annotations = {}
        self.labels = {}

        with open(os.path.join(ca_path, 'nrPDB-EC_annot.tsv'), 'r') as f:
            for idx, line in enumerate(f):
                if idx == 1:
                    arr = line.rstrip().split('\t')
                    for ec in arr:
                        ec_annotations[ec] = ec_cnt
                        ec_num[ec] = 0
                        ec_cnt += 1

                elif idx > 2:
                    arr = line.rstrip().split('\t')
                    protein_labels = []
                    if len(arr) > level_idx:
                        protein_ec_list = arr[level_idx]
                        protein_ec_list = protein_ec_list.split(',')
                        for ec in protein_ec_list:
                            if len(ec) > 0:
                                protein_labels.append(ec_annotations[ec])
                                ec_num[ec] += 1
                    self.labels[arr[0]] = np.array(protein_labels)

        self.num_classes = len(ec_annotations)
        self.weights = np.zeros((ec_cnt,), dtype=np.float32)
        for ec, idx in ec_annotations.items():
            self.weights[idx] = len(self.labels)/ec_num[ec]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        protein_name, pos, ori, amino = self.data[idx]
        label = np.zeros((self.num_classes,)).astype(np.float32)
        if len(self.labels[protein_name]) > 0:
            label[self.labels[protein_name]] = 1.
            
        file_name = self.name[idx]
        try:
            id = self.files_name[file_name]
            file_path = self.root+"/"+self.split+"/"+file_name+'_'+id+".pdb"
            side_chain_embs, bb_embs = self.protein_to_graph(file_path, amino, pos)
        except: 
            side_chain_embs = torch.zeros((len(amino), 8))
            bb_embs = torch.zeros((len(amino),6))

        if self.split == "train":
            pos = pos + self.random_state.normal(0.0, 0.05, pos.shape)

        pos = pos.astype(dtype=np.float32)
        ori = ori.astype(dtype=np.float32)
        seq = np.expand_dims(a=np.arange(pos.shape[0]), axis=1).astype(dtype=np.float32)

        esm_path = self.ca_path+"/"+'esm2_emb'+"/"+protein_name+".npy"
        esm_emb = np.load(esm_path)
        esm_emb = esm_emb[0][1:-1]

        assert bb_embs.shape[0] == amino.shape[0] 
        data = Data(x = torch.from_numpy(amino),    # [num_nodes, num_node_features]
                    edge_index = None,              # [2, num_edges]
                    edge_attr = None,               # [num_edges, num_edge_features]
                    y = label,
                    ori = torch.from_numpy(ori),    # [num_nodes, 3, 3]
                    seq = torch.from_numpy(seq),    # [num_nodes, 1]
                    pos = torch.from_numpy(pos),
                    side_chain_embs = side_chain_embs,
                    bb_embs = bb_embs,
                    esm_emb = torch.from_numpy(esm_emb)
                    )    # [num_nodes, num_dimensions]

        return data
    
    def _normalize(self,tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def get_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names, b'N')
        mask_ca = np.char.equal(atom_names, b'CA')
        mask_c = np.char.equal(atom_names, b'C')
        mask_cb = np.char.equal(atom_names, b'CB')
        mask_g = np.char.equal(atom_names, b'CG') | np.char.equal(atom_names, b'SG') | np.char.equal(atom_names, b'OG') | np.char.equal(atom_names, b'CG1') | np.char.equal(atom_names, b'OG1')
        mask_d = np.char.equal(atom_names, b'CD') | np.char.equal(atom_names, b'SD') | np.char.equal(atom_names, b'CD1') | np.char.equal(atom_names, b'OD1') | np.char.equal(atom_names, b'ND1')
        mask_e = np.char.equal(atom_names, b'CE') | np.char.equal(atom_names, b'NE') | np.char.equal(atom_names, b'OE1')
        mask_z = np.char.equal(atom_names, b'CZ') | np.char.equal(atom_names, b'NZ')
        mask_h = np.char.equal(atom_names, b'NH1')

        pos_n = np.full((len(amino_types),3),np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        # pos_n = self.ceterize(pos_n)
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types),3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types),3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        # pos_c = self.ceterize(pos_c)
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types),3),np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        # pos_cb = self.ceterize(pos_cb)
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types),3),np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        # pos_g = self.ceterize(pos_g)
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types),3),np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        # pos_d = self.ceterize(pos_d)
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types),3),np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        # pos_e = self.ceterize(pos_e)
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types),3),np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        # pos_z = self.ceterize(pos_z)
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types),3),np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        # pos_h = self.ceterize(pos_h)
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h


    def side_chain_embs(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(self.compute_diherals(v1, v2, v3),1)
        angle2 = torch.unsqueeze(self.compute_diherals(v2, v3, v4),1)
        angle3 = torch.unsqueeze(self.compute_diherals(v3, v4, v5),1)
        angle4 = torch.unsqueeze(self.compute_diherals(v4, v5, v6),1)
        angle5 = torch.unsqueeze(self.compute_diherals(v5, v6, v7),1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4),1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)),1)
        
        return side_chain_embs

    
    def bb_embs(self, X):   
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6 
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self.compute_diherals(u0, u1, u2)
        
        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2]) 
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features

    
    def compute_diherals(self, v1, v2, v3):
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion
    
    
    def protein_to_graph(self, pFilePath, amino, pos_ca_ori):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with open(pFilePath) as f:
                pdb_structure = pdb.PDBParser().get_structure('protein', f)

            pos = np.full((len(amino),3),np.nan)
            pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = pos, pos, pos, pos, pos, pos, pos, pos, pos

            for residue in pdb_structure.get_residues():
                for atom in residue.get_atoms():
                    atom_name = atom.get_name()
                    coord = atom.get_coord()
                    if atom_name == 'N':
                        pos_n[i] = coord
                    elif atom_name == 'CA':
                        pos_ca[i] = coord
                    elif atom_name == 'C':
                        pos_c[i] = coord
                    elif atom_name == 'CB':
                        pos_cb[i] = coord
                    elif atom_name in ['CG', 'SG', 'OG', 'CG1', 'OG1']:
                        pos_g[i] = coord
                    elif atom_name in ['CD', 'SD', 'CD1', 'OD1', 'ND1']:
                        pos_d[i] = coord
                    elif atom_name in ['CE', 'NE','OE1']:
                        pos_e[i] = coord
                    elif atom_name in ['CZ', 'NZ']:
                        pos_z[i] = coord
                    elif atom_name == 'NH1':
                        pos_h[i] = coord
                    
            pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = torch.FloatTensor(pos_n), torch.FloatTensor(pos_ca),torch.FloatTensor(pos_c), torch.FloatTensor(pos_cb),torch.FloatTensor(pos_g), torch.FloatTensor(pos_d), torch.FloatTensor(pos_e), torch.FloatTensor(pos_z), torch.FloatTensor(pos_h)
            # if data only contain pos_ca, we set the position of C and N as the position of CA
            pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
            pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]


            side_chain_embs = self.side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
            side_chain_embs[torch.isnan(side_chain_embs)] = 0

            # three backbone torsion angles
            bb_embs = self.bb_embs(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
            bb_embs[torch.isnan(bb_embs)] = 0

            return side_chain_embs, bb_embs


