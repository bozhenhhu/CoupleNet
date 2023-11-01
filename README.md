We publicize partial codes before acceptance.

# CoupleNet
Learning Complete Protein Representation by Deep Coupling of Sequence and Structure


## File Specification
datasets.py gives some dataset functions to process data, including the amino acid types and the physicochemical properties of each residue, namely, a
steric parameter, hydrophobicity, volume, polarizability, isoelectric point, helix probability, and sheet probability. Besides, the geometric features are included.

test_go.py gets the f_max for a single protein.





## Installation

Install PyTorch 1.13.1:
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Install PyG, transformers:
```
pip install torch-geometric
pip install transformers
```

Install PyTorch Scatter and PyTorch Sparse:
```
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```



### License
The code is released under MIT License.


### Related Repos
1. [CDConv](https://github.com/hehefan/Continuous-Discrete-Convolution) &emsp; 2. [GearNet](https://github.com/DeepGraphLearning/GearNet) 

