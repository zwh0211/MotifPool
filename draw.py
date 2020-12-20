import torch_geometric
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/NCI1',name='NCI1')
print(dataset[0])
print(dataset[0].x)
for g,i in enumerate(dataset[0].x):
    print(i,g)
print(dataset[0].y)