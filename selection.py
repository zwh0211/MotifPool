import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch.utils.data.dataloader import default_collate

import torch_sparse
import torch_cluster
import torch_spline_conv
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, JumpingKnowledge
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.datasets import TUDataset, qm9
from torch_geometric.utils import erdos_renyi_graph

from itertools import permutations
from math import factorial
import random

from search import count_motif
from enumerate_motif import rec_enumerate_motif, enumerate_motif, compute_ratio, post_process
from enumerate_motif import max_dist


def MotifSelect(dataset, N, k, num_sampling=50):
    '''
    Args:

    '''
    sampled = random.sample(range(len(dataset)), num_sampling)
    motifs = []
    scores = []
    for n in range(3, N+1):

        cur = rec_enumerate_motif(n, 0)
        for m in cur:
            dis = max_dist(m, n)
            ratio = compute_ratio(m, n)
            m = post_process(m)
            cnt = 0
            cnt_erdos = 0
            for g in sampled:
                graph = dataset[g]
                num_nodes = graph.num_nodes
                num_edges = graph.num_edges
                prob = num_edges/((num_nodes-1)*num_nodes)
                erdos_renyi = erdos_renyi_graph(num_nodes, prob)
                cnt_erdos += count_motif(erdos_renyi, m, num_nodes)
                cnt += count_motif(graph.edge_index, m, graph.num_nodes)
            if cnt == 0:
                continue

            score = ratio*n*n*(cnt/cnt_erdos)*(n-1-dis)
            print(m, ratio, cnt, cnt_erdos, cnt/cnt_erdos, dis, score)
            motifs.append((m, score))

    print(sorted(motifs, key=lambda m: m[1]))


NCI1Dataset = TUDataset(root='/tmp/NCI1', name='NCI1')
DD = TUDataset(root='/tmp/DD', name='DD')
Proteins = TUDataset(root='tmp/PROTEINS', name='PROTEINS')
imdb = TUDataset(root='tmp/IMDB-BINARY', name='IMDB-BINARY')
MotifSelect(DD, 4, 2)
