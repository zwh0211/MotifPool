import torch
import numpy

import time
import torch_geometric
import torch_scatter
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import PPI
import argparse
from torch_geometric.datasets import TUDataset, qm9
from torch_geometric.utils import degree

from itertools import permutations

import utils
from utils import get_adjancency_list


class calc_args():
    '''
    Arguments for searching program.
    '''
    
    def __init__(
        self,
        graph_adj_list={},
        graph_n=0,
        graph_m=0,
        motif_adj_list={},
        motif_n=0,
        motif_m=0,
        motif_count=-1,
    ):

        self.graph_adj_list = graph_adj_list
        self.graph_n = graph_n
        self.graph_m = graph_m
        self.motif_adj_list = motif_adj_list
        self.motif_n = motif_n
        self.motif_m = motif_m
        self.motif_count = motif_count

    def update_graph(self, graph_adj_list, graph_n, graph_m):

        self.graph_adj_list = graph_adj_list
        self.graph_n = graph_n
        self.graph_m = graph_m

    def update_motif(self, motif_adj_list, motif_n, motif_m):

        self.motif_adj_list = motif_adj_list
        self.motif_n = motif_n
        self.motif_m = motif_m

    def update_count(self):
        self.motif_count += 1

global_args = calc_args()


def count_motif(edge_index, motif, graph_n):

    '''
    Count the number of different motif instances in input graph, using
    VF2 algorithm.

    Args:
        edge_index(torch.tensor): edge index of the given graph stored in sparse
            -COO matrix.
        motif(torch.tensor): edge index of the given motif stored in sparse
            -COO matrix.
        graph_n(int): number of nodes of the input graph.

    Returns: cnt
        cnt(int): number of different motif instances in input graph.
    
    '''

    global calc_args
    global_args.motif_count = 0

    edge_index = edge_index.to('cpu').numpy()
    graph_m = edge_index.shape[1]
    graph_adj_list = get_adjancency_list(edge_index, graph_n)

    global_args.update_graph(graph_adj_list, graph_n, graph_m)
    motif_found=set([])

    motif = motif.to('cpu').numpy()
    motif_n = motif.max() + 1
    motif_m = motif.shape[1]
    motif_adj_list = get_adjancency_list(motif, motif_n)

    global_args.update_motif(motif_adj_list, motif_n, motif_m)
    VF2(motif_found)

    return global_args.motif_count


def VF2(motif_found):
    '''
    Initiating VF2 algorithm.
    '''
    for i in range(global_args.graph_n):
        state = [[i], [0]]
        match(state, motif_found)


def match(state, motif_found):
    '''
    Recursive DFS for VF2 algorithm.
    '''

    graph_match = state[0]
    motif_match = state[1]

    adj_list = global_args.graph_adj_list

    if (len(motif_match) == global_args.motif_n):
        nodes=tuple(sorted(graph_match.copy()))
        if not nodes in motif_found:
            motif_found.add(nodes)
            global_args.update_count()
        return

    next_motif_node = len(motif_match)
    next_graph_nodes = []

    for i in graph_match:
        for j in adj_list[i]:
            if j not in graph_match:
                next_graph_nodes.append(j)

    next_graph_nodes = list(set(next_graph_nodes))
    adj_motif_nodes = set(global_args.motif_adj_list[next_motif_node])\
        & set(motif_match)

    for i in next_graph_nodes:
        adj_graph_nodes = set(adj_list[i]) & set(graph_match)

        if not len(adj_graph_nodes) == len(adj_motif_nodes):
            continue

        new_graph_match = graph_match + [i]

        flag = 1
        for j_g in adj_graph_nodes:
            j_m = motif_match[graph_match.index(j_g)]
            if j_m not in adj_motif_nodes:
                flag = 0
                break

        if flag:
            new_motif_match = motif_match + [next_motif_node]
            new_state = [new_graph_match, new_motif_match]
            match(new_state, motif_found)

    return

'''
NCI1Dataset = TUDataset(root='/tmp/NCI1', name='NCI1')

MotifK3 = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]],
                           dtype=torch.long)

MotifS3 = torch.tensor(
        [[0, 1, 0, 2, 0, 3], [1, 0, 2, 0, 3, 0]], dtype=torch.long)

MotifC5 = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]],
                           dtype=torch.long)

MotifC6 = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0], [
        1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]], dtype=torch.long)

a = NCI1Dataset[1]
print(count_motif(a.edge_index,MotifS3,a.num_nodes))
'''