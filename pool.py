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

class pool_args():
    '''
    Arguments for pooling function.
    '''

    def __init__(self,
                 graph_adj_list={},
                 graph_n=0,
                 graph_m=0,
                 motif_adj_list={},
                 motif_n=0,
                 motif_m=0,
                 motif_index=0,
                 motif_count=-1,
                 ):

        self.graph_adj_list = graph_adj_list
        self.graph_n = graph_n
        self.graph_m = graph_m
        self.motif_adj_list = motif_adj_list
        self.motif_n = motif_n
        self.motif_m = motif_m
        self.motif_index = motif_index
        self.motif_count = motif_count
        self.is_clique = (motif_n*(motif_n-1) == motif_m)

    def update_graph(self, graph_adj_list, graph_n, graph_m):

        self.graph_adj_list = graph_adj_list
        self.graph_n = graph_n
        self.graph_m = graph_m

    def update_motif(self, motif_adj_list, motif_n, motif_m, motif_index):

        self.motif_adj_list = motif_adj_list
        self.motif_n = motif_n
        self.motif_m = motif_m
        self.motif_index = motif_index
        self.is_clique = (motif_n*(motif_n-1) == motif_m)

    def update_count(self):
        self.motif_count += 1


class pool_result():
    '''
    Contains pooling result.
    '''

    def __init__(self, graph_n):

        self.graph_n = graph_n
        self.coarsened_edge_index = []
        self.partition_vector = [[] for _ in range(graph_n)]

    def update_edge_index(self, x, y):
        self.coarsened_edge_index.append((x, y))
        self.coarsened_edge_index.append((y, x))

    def update_par_vector(self, x, motif_cnt):
        self.partition_vector[x].append(motif_cnt)

    def post_process(self):
        idx = list(set(self.coarsened_edge_index))
        self.coarsened_edge_index = torch.tensor([[a, b]for (a, b) in idx],
                                                 dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu').t()

        self.partition_vector = torch.tensor([i[0] for i in self.partition_vector],
                                             dtype=torch.int64, device='cuda' if torch.cuda.is_available() else 'cpu')

        return


global_args = pool_args()


def simple_motif_cover(edge_index, motif_list,graph_n):
    '''
    Calculate the motif cover of a graph with motifs in motif_list. This 
    version maps each node to a single motif, returns the partition vector
    and edge index of the coarsened graph.

    Args:
        edge_index(torch.tensor): edge index of the given graph stored in sparse
            -COO matrix.
        motif_list(list): a list of motifs for extraction. All motifs are represented
            by its edge_index stored in sparse-COO form with torch.tensor.
        graph_n(int): size of node set of the input graph. 

    Returns(coarsened_edge_index, partition_vector):
        coarsened_edge_index(torch.tensor): edge index of the coarsened graph.
        partition_vector(torch.tensor): partition vector that maps each node to a motif
            subgraph.

    '''
    
    global pool_args, result
    global_args.motif_count=-1

    edge_index = edge_index.to('cpu').numpy()
    graph_m = edge_index.shape[1]
    graph_adj_list = get_adjancency_list(edge_index,graph_n)

    result = pool_result(graph_n)

    global_args.update_graph(graph_adj_list, graph_n, graph_m)
    avail_node = set(range(global_args.graph_n))

    for motif_index, motif in enumerate(motif_list):
        motif = motif.to('cpu').numpy()
        motif_n = motif.max()+1
        motif_m = motif.shape[1]
        motif_adj_list = get_adjancency_list(motif,motif_n)

        global_args.update_motif(motif_adj_list, motif_n, motif_m, motif_index)
        VF2(avail_node)
    
    for nd in avail_node:
        global_args.update_count()
        result.update_par_vector(nd,global_args.motif_count)
    
    result.post_process()
    return result.coarsened_edge_index, result.partition_vector


def VF2(avail_node):
    '''
    Initiating VF2 algorithm.
    '''
    for i in range(global_args.graph_n):
        state = [[i], [0], 0]
        if i in avail_node:
            match(state, avail_node)


def match(state, avail_node):
    '''
    Recursive DFS for VF2 algorithm.
    '''

    graph_match = state[0]
    motif_match = state[1]
    overlap = state[2]

    if overlap > 1:
        return

    adj_list = global_args.graph_adj_list

    if(len(motif_match) == global_args.motif_n):
        cnt = 0
        for i in graph_match:
            if i not in avail_node:
                cnt += 1
            if cnt > 1:
                return

        global_args.update_count()
        count = global_args.motif_count

        for i in graph_match:
            result.update_par_vector(i, count)

            if i in avail_node:
                avail_node.remove(i)

            for neighbor in adj_list[i]:
                if len(result.partition_vector[neighbor]) > 0:
                    for m in result.partition_vector[neighbor]:
                        if not m == count:
                            result.update_edge_index(count, m)

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
        '''
        if global_args.is_clique and i < graph_match[-1]:
            continue
        '''
        adj_graph_nodes = set(adj_list[i]) & set(graph_match)

        if not len(adj_graph_nodes) == len(adj_motif_nodes):
            continue

        new_graph_match = graph_match+[i]

        flag = 1
        for j_g in adj_graph_nodes:
            j_m = motif_match[graph_match.index(j_g)]
            if j_m not in adj_motif_nodes:
                flag = 0
                break

        if flag:
            new_motif_match = motif_match+[next_motif_node]
            new_overlap = overlap

            if i not in avail_node:
                new_overlap += 1

            new_state = [new_graph_match, new_motif_match, new_overlap]
            match(new_state, avail_node)

    return

'''
EDataset = TUDataset(root='/tmp/NCI1', name='ENZYMES')
print(EDataset[423])

a = NCI1Dataset[63].edge_index
print(a)
b = utils.GenerateMotifSet(['C6','C5','S3', 'K2'])
print(utils.ConnectSearch(a))
p, q = simple_motif_cover(a, b)
print(p, q)
'''