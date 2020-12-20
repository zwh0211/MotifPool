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

from pool import simple_motif_cover
from utils import GenerateMotifSet

import tqdm

class MotifPool:
    '''
    MotifPool module.
    A MotifPool object will compute and store the coarsened representation
    of each graph in the dataset.


    Args:
        dataset(torch_geometric.dataset): dataset for pooling.
        assign_motiflist(bool): set True if the motiflist is pre-assigned.
            Defaults to False.
        motiflist(list, optional): a list of motifs for extraction. Only 
            available if assign_motiflist is set True. Defaults to an empty list.

            Every motif is represented by a torch.tensor sparse-COO matrix or
            a string.

            Example: Motif K3 can be stored by 
                torch.tensor([[0,1,0,2,1,2],[1,0,2,0,2,1]],dtype=torch.long)
                or 'K3'.
        N(int): maximum size of motifs, a parameter for the motif selection 
            process, valid if assign_motiflist is set True. Defaults to 5. 
            See paper for more details.
        k(int): number of motifs for extraction(except K2), a parameter for 
            the motif selection process, valid if assign_motiflist is set True.
            Defaults to 3. See paper for more details.
        overlap(int): maximum number of overlaped nodes between different 
            motifs. Defaults to 1. See paper for more details.
    '''

    def __init__(self, dataset,
                 assign_motiflist=True,
                 motiflist=[],
                 N=5,
                 k=3,
                 overlap=1):

        self.dataset = dataset

        if assign_motiflist == True:
            if isinstance(motiflist[0], str):
                self.motiflist = GenerateMotifSet(motiflist)
            else:
                self.motiflist = motiflist
        else:
            self.motiflist = MotifSelect(dataset, 10, N, k)

        self.cache_coarsened_graph = []
        self.cache_partition_vector = []

    def compute(self, mode='simple'):
        '''
        Compute the coarsened representation of each graph in self.dataset with
        motifs in self.motiflist.

        Args:
            mode(str): determine the output mode('simple' or 'advanced').
                When choosing mode 'simple', for each graph, the partition vector
                and edge_index of the coarsened graph will be stored as a tuple
                (list, torch.tensor). Used for simple feature aggregators like
                'sum' and 'add'.
                When choosing mode 'advanced', for each graph, the partition vector,
                the motif type vector(maps a extracted motif to a certain type),
                the edge weight matrix of each extracted motif-subgraph and the
                weighted edge_index of the coarsened graph will be calculated and
                stored. Used for the k-hop neighborhood aggregator mentioned in the 
                paper. 
        '''

        if len(self.cache_coarsened_graph) > 0:
            return

        if mode=='simple':
            for i in tqdm.trange(len(self.dataset)):
                graph=self.dataset[i]
                x, y = simple_motif_cover(graph.edge_index, self.motiflist,graph.num_nodes)
                self.cache_coarsened_graph.append(x)
                self.cache_partition_vector.append(y)

        return

    def get_representation(self, index):
        '''
        Find the coarsened representation of each graph with given indices in self.dataset.

        Args:
            index(list): list of indices in self.dataset.

        Returns:(edge_indices,partition_vectors):
            edge_indices(list),partition_vectors(list):
        '''
        edge_indices=[self.cache_coarsened_graph[i] for i in index]
        partition_vectors=[self.cache_partition_vector[i] for i in index]
        return edge_indices,partition_vectors


class HierarchyPool:
    '''
    Module for hierarchical pooling.
    This module will compute and store a hierarchy of graph pooling results.

    Args:
        dataset(torch_geometric.dataset): dataset for pooling.
        num_layers(int): number of pooling layers.
        arg_list(list): list of arguments of each pooling layer. Every element in the list
            will be a dictionary for argument-passing.

    '''

    def __init__(self, dataset, num_layers, arg_list):
        self.dataset = dataset
        self.num_layers = num_layers

        pass
