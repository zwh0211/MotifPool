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

from itertools import permutations
from math import factorial
import random

from search import count_motif


def connect_search(graph, n):
    '''
    Count the number of different connect components in input graph.

    Args:
        graph(list): edge_index of input graph, stored in sparse-COO matrix.
        n(int): node size of input graph.

    Return: cnt
        cnt(int): number of different connect components in input graph.
    '''
    m = len(graph)
    mark = [0 for _ in range(n)]
    cnt = 0

    def visit(d):
        if mark[d] > 0:
            return

        mark[d] = cnt
        for e in graph:
            if e[0] == d:
                visit(e[1])

    for d in range(n):
        if mark[d] == 0:
            cnt += 1
            visit(d)

    return cnt

def max_dist(motif,n):

    m = len(motif)
    mark = [100 for _ in range(n)]
    max_dists=[]
    cnt = 0

    def visit(d,l):
        if mark[d]<=l:
            return

        mark[d]=l

        for e in motif:
            if e[0] == d:
                visit(e[1],l+1)

    for d in range(n):
        for i in range(n):
            mark[i]=100
        
        visit(d,0)
        max_dists.append(max(mark))
    return max(max_dists)



def compute_aut(motif, n):
    '''
    Compute the automorphism group of a given graph.

    Args:
        motif(list): edge_index of input graph, stored in sparse-COO matrix.
        n(int): node size of input graph.

    Return: aut
        aut(list): automorphism group of input graph.
    '''
    m = len(motif)
    aut = []

    for perm in list(permutations(range(n))):
        flag = 1
        for edge in motif:
            if (perm[edge[0]], perm[edge[1]]) not in motif:
                flag = 0
        if flag:
            aut.append(perm)

    return aut

def compute_ratio(motif,n):
    return len(compute_aut(motif,n))/factorial(n)


def are_same(motif1, motif2, n):
    '''
    Judge if two edge_indices are different representations of one graph.

    Args:
        motif1, motif2(list): edge_indice of two graphs.
        n(int): node size of input graph.

    Return: res
        res(int): 1 for two edge_indices are different representations of one 
            graph, 0 for not.
    '''
    m1 = len(motif1)
    m2 = len(motif2)
    outer_flag1 = 0
    outer_flag2 = 0

    for perm in list(permutations(range(n))):
        inner_flag = 1

        for edge in motif1:
            if (perm[edge[0]], perm[edge[1]]) not in motif2:
                inner_flag = 0

        if inner_flag:
            outer_flag1 = 1
            break

    for perm in list(permutations(range(n))):
        inner_flag = 1

        for edge in motif2:
            if (perm[edge[0]], perm[edge[1]]) not in motif1:
                inner_flag = 0

        if inner_flag:
            outer_flag2 = 1
            break

    return outer_flag1 + outer_flag2 == 2


def dfs_enumerate_motif(N, bound=0):
    '''
    Enumerate all possible motifs with order N.

    Args:
        N(int): motif order.
        bound(float): lower bound of symmetry ratio of selected motifs.
            Defaults to 1/(2*N).
    
    Return: result
        result(list, with torch.tensor elements): list of edge_index of motifs
            obtained.
    '''
    '''
    if bound==0:
        bound=1/((N-1)*N)
    '''
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    m = len(edges)

    motif_found = []

    for i in range(1 << m):
        t = 1
        cur_edges = []
        for j in range(m):
            if t & i > 0:
                cur_edges.append(edges[j])
                cur_edges.append((edges[j][1], edges[j][0]))
            t = t << 1
        if not connect_search(cur_edges, N) == 1:
            continue

        flag = 1
        for motif in motif_found:
            if are_same(cur_edges, motif, N):
                flag = 0
                break

        if flag:
            ratio = len(compute_aut(cur_edges, N)) / factorial(N)
            if ratio > bound:
                motif_found.append(cur_edges.copy())
                print(cur_edges, ratio)

    print(len(motif_found))
    return motif_found


def rec_enumerate_motif(N, bound=0):
    '''
    Recursively enumerate all possible motifs with order N.

    Args:
        N(int): motif order.
        bound(float): lower bound of symmetry ratio of selected motifs.
            Defaults to 1/(2*N).
    
    Return: result
        result(list, with torch.tensor elements): list of edge_index of motifs
            obtained.
    '''
    '''
    if bound==0:
        bound=1/((N-1)*N)
    '''

    if N == 2:
        return [[(0, 1), (1, 0)]]

    else:
        motif_found = []
        pre = rec_enumerate_motif(N - 1)
        for m in pre:
            for i in range(1 << (N - 1)):
                cur_edges = m.copy()
                t = 1
                for j in range(N - 1):
                    if i & t > 0:
                        cur_edges.append((j, N - 1))
                        cur_edges.append((N - 1, j))
                    t = t << 1
                if not connect_search(cur_edges, N) == 1:
                    continue

                flag = 1
                for motif in motif_found:
                    if are_same(cur_edges, motif, N):
                        flag = 0
                        break

                if flag:
                    ratio = len(compute_aut(cur_edges, N)) / factorial(N)
                    if ratio > bound:
                        motif_found.append(cur_edges.copy())

    return motif_found.copy()


def post_process(motif):
    
    return torch.tensor(motif, dtype=torch.long, device='cpu').t()


def enumerate_motif(N,bound=0):
    return post_process(rec_enumerate_motif(N,bound))