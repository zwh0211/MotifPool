import torch
import numpy as np
from torch_geometric.utils import degree
import math
from itertools import permutations


def get_adjancency_list(edge_index,graph_n):
    ''' 
    Transforms edge_index(sparse-COO form adjacency matrix of graph)
    into an adjacency list.

    Args:
        edge_index(numpy.ndarray): sparse-COO formed adjacency 
            matrix of the inpnt graph.
        graph_n(int): size of node set of the input graph. 

    Returns(adj_list):
        adj_list(dict): adjacency list of the input graph.

    '''

    n = graph_n
    m = edge_index.shape[1]

    adj_list = {}

    for i in range(n):
        adj_list[i] = []

    for i in range(m):
        x = edge_index[0][i]
        y = edge_index[1][i]
        adj_list[x].append(y)

    return adj_list


def AddFeatures(Dataset):
    if Dataset.data.x is not None:
        return Dataset

    max_degree = 0.
    degrees = []
    slices = [0]

    for data in Dataset:
        degrees.append(degree(data.edge_index[0], data.num_nodes, torch.float))
        max_degree = max(max_degree, degrees[-1].max().item())
        slices.append(data.num_nodes)

    Dataset.data.x = torch.cat(degrees, dim=0).div_(max_degree).view(-1, 1)
    Dataset.slices['x'] = torch.tensor(
        slices, dtype=torch.long, device=Dataset.data.x.device).cumsum(0)

    return Dataset


def weightMatToList(Mat):
    edge_index = []
    edge_attr = []
    tmp = 0
    for i in range(Mat.shape[0]):
        for j in range(Mat.shape[0]):
            if Mat[i][j] != 0:
                edge_index.append([i, j])
                edge_attr.append(Mat[i][j])
                tmp += 1
    # print(Mat.shape[0],tmp)

    return edge_index, edge_attr


def motifListToMat(m, numNodes):

    mSize = len(m)
    motifMat = []
    alloVec = []
    for i in range(numNodes):
        tmp = []
        if i in m:
            alloVec.append(1)
        else:
            alloVec.append(0)
        for j in range(mSize):
            if i == m[j]:
                tmp.append(1)
            else:
                tmp.append(0)
        motifMat.append(tmp)

    return motifMat, alloVec


def edgeIdxAttr2WeightMat(edge_index, edge_attr, numNodes):
    res = np.zeros((numNodes, numNodes))
    if edge_attr is not None:
        for i, edge in enumerate(edge_index):
            res[edge[0]][edge[1]] = edge_attr[i]
    else:
        for i, edge in enumerate(edge_index):
            res[edge[0]][edge[1]] = 1

    return torch.Tensor(res)


def maxDistance(m):
    if m == 'K7':
        return 1

    if m == 'K6':
        return 1

    if m == 'K5':
        return 1

    if m == 'C5':
        return 2

    if m == 'K4':
        return 1

    if m == 'V4':
        return 2

    if m == 'C4':
        return 2

    if m == 'K3':
        return 1

    if m == 'K2':
        return 0

    if m == 'S3':
        return 2

    if m == 'S4':
        return 2

    if m == 'S5':
        return 2

    if m == 'C6':
        return 3


def ConnectSearch(Graph):
    Graph = Graph.numpy()
    GraphNodeSize = Graph.max()+1
    GraphEdgeSize = Graph.shape[1]

    mark = [0 for _ in range(GraphNodeSize)]
    cnt = 0

    def visit(nd):
        if mark[nd] > 0:
            return
        mark[nd] = cnt
        for m in range(GraphEdgeSize):
            if Graph[0][m] == nd:
                visit(Graph[1][m])

    for nd in range(GraphNodeSize):
        if mark[nd] == 0:
            cnt += 1
            visit(nd)
    markcnt = [0 for _ in range(cnt+1)]
    for nd in range(GraphNodeSize):
        markcnt[mark[nd]] += 1
    return cnt


def AutomorphismCompute(Motif):
    # Compute the automorphism group of a given motif

    Motif = Motif.numpy()
    MotifNodeSize = Motif.max() + 1
    MotifEdgeSize = Motif.shape[1]
    MotifEdgeList = []
    for i in range(MotifEdgeSize):
        MotifEdgeList.append((Motif[0][i], Motif[1][i]))

    Aut = []
    for perm in list(permutations(range(MotifNodeSize))):
        flag = 1
        for edge in MotifEdgeList:
            if (perm[edge[0]], perm[edge[1]]) not in MotifEdgeList:
                flag = 0
        if flag:
            Aut.append(perm)

    # if len(Aut) == len(list(permutations(range(MotifNodeSize)))):
     #   IsComplete = 1
    return Aut


def PermCompute(Tup, Aut):
    # Compute the permutations of a given tuple defined in permutation group Aut

    res = []
    for perm in Aut:
        res.append(tuple([Tup[perm[i]] for i in range(len(Tup))]))

    return res


def IsComplete(MotifNodeSize, MotifEdgeSize):
    # Judge if the Motif is a complete graph

    return MotifNodeSize*(MotifNodeSize-1) == MotifEdgeSize


def GenerateMotifSet(MotifList):
    # Generate Motif tensor set with a motif label list
    MotifK2 = torch.tensor([[0, 1], [1, 0]],
                           dtype=torch.long)

    MotifK3 = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]],
                           dtype=torch.long)

    MotifK4 = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                            [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]],
                           dtype=torch.long)

    MotifK5 = torch.tensor(
        [[0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 4],
         [1, 0, 2, 0, 3, 0, 4, 0, 2, 1, 3, 1, 4, 1, 3, 2, 4, 2, 4, 3]],
        dtype=torch.long)

    MotifK6 = torch.tensor(
        [
            [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 1, 2, 1, 3, 1, 4,
                1, 5, 2, 3, 2, 4, 2, 5, 3, 4, 3, 5, 4, 5],
            [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 2, 1, 3, 1, 4,
                1, 5, 1, 3, 2, 4, 2, 5, 2, 4, 3, 5, 3, 5, 4]
        ], dtype=torch.long
    )

    MotifV4 = torch.tensor(
        [[0, 1, 0, 2, 0, 3, 1, 2, 1, 3], [1, 0, 2, 0, 3, 0, 2, 1, 3, 1]],
        dtype=torch.long)

    MotifC4 = torch.tensor([[0, 1, 0, 3, 1, 2, 2, 3], [1, 0, 3, 0, 2, 1, 3, 2]],
                           dtype=torch.long)

    MotifC5 = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]],
                           dtype=torch.long)

    MotifC6 = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0], [
        1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]], dtype=torch.long)

    MotifK7 = torch.tensor(
        [[
            0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 2, 3,
            2, 4, 2, 5, 2, 6, 3, 4, 3, 5, 3, 6, 4, 5, 4, 6, 5, 6
        ],
            [
            1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 3,
            2, 4, 2, 5, 2, 6, 2, 4, 3, 5, 3, 6, 3, 5, 4, 6, 4, 6, 5
        ]],
        dtype=torch.long)

    MotifS3 = torch.tensor(
        [[0, 1, 0, 2, 0, 3], [1, 0, 2, 0, 3, 0]], dtype=torch.long)

    MotifS4 = torch.tensor(
        [[0, 1, 0, 2, 0, 3, 0, 4], [1, 0, 2, 0, 3, 0, 4, 0]], dtype=torch.long)

    MotifS5 = torch.tensor([[0, 1, 0, 2, 0, 3, 0, 4, 0, 5], [
        1, 0, 2, 0, 3, 0, 4, 0, 5, 0]], dtype=torch.long)

    Motifset = []
    for m in MotifList:
        if m == 'K7':
            Motifset.append(MotifK7)

        if m == 'K6':
            Motifset.append(MotifK6)

        if m == 'K5':
            Motifset.append(MotifK5)

        if m == 'C5':
            Motifset.append(MotifC5)

        if m == 'K4':
            Motifset.append(MotifK4)

        if m == 'V4':
            Motifset.append(MotifV4)

        if m == 'C4':
            Motifset.append(MotifC4)

        if m == 'K3':
            Motifset.append(MotifK3)

        if m == 'K2':
            Motifset.append(MotifK2)

        if m == 'S3':
            Motifset.append(MotifS3)

        if m == 'S4':
            Motifset.append(MotifS4)

        if m == 'S5':
            Motifset.append(MotifS5)

        if m == 'C6':
            Motifset.append(MotifC6)

    return Motifset


def AddFeatures(Dataset):
    if Dataset.data.x is not None:
        return Dataset

    max_degree = 0.
    degrees = []
    slices = [0]

    for data in Dataset:
        degrees.append(degree(data.edge_index[0], data.num_nodes, torch.float))
        max_degree = max(max_degree, degrees[-1].max().item())
        slices.append(data.num_nodes)

    Dataset.data.x = torch.cat(degrees, dim=0).div_(max_degree).view(-1, 1)
    Dataset.slices['x'] = torch.tensor(
        slices, dtype=torch.long, device=Dataset.data.x.device).cumsum(0)

    return Dataset

    for data0 in Dataset:
        xs = [0]*data0.num_nodes
        for e in data0.edge_index:
            xs[e[0]] += 1
            xs[e[1]] += 1
        data0.x = torch.tensor(xs)

    return Dataset
