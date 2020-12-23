from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
import torch_geometric
import torch_scatter
import torch_sparse
import torch_spline_conv
from torch.nn import BatchNorm1d, Linear, ModuleList
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.nn import (DenseSAGEConv, GCNConv, JumpingKnowledge,
                                SAGEConv, SAGPooling, TopKPooling,
                                dense_diff_pool, dense_mincut_pool, graclus)
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.transforms import ToDense

import pool
import utils
from cover import MotifPool


class ConvBlock(torch.nn.Module):
    '''
    Convolutional block with two graphSAGE modules followed by
     a pooling operation.

    Args:
        in_channels, hidden_channels, out_channels(int):
            number of in, hidden, out channels.
        mode(str): mode for JumpingKnowledge module aggregation,
            from 'cat', 'max', or 'lstm'. Defaults to 'cat'.
    '''

    def __init__(self, in_channels, hidden_channels, out_channels, mode='cat', dense=False):
        super(ConvBlock, self).__init__()

        module = DenseSAGEConv if dense else SAGEConv
        self.convs = ModuleList(
            [module(in_channels, hidden_channels),
             module(hidden_channels, out_channels)])

        self.jump = JumpingKnowledge(mode, hidden_channels, 2)
        self.linear = Linear(hidden_channels*2, out_channels)
        self.dense = dense

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        self.jump.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        xs = []
        for conv in self.convs:
            if self.dense:
                print(x.size())
                print(data.adj.size())
                print(data.mask.size())
                print(conv)
                x = conv(x, data.adj, mask=data.mask)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            xs.append(x)

        x = self.jump(xs)
        x = F.relu(self.linear(x))

        if self.dense and data.mask is not None:
            x = x * data.mask.unsqueeze(-1).type(x.dtype)

        return x


class DiffConvBlock(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DiffConvBlock, self).__init__()

        module = DenseSAGEConv
        self.conv1 = module(in_channels, hidden_channels)
        self.conv2 = module(hidden_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        print('x,adj,mask')
        print(data.x.size())
        print(data.adj.size())
        print(data.mask.size())
        print(self.conv1)
        x = self.conv1(data.x, data.adj, mask=data.mask)
        x = F.relu(x)
        print(x.size())
        print(data.adj.size())
        print(data.mask.size())
        print(self.conv2)
        x = self.conv2(x, data.adj, mask=data.mask)
        x = F.relu(x)
        return x


class BaselineModel(torch.nn.Module):
    '''
    Baseline model, with 1 or 2 convolutional blocks followed by a MLP.

    Args:
        dataset(torch_geometric.Dataset): input graph dataset.
        hidden(int): size of hidden channels.
        num_blocks(int): number of convolutional blocks. Defaults to 2.
        dropout(float, optional): dropout probability of MLP. Optional 
            and defaults to 0.3.
        global_readout(bool): set if using a global readout process to
            concatenate all outputs of each convolutional block. Defaults
            to True.
        global_pool_op(str): global pooling operation. Can be 'add', 'mean',
            or 'max'. Defaults to 'add'.
    '''

    def __init__(self, dataset, hidden, device,
                 num_blocks=2,
                 dropout=0.3,
                 global_readout=True,
                 global_pool_op='add',
                 dense=False,
                 motif_list=None,
                 ):
        super(BaselineModel, self).__init__()

        self.hidden = hidden
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.global_readout = global_readout
        self.device = device
        #print(dense)
        self.dense = dense
        self.dataset = DenseDataset(dataset) if self.dense else dataset
        self.motif_list = motif_list
        gps = global_pool_op if isinstance(
            global_pool_op, list) else [global_pool_op]

        if global_pool_op == 'add':
            self.global_pool_op = torch_geometric.nn.global_add_pool

        elif global_pool_op == 'max':
            self.global_pool_op = torch_geometric.nn.global_max_pool

        elif global_pool_op == 'avg':
            self.global_pool_op = torch_geometric.nn.global_mean_pool

        self.dense_global_pool_op = []

        for op in gps:
            if callable(op):
                self.dense_global_pool_op.append(op)
            elif op == 'add':
                self.dense_global_pool_op.append(
                    lambda x, _1, _2: torch.sum(x, dim=1))
            elif op == 'max' or op == 'min':
                self.dense_global_pool_op.append(
                    lambda x, _1, _2: getattr(torch, op)(x, dim=1)[0])
            else:
                self.dense_global_pool_op.append(
                    lambda x, _1, _2: getattr(torch, op)(x, dim=1))

        self.sparse_global_pool_op = [
            getattr(torch_geometric.nn, f'global_{op}_pool') for op in gps]

        self.blocks = torch.nn.ModuleList()
        #print(self.dense)
        self.blocks.append(ConvBlock(dataset.num_features,
                                     hidden, hidden, dense=self.dense))
        for _ in range(1, num_blocks):
            self.blocks.append(
                ConvBlock(hidden, hidden, hidden, dense=self.dense))

        self.out_dim = num_blocks*hidden
        self.bn = BatchNorm1d(self.out_dim)
        self.linear1 = Linear(self.out_dim, hidden)
        self.linear2 = Linear(hidden, dataset.num_classes)

    def collate(self, index):
        #print(self.dataset)

        return Batch.from_data_list(self.dataset[list(index.view(-1))]).to(self.device)

    def global_pool(self, data):
        if self.dense:
            #print(data)
            return [op(data.x) for op in self.dense_global_pool_op]
        return self.global_pool_op(data.x, data.batch, data.num_graphs)

    def reset_parameters(self):
        for block in self.blocks:
            block.reset_parameters()

        self.bn.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def pool(self, data, layer):
        return data

    def densify(self, data):
        data.x, data.mask = to_dense_batch(data.x, data.batch)
        data.adj = to_dense_adj(data.edge_index, data.batch)
        data.edge_index, data.edge_attr, data.batch = None, None, None

        return data

    def forward(self, index):
        batchsize = len(index)
        #print(index, '----------')
        data = self.collate(index)
        #print(self.dense)

        data.x = self.blocks[0](data)
        xs = [self.global_pool(data)]

        for layer, block in enumerate(self.blocks[1:], 1):

            data = self.pool(data, layer)

            data.x = block(data)
            data.to(self.device)
            xs.extend([self.global_pool(data)])

        x = torch.cat(xs, dim=1)
        x = self.bn(x)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear2(x)

        return F.softmax(x, dim=1)

    def __repr__(self):
        return self.__class__.__name__


class MotifPoolModel(BaselineModel):
    '''
    MotifPool model.

    Extension of the BaselineModel class. A MotifPool process is performed
    after every convolutional block. 

    Args:
        cover_func(callable): a cover function that generates a list of node 
            cover matrices for each graph in the dataset stored in self.cache.
            Used for pre-computation.
        aggregate_func(callable): a aggregate function that compute the pooling 
            process by coarsening the graph and aggregating the node features.


    '''

    def __init__(self, dataset, **kwargs):
        super(MotifPoolModel, self).__init__(dataset=dataset, **kwargs)
        self.cache_partition_vector = []
        self.cache_edge_indices = []
        print(self.motif_list)
        self.pooling_module = MotifPool(dataset, assign_motiflist=True,
                                        motiflist=self.motif_list)

        self.pooling_module.compute()

    def collate(self, index):
        self.cache_edge_indices, self.cache_partition_vector =\
            self.pooling_module.get_representation(index)
        return Batch.from_data_list(self.dataset[list(index.view(-1))]).to(self.device)

    def pool(self, dataset, layer):
        pooled = []

        for i, graph in enumerate(dataset.to_data_list()):
            new_edge_index = self.cache_edge_indices[i]
            partition_vector = self.cache_partition_vector[i]
            # a,b=pool.simple_motif_cover(graph.edge_index,utils.GenerateMotifSet(['C6','C5','S3', 'K2']))
            new_x = torch_scatter.scatter_add(graph.x, partition_vector, dim=0)
            new_graph = torch_geometric.data.Data(new_x, new_edge_index)
            pooled.append(new_graph)

        return torch_geometric.data.Batch.from_data_list(pooled)


class TopKModel(BaselineModel):
    '''
    TopKPooling model

    Extension of the BaselineModel class. A TopKPooling process is performed
    after every convolutional block. 
    Args:
        ratio (float, optional): The ratio of nodes in the coarsened graphs 
            with respect to to the number of nodes in the input graph.
            Defaults to 0.5.
    '''

    def __init__(self, dataset, ratio=0.5, **kwargs):
        super(TopKModel, self).__init__(dataset=dataset, **kwargs)
        self.ratio = ratio
        self.pooling_module = TopKPooling(self.hidden, ratio)

    def reset_parameters(self):
        super(TopKModel, self).reset_parameters()
        self.pooling_module.reset_parameters()

    def pool(self, data, layer):
        data.x, data.edge_index, data.edge_attr, data.batch, _, _ = self.pooling_module(
            data.x, data.edge_index, data.edge_attr, data.batch)

        return data


class SAGPoolModel(BaselineModel):
    '''
    SAGPool model

    Extension of the BaselineModel class. A TopKPooling process is performed
    after every convolutional block. 
    Args:
        ratio (float, optional): The ratio of nodes in the coarsened graphs 
            with respect to to the number of nodes in the input graph.
            Defaults to 0.5.
        gnn(str/torch.nn.Module optional): Convolutional model used in the pooling method.
            this arg can be a name string or a torch.geometric.nn module.
            Defaults to 'GCNConv'
    '''

    def __init__(self, dataset, ratio=0.5, gnn='GCNConv', **kwargs):
        super(SAGPoolModel, self).__init__(dataset=dataset, **kwargs)
        self.ratio = ratio
        self.gnn = getattr(torch_geometric.nn, gnn) if isinstance(
            gnn, str) else gnn

        self.pooling_module = SAGPooling(
            self.hidden, ratio=self.ratio, GNN=self.gnn)

    def reset_parameters(self):
        super(SAGPoolModel, self).reset_parameters()
        self.pooling_module.reset_parameters()

    def pool(self, data, layer):
        data.x, data.edge_index, data.edge_attr, data.batch, _, _ = self.pooling_module(
            data.x, data.edge_index, data.edge_attr, data.batch)

        return data


class DiffPoolModel(BaselineModel):
    """
    DiffPool Model

    Extends the BaselineModel class. This model performs pooling using DiffPool

    Args:
        ratio (float, optional): The ratio of nodes in the coarsened graphs 
            with respect to to the maximum number of nodes in the dataset.
            Defaults to 0.25.

    """

    def __init__(self, dataset, ratio=0.25, dense=True, **kwargs):
        super(DiffPoolModel, self).__init__(
            dataset=dataset, dense=True, **kwargs)

        num_nodes = self.dataset.max_nodes

        self.link_loss = 0.0
        self.ent_loss = 0.0
        self.ratio = ratio
        self.pool_blocks = torch.nn.ModuleList()

        for layer in range(self.num_blocks):
            num_nodes = ceil(ratio * float(num_nodes))
            self.pool_blocks.append(DiffConvBlock(
                self.hidden, self.hidden, num_nodes))

    def reset_parameters(self):
        super(DiffPoolMode, self).reset_parameters()

        for block in self.pool_blocks:
            block.reset_parameters()

    def collate(self, index):
        data = super(DiffPoolModel, self).collate(index)
        data.old_x = data.x

        return data

    def pool(self, data, layer):
        print(layer)
        print('old,now')
        print(data.old_x.size())
        print(data.x.size())
        #data.x, data.old_x = data.old_x, data.x

        print(self.pool_blocks[0])
        print(data.x.size())
        print(data.adj.size())
        print('--------------')
        s = self.pool_blocks[layer - 1](data)
        print('--------------')
        print(data.old_x.size())
        print(data.adj.size())
        print(s.size())
        data.x, data.adj, link_loss, ent_loss = dense_diff_pool(
            data.old_x, data.adj, s)
        data.old_x = data.x
        data.mask = None

        if layer == 1:
            self.link_loss = link_loss
            self.ent_loss = ent_loss

        else:
            self.link_loss += link_loss
            self.ent_loss += ent_loss

        return dataset

    # def forward(self, index):
     #   return super(DiffPoolModel, self).forward(index), self.link_loss, self.ent_loss


class PoolLoss(torch.nn.Module):
    """DiffPool Loss.

    Computes the llink and hentropy losses as described by Ying et al. in
        "Hierarchical Graph Representation Learning with Differentiable
        Pooling".

    Args:
        link_weight (float, optional): Weight applied to the link loss.
            Defaults to 1.
        ent_weight (float, optional): Weight applied to the entropy loss.
            Defaults to 1.
    """

    def __init__(self, link_weight=1., ent_weight=1., *args, **kwargs):
        super(PoolLoss, self).__init__()
        self.loss = torch.nn.modules.loss.NLLLoss(*args, **kwargs)
        self.link_weight = link_weight
        self.ent_weight = ent_weight

    def forward(self, input, target):
        output, link_loss, ent_loss = input
        output = torch.log(output)

        return self.loss.forward(output, target) \
            + self.link_weight*link_loss \
            + self.ent_weight*ent_loss


class DenseDataset(Dataset):
    """Dense Graphs Dataset.

    Args:
        data_list (list): list of graphs.
    """

    def __init__(self, data_list):
        super(DenseDataset, self).__init__("")

        self.data = Batch()
        self.max_nodes = max([data.num_nodes for data in data_list])
        to_dense = ToDense(self.max_nodes)
        dense_list = [to_dense(data) for data in data_list]

        if 'cover_index' in data_list[0]:
            self.max_clusters = max([data.num_clusters for data in data_list])

            for data in dense_list:
                data.cover_mask = torch.zeros(
                    self.max_clusters, dtype=torch.uint8)
                data.cover_mask[:data.num_clusters] = 1
                data.cover_index = torch.sparse_coo_tensor(
                    indices=data.cover_index,
                    values=torch.ones_like(data.cover_index[0]),
                    size=torch.Size([self.max_nodes, self.max_clusters]),
                    dtype=torch.float
                ).to_dense()

        for key in dense_list[0].keys:
            self.data[key] = default_collate([d[key] for d in dense_list])

    def len(self):
        if self.data.x is not None:
            return self.data.x.size(0)

        if 'adj' in self.data:
            return self.data.adj.size(0)

        return 0

    def get(self, idx):
        mask = self.data.mask[idx]
        max_nodes = mask.type(torch.uint8).argmax(-1).max().item() + 1
        out = Batch()

        for key, item in self.data('x', 'pos', 'mask'):
            out[key] = item[idx, :max_nodes]

        out.adj = self.data.adj[idx, :max_nodes, :max_nodes]

        if 'y' in self.data:
            out.y = self.data.y[idx]

        if 'cover_index' in self.data:
            cover_mask = self.data.cover_mask[idx]
            max_clusters = cover_mask.type(
                torch.uint8).argmax(-1).max().item() + 1
            out.cover_index = self.data.cover_index[idx,
                                                    :max_nodes, :max_clusters]
            out.cover_mask = cover_mask[:, :max_clusters]

        return out

    def index_select(self, idx):
        return self.get(idx)


class Graclus(BaselineModel):

    def __init__(self, dataset, **kwargs):
        super(Graclus, self).__init__(dataset=dataset, **kwargs)
        self.pool_op = torch_geometric.nn.avg_pool

    def pool(self, data, layer):
        cluster = graclus(data.edge_index)
        return self.pool_op(cluster, data)
