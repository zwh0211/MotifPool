import sys
import argparse

import numpy as np
import torch
from torch.nn.modules.loss import NLLLoss
from torch.optim import Adam

import torch_geometric
from torch_geometric.datasets import TUDataset

import sklearn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import datasets


import skorch
from skorch.dataset import CVSplit
from skorch.helper import predefined_split
from skorch.dataset import Dataset

import tqdm

from skorch import NeuralNetClassifier
import utils

import models
from models import DiffPool,MinCutPool,Graclus
#from models import BaselineModel, TopKModel, SAGPoolModel, DiffPoolModel

class TestScoring:
    def __init__(self, test_dataset):
        self.test_dataset = test_dataset
        self.y_true = [y for _, y in test_dataset]
    
    def __call__(self, net, X=None, y=None):
        y_pred = net.predict(self.test_dataset)

        return accuracy_score(self.y_true, y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='MinCutPool',
                        help='Select pooling model for validation.')
    parser.add_argument('-d', '--dataset', type=str, default='MUTAG',
                        help="Dataset used for validation from TUDataset.")
    parser.add_argument('--max_epochs', type=int, default=300,
                        help='Maximum number of epochs for training.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden channels.')
    parser.add_argument('--num_blocks', type=int, default=2,
                        help='Number of convolutional blocks.')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay.')
    parser.add_argument('--global_readout', type=bool, default=True,
                        help='Use global readout function or not.')
    parser.add_argument('--global_pool_op', type=str, default='add',
                        help='Global readout operation.')
    parser.add_argument('-b', '--batch_size', type=int, default=-1,
                        help='Batch size.')
    parser.add_argument('-N', type=int, default=5,
                        help='Parameter N for MotifPool module. See paper for more details.')
    parser.add_argument('-k', type=int, default=4,
                        help='Parameter k for MotifPool module. See paper for more details.')
    parser.add_argument('--seed', type=int, default=114514,
                        help='Global random seed.')
    parser.add_argument('--num_folds', type=int, default=10,
                        help='Number for n-folds validation.')
    parser.add_argument('--GPUIdx',type=int, default=None,
                        help='which GPU should run the code')
    parser.add_argument("--motifList", default=None,
                        type=str, help="motif list,default:'K6,K5,K4,K3,K2'")

    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setting device
    device = ''
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = 'cuda'
        if args.GPUIdx is not None:
            torch.cuda.set_device(args.GPUIdx)
    else:
        device = 'cpu'

    # Processing dataset
    dataset = TUDataset(root='/tmp/'+args.dataset, name=args.dataset)

    
    if dataset.data.x is None:
        dataset=utils.AddFeatures(dataset)
    dataset.data.edge_attr = None
    
    MotifList = None
    if args.motifList is not None:
        args.motifList = str(args.motifList).upper()
        MotifList = args.motifList.split(',')
    if args.model == 'MotifPool':
        pass
        #Model = MotifPoolModel
    elif args.model == 'Baseline':
        Model = BaselineModel
    elif args.model == 'TopK':
        Model = TopKModel
    elif args.model == 'SAGPool':
        Model = SAGPoolModel
    elif args.model == 'DiffPool':
        Model = DiffPool

    elif args.model == 'MinCutPool':
        Model = MinCutPool
    elif args.model =='graclus':
        Model = Graclus
    else:
        print('no such model')
        exit(1)

    ind_X = np.arange(len(dataset)).reshape((-1, 1))
    y = dataset.data.y.numpy()
    X = np.arange(len(dataset)).reshape((-1, 1))
    cvs = StratifiedShuffleSplit(
        test_size=0.1, n_splits=10, random_state=42)
    sss_split = 1./9
    results = []
    test_acc = []

    for out_iter, (train_idx, test_idx) in enumerate(cvs.split(X,y)):
        train_X = X[train_idx]
        train_y = y[train_idx]
        test_X = X[test_idx]
        test_y = y[test_idx]
        #in_sss = StratifiedShuffleSplit(n_splits=1, test_size=sss_split, random_state=42)
        #train_idx, val_idx = next(in_sss.split(train_X,train_y))

        #val_X = train_X[val_idx]
        #val_y = train_y[val_idx]
        #train_X = train_X[train_idx]
        #train_y = train_y[train_idx]

        train_ds = Dataset(train_X,train_y)
        test_ds = Dataset(test_X,test_y)

        net = NeuralNetClassifier(
            module=Model,
            module__dataset=dataset,
            module__hidden=args.hidden,
            #module__num_blocks=args.num_blocks,
            module__dropout=args.dropout,
            #module__global_readout=args.global_readout,
            module__global_pool_op=args.global_pool_op,
            module__device=device,
            #module__motif_list=MotifList,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            criterion=models.PoolLoss if args.model == 'DiffPool' or 'MinCutPool' else NLLLoss,
            optimizer=Adam,
            optimizer__weight_decay=args.weight_decay,
            iterator_train__shuffle=True,
            iterator_train__num_workers=0,
            iterator_valid__num_workers=0,
            train_split=predefined_split(test_ds),
            device=device,
        )

        net.fit(train_X,train_y)

        res = max(net.history[:,'valid_acc'])
        print(out_iter,'fold:', res)
        results.append(res)
    

    filename = 'NCI109res.csv'
    f = open(filename,'a')
    for item in results:
        f.write(str(item))
        f.write(',')
    f.write(str(np.mean(results)))
    if args.model == 'MotifPool':
        f.write(',')
        f.write(args.motifList)
    f.write(',')
    f.write(args.model)
    f.write('\n')
    print(results)
    print('mean:',np.mean(results))



    #print(X,y)
    #scores = cross_val_score(net, cv_split,y,cv=10,scoring='accuracy')
    #net.fit(ind_X, y)
