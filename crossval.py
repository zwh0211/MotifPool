import sys
import argparse

import numpy as np
import torch
import torch_geometric
from torch_geometric.datasets import TUDataset

import sklearn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import ParameterGrid


import skorch
import tqdm

import utils

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='MotifPool',
                        help='Select pooling model for validation.')
    parser.add_argument('-d', '--dataset', type=str, default='NCI1',
                        help="Dataset used for validation from TUDataset.")
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help='Maximum number of epochs for training.')
    parser.add_argument('-N', type=int, default=5,
                        help='Parameter N for MotifPool module. See paper for more details.')
    parser.add_argument('-k', type=int, default=4,
                        help='Parameter k for MotifPool module. See paper for more details.')
    parser.add_argument('--seed', type=int, default=114514,
                        help='Global random seed.')
    parser.add_argument('--num_folds', type=int, default=10,
                        help='Number for n-folds validation.')
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    numpy.random.seed(seed)

    # Setting device
    device = ''
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = 'cuda'
    else:
        device = 'cpu'

    # Processing dataset
    dataset = TUDataset(root='/tmp/'+args.dataset, name=args.dataset)

    # Needs to be done(1)
    '''
    if dataset.data.x is None:
        dataset=utils.AddFeatures(dataset)
    '''

    ind_X = np.arange(len(dataset)).reshape((-1, 1))
    y = dataset.data.y.numpy()

    skf = StratifiedKFold(n_splits=args.num_folds,
                          shuffle=True, random_state=seed)
    skf_pbar = tqdm(list(skf.split(ind_X, y)), leave=True,
                    desc='n-folds validation process:')
