"""
%load_ext autoreload
%autoreload 2
"""

from dnn import *
from common import *

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import math
import argparse

import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import time
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold
from sklearn import metrics
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import tensorflow as tf
from torch.optim.optimizer import Optimizer, required


'''
DATA_PATH = Path('/Volumes/T5/CHAMPS/tabular-features')
RESULT_PATH = Path('results')
N_FOLD = 10
EPOCH = 1000
BATCH_SIZE = 256
DEFAULT_LR = 1e-4
DEFAULT_DECAY = 5*1e-5
EARLY_STOPPING_ROUNDS = 50
DISP_INTERVAL = 10
CTYPE = '1JHN'
HIDDEN_DIM = 128
'''


def main():
    #%%
    X = pd.read_csv(DATA_PATH/'train'/f'{CTYPE}_full.csv', index_col=0)
    X = reduce_mem_usage(X)
    y = X['scalar_coupling_constant']
    X = X.drop(['scalar_coupling_constant'], axis=1)

    X_test = pd.read_csv(DATA_PATH/'test'/f'{CTYPE}_full.csv', index_col=0)
    X_test = reduce_mem_usage(X_test)


    #%%
    X = X.fillna(0)
    X_test = X_test.fillna(0)

    X_all = pd.concat([X, X_test])
    cat_features = []
    for col in X_all.columns:
        if col[-5:] == '_atom' or col in ['atom_A', 'atom_B']:
            cat_features.append(col)
    print(cat_features)
    for col in cat_features:
        print(col)
        X_all[col] = label_encode(X_all[col])

    print('dummie', X_all.shape)
    X_all = pd.get_dummies(X_all, columns=cat_features, drop_first=True, dummy_na=True)
    print('->', X_all.shape)

    X = X_all.iloc[:len(X)]
    X_test = X_all.iloc[len(X):]
    del X_all; gc.collect()


    index_train = X['id']
    groups = X['molecule_name']
    if CTYPE[:2] == '1J':
        X_t = X.drop(['atom_index_0','atom_index_1','id', 'type', 'molecule_name'],axis=1)
    elif CTYPE[:2] == '2J':
        X_t = X.drop(['atom_index_0','atom_index_1','atom_index_A','id', 'type', 'molecule_name'],axis=1)
    elif CTYPE[:2] == '3J':
        X_t = X.drop(['atom_index_0','atom_index_1','atom_index_A','atom_index_B','id','type', 'molecule_name'],axis=1)
    y_t = y

    index_test = X_test['id']
    if CTYPE[:2] == '1J':
        X_test_t = X_test.drop(['atom_index_0','atom_index_1','id', 'type', 'molecule_name'],axis=1)
    elif CTYPE[:2] == '2J':
        X_test_t = X_test.drop(['atom_index_0','atom_index_1','atom_index_A','id', 'type', 'molecule_name'],axis=1)
    elif CTYPE[:2] == '3J':
        X_test_t = X_test.drop(['atom_index_0','atom_index_1','atom_index_A','atom_index_B','id','type', 'molecule_name'],axis=1)

    sc = StandardScaler()
    X_t = sc.fit_transform(X_t)
    X_test_t = sc.transform(X_test_t)

    #%%
    folds = GroupKFold(n_splits=N_FOLD)
    folds.get_n_splits(X_t, y_t, groups)
    fold_split = folds.split(X_t, y_t, groups)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # for test set prediction
    X_test_t = torch.tensor(X_test_t, dtype=torch.float).to(device)
    test_ds = torch.utils.data.TensorDataset(X_test_t)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    #
    oof = np.zeros(len(X_t))
    prediction = np.zeros(len(X_test_t))
    avg_losses = []
    avg_val_losses = []

    for fold_i, (train_idx, valid_idx) in enumerate(fold_split):
        print(f'Fold {fold_i + 1} started at {time.ctime()}')

        # dataset
        X_train = torch.tensor(X_t[train_idx.astype(int)], dtype=torch.float).to(device)
        X_valid = torch.tensor(X_t[valid_idx.astype(int)], dtype=torch.float).to(device)
        y_train = torch.tensor(np.array(y_t)[train_idx.astype(int), np.newaxis],
                               dtype=torch.float).to(device)
        y_valid = torch.tensor(np.array(y_t)[valid_idx.astype(int), np.newaxis],
                               dtype=torch.float).to(device)
        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        valid_ds = torch.utils.data.TensorDataset(X_valid, y_valid)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)


        # define model each fold
        model = Simple_NN(X_train.shape[1], HIDDEN_DIM, activation=nn.LeakyReLU())
        model.to(device)

        # criterion = nn.L1Loss()
        criterion = nn.SmoothL1Loss()
        mae = nn.L1Loss()

        step_size = 5
        base_lr, max_lr = DEFAULT_LR, 5*DEFAULT_LR
    #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=max_lr)
        optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=max_lr)
        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                             step_size=step_size, mode='exp_range',
                             gamma=0.99994)

        early_stopping = EarlyStopping(patience=EARLY_STOPPING_ROUNDS, verbose=True)
        best_weight = {'epoch': None, 'state_dict': None}


        if torch.cuda.device_count() > 1:
            print('{} gpus found.'.format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)


        for epoch in range(EPOCH):
            start_time = time.time()
            model.train()
            avg_loss = 0.
            avg_mae = 0.

            # train
            for batch_i, (x, y) in enumerate(train_loader):
                y_pred = model(x)
                if scheduler:
                    scheduler.batch_step()
                loss = criterion(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss += loss.item() / len(train_loader)

            # valid
            model.eval()

            oof_fold = np.zeros(X_valid.size(0))
            prediction_fold = np.zeros(len(X_test_t))
            avg_val_loss = 0.

            for batch_i, (x, y) in enumerate(valid_loader):
                y_pred = model(x).detach()
                loss = criterion(y_pred, y)
                metric = mae(y_pred, y)
                avg_val_loss += loss.item() / len(valid_loader)
                avg_mae += metric.item() / len(valid_loader)

            elapsed_time = time.time() - start_time

            if early_stopping(avg_val_loss, model): # score updated
                print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t MAE={:.4f} \t time={:.2f}s'.format(
                    epoch + 1, EPOCH, avg_loss, avg_val_loss, avg_mae, elapsed_time))
                best_weight['epoch'] = epoch
                best_weight['state_dict'] = model.state_dict()
            if early_stopping.early_stop:
                print("Early stopping!")
                break

        avg_losses.append(avg_loss)
        avg_val_losses.append(avg_val_loss)

        # predict
        print('best epoch for fold {} is {}'.format(fold_i+1, best_weight['epoch']+1))
        model.load_state_dict(best_weight['state_dict'])
        for batch_i, (x, _) in enumerate(valid_loader):
            y_pred = model(x).detach()
            oof_fold[batch_i * BATCH_SIZE:(batch_i+1) * BATCH_SIZE] = y_pred.cpu().numpy()[:, 0]
        for batch_i, (x,) in enumerate(test_loader):
            y_pred = model(x).detach()
            prediction_fold[batch_i * BATCH_SIZE:(batch_i+1) * BATCH_SIZE] = y_pred.cpu().numpy()[:, 0]

        oof[valid_idx] = oof_fold
        prediction += prediction_fold / N_FOLD

    # results
    overall_mae = mean_absolute_error(oof, y_t.values)
    overall_logmae = np.log(overall_mae)

    print('Overall \t loss={:.4f} \t val_loss={:.4f} \t MAE={:.4f} \t logMAE={:.4f}'.format(
        np.average(avg_losses), np.average(avg_val_losses), overall_mae, overall_logmae))


    res = []
    res_dict = {
        'oof': oof,
        'prediction': prediction
    }
    res.append((CTYPE, res_dict))

    with open(f'{CTYPE}_DNN.pkl', 'wb') as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    #%%
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='~/kaggle-data/champs/tabular-features',
                        help="root path to data")
    parser.add_argument("--epoch", type=int, default=1000,
                        help="number of epochs")
    parser.add_argument("--fold", type=int, default=5,
                        help="number of folds")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="number of batch size")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="number of hidden_dim")
    parser.add_argument("--types", type=str, default='1JHN',
                        help="which type to train (default: 1JHN)")


    opt = parser.parse_args()
    print(opt)

    DATA_PATH = Path(opt.data)
    RESULT_PATH = Path('results')
    RESULT_PATH.mkdir(exist_ok=True)
    N_FOLD = opt.fold
    EPOCH = opt.epoch
    BATCH_SIZE = opt.batch_size
    HIDDEN_DIM = opt.hidden_dim
    DEFAULT_LR = 1e-4
    DEFAULT_DECAY = 5*1e-5
    EARLY_STOPPING_ROUNDS = 50
    DISP_INTERVAL = 10
    CTYPE = opt.types

    main()
