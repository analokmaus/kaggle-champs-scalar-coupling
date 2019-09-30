from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import argparse

import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

import lightgbm as lgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold
from sklearn import metrics
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from common import *


#%%
'''
ORIGIN_PATH = Path('/Volumes/T5/CHAMPS/')
DATA_PATH = ORIGIN_PATH/'tabular_features'
RESULT_PATH = Path('results')
ITER = 20000
N_FOLD = 5
CPU = 4
'''


#%%
PARAMS = {
    '3JHH': {
        'num_leaves': 128,
        'objective': 'huber',
        'learning_rate': 0.12,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.8,
        "metric": 'mae',
        "verbosity": -1,
        'lambda_l1': 0.8,
        'lambda_l2': 0.2,
        'feature_fraction': 0.7,
    },
    '3JHN': {
        'num_leaves': 256,
        'objective': 'huber',
        'learning_rate': 0.12,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.8,
        "metric": 'mae',
        "verbosity": -1,
        'lambda_l1': 0.8,
        'lambda_l2': 0.2,
        'feature_fraction': 0.7,
    },
    '3JHC': {
        'num_leaves': 256,
        'feature_fraction': 0.5,
        'objective': 'huber',
        'learning_rate': 0.15,
        "bagging_seed": 11,
        "metric": 'mae',
        'bagging_fraction': 0.75,
        'bagging_freq': 1,
        'boosting_type': 'gbdt',
        'alpha':0.85
    },
    '2JHN': {
        'num_leaves': 380,
        'objective': 'huber',
        'learning_rate': 0.15,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.8,
        "metric": 'mae',
        "verbosity": -1,
        'lambda_l1': 0.8,
        'lambda_l2': 0.2,
        'feature_fraction': 0.7,
    },
    '2JHH': {
        'num_leaves': 380,
        'objective': 'huber',
        'learning_rate': 0.15,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.8,
        "metric": 'mae',
        "verbosity": -1,
        'lambda_l1': 0.8,
        'lambda_l2': 0.2,
        'feature_fraction': 0.7,
    },
    '2JHC': {
        'num_leaves': 380,
        'objective': 'huber',
        'learning_rate': 0.15,
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "subsample": 0.8,
        "metric": 'mae',
        "verbosity": -1,
        'lambda_l1': 0.8,
        'lambda_l2': 0.2,
        'feature_fraction': 0.7,
    },
    '1JHN': {
        'num_leaves': 4,
        'feature_fraction': 0.5,
        'objective': 'huber',
        'learning_rate': 0.15,
        "bagging_seed": 11,
        "metric": 'mae',
        'bagging_fraction': 0.75,
        'bagging_freq': 1,
        'boosting_type': 'gbdt',
        'alpha':0.85,
    },
    '1JHC': {
        'num_leaves': 128,
        'feature_fraction': 1.0,
        'objective': 'huber',
        'learning_rate': 0.15,
        "bagging_seed": 11,
        "metric": 'mae',
        'bagging_fraction': 0.75,
        'bagging_freq': 1,
        'boosting_type': 'gbdt',
        'alpha':0.85
    }
}


def main():

    #%%
    n_estimators_default = ITER
    n_fold = N_FOLD


    #%%
    for t in CTYPES:
        params = PARAMS[t]
        params['random_state'] = SEED
        params['num_threads'] = CPU

        # Train set
        X = pd.read_csv(DATA_PATH/'train'/f'{t}_full.csv', index_col=0)
        X = reduce_mem_usage(X)
        y_all = pd.read_csv(ORIGIN_PATH/'scalar_coupling_contributions.csv').drop('type', axis=1)
        y_all = reduce_mem_usage(y_all)
        X = X.merge(y_all, on=['molecule_name', 'atom_index_0', 'atom_index_1'], how='left')
        ys = {
            'sum': X['scalar_coupling_constant'],
            'fc': X['fc'],
            'sd': X['sd'],
            'pso': X['pso'],
            'dso': X['dso'],
        }
        X = X.drop(['scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso'], axis=1)

        X_test = pd.read_csv(DATA_PATH/'test'/f'{t}_full.csv', index_col=0)
        X_test = reduce_mem_usage(X_test)

        X_all = pd.concat([X, X_test])
        cat_features = []
        for col in X_all.columns:
            if col[-5:] == '_atom' or col in ['atom_A', 'atom_B']:
                cat_features.append(col)
        print(cat_features)
        for col in cat_features:
            print(col)
            X_all[col] = label_encode(X_all[col])


        X = X_all.iloc[:len(X)]
        X_test = X_all.iloc[len(X):]
        del X_all; gc.collect()


        index_train = X['id']
        groups = X['molecule_name']
        if t[:2] == '1J':
            X_t = X.drop(['atom_index_0','atom_index_1','id', 'type', 'molecule_name'],axis=1)
        elif t[:2] == '2J':
            X_t = X.drop(['atom_index_0','atom_index_1','atom_index_A','id', 'type', 'molecule_name'],axis=1)
        elif t[:2] == '3J':
            X_t = X.drop(['atom_index_0','atom_index_1','atom_index_A','atom_index_B','id','type', 'molecule_name'],axis=1)

        index_test = X_test['id']
        if t[:2] == '1J':
            X_test_t = X_test.drop(['atom_index_0','atom_index_1','id', 'type', 'molecule_name'],axis=1)
        elif t[:2] == '2J':
            X_test_t = X_test.drop(['atom_index_0','atom_index_1','atom_index_A','id', 'type', 'molecule_name'],axis=1)
        elif t[:2] == '3J':
            X_test_t = X_test.drop(['atom_index_0','atom_index_1','atom_index_A','atom_index_B','id','type', 'molecule_name'],axis=1)


        params['categorical_feature'] = [X_t.columns.get_loc(x) for x in cat_features]

        for ytype, y_t in ys.items():
            res = []
            if opt.icm and ytype == 'sum':
                continue
            elif not opt.icm and ytype != 'sum':
                continue

            # Split data
            folds = GroupKFold(n_splits=n_fold)
            folds.get_n_splits(X_t,y_t,groups)

            # Train!
            print(f'Starting {t} / {ytype}')
            print(f'Params:\n{params}')
            result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t,
                                                      params=params, folds=folds, model_type='lgb', eval_metric='mae',
                                                      plot_feature_importance=True, verbose=1000, early_stopping_rounds=200,
                                                      n_estimators=n_estimators_default, groups=groups,
                                                      feature_importance_path=f'results/{t}.png')
            if opt.icm:
                res.append((f'{t}_{ytype}', result_dict_lgb3))
                with open(RESULT_PATH/f'{t}_{ytype}.pkl', 'wb') as f:
                    pickle.dump(res, f)
            else:
                res.append((t, result_dict_lgb3))
                with open(RESULT_PATH/f'{t}.pkl', 'wb') as f:
                    pickle.dump(res, f)


if __name__ == "__main__":
    #%%
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='~/kaggle-data/champs/',
                        help="root path to data")
    parser.add_argument("--iter", type=int, default=20000,
                        help="number of iterations(trees)")
    parser.add_argument("--fold", type=int, default=5,
                        help="number of folds")
    parser.add_argument("--cpu", type=int, default=20,
                        help="number of cpu")
    parser.add_argument("--types", type=str, default='1JHN',
                        help="which type to train (default: 1JHN)")
    parser.add_argument("--icm", action='store_true',
                        help="whether to predict independent contribution")

    opt = parser.parse_args()
    print(opt)

    ORIGIN_PATH = Path(opt.data)
    DATA_PATH = ORIGIN_PATH/'tabular-features'
    RESULT_PATH = Path('results')
    RESULT_PATH.mkdir(exist_ok=True)
    ITER = opt.iter
    N_FOLD = opt.fold
    CPU = opt.cpu
    SEED = 2019
    if opt.types == '3J':
        CTYPES = ['3JHH', '3JHC', '3JHN']
    elif opt.types == '2J':
        CTYPES = ['2JHH', '2JHC', '2JHN']
    else:
        CTYPES = opt.types.split(',')rm *

    main()
