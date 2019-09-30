# %%
import pickle
from altair.vega import v3
from itertools import product
from catboost import CatBoostRegressor, CatBoostClassifier
from numba import jit
import networkx as nx
import altair as alt
import json
from IPython.display import HTML
import warnings
import seaborn as sns
import gc
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
import datetime
import time
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.svm import NuSVR, SVR
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
%matplotlib inline
pd.options.display.precision = 15

warnings.filterwarnings("ignore")


%matplotlib inline


# %%

def prepare_altair():
    """
    Helper function to prepare altair for working.
    """
    vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION
    vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
    vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
    vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
    noext = "?noext"

    paths = {
        'vega': vega_url + noext,
        'vega-lib': vega_lib_url + noext,
        'vega-lite': vega_lite_url + noext,
        'vega-embed': vega_embed_url + noext
    }

    workaround = f"""    requirejs.config({{
        baseUrl: 'https://cdn.jsdelivr.net/npm/',
        paths: {paths}
    }});
    """

    return workaround


def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)
    # Cache will stay outside and
    return wrapped


@add_autoincrement
def render(chart, id="vega-chart"):
    """
    Helper function to plot altair visualizations.
    """
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vg_embed) {{
        const spec = {chart};
        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
        console.log("anything?");
    }});
    console.log("really...anything?");
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart) if isinstance(
                chart, dict) else chart.to_json(indent=None)
        )
    )


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


@jit
def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(y_true, y_pred):
    """
    Fast auc eval function for lgb.
    """
    return 'auc', fast_auc(y_true, y_pred), True


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true - y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,
                           verbose=10000, early_stopping_rounds=200, n_estimators=50000, groups=None):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type

    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]

    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                            'catboost_metric_name': 'MAE',
                            'sklearn_scoring_function': metrics.mean_absolute_error},
                    'group_mae': {'lgb_metric_name': 'mae',
                                  'catboost_metric_name': 'MAE',
                                  'scoring_function': group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                            'catboost_metric_name': 'MSE',
                            'sklearn_scoring_function': metrics.mean_squared_error}
                    }

    result_dict = {}

    # out-of-fold predictions on train data
    oof = np.zeros(len(X))

    # averaged predictions on train data
    prediction = np.zeros(len(X_test))

    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()

    # split and train on folds
    if groups is not None:
        fold_split = folds.split(X, y, groups)
    else:
        fold_split = folds.split(X)
    for fold_n, (train_index, valid_index) in enumerate(fold_split):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(
                **params, n_estimators=n_estimators, n_jobs=-1)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)
                                ], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(
                data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(
                data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist,
                              early_stopping_rounds=200, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(
                X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(
                X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](
                y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')

            y_pred = model.predict(X_test).reshape(-1,)

        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                      cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid.reshape(-1,)
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](
                y_valid, y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](
                y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0)

    prediction /= folds.n_splits

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(
        np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= folds.n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(
                cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(
                by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')

            result_dict['feature_importance'] = feature_importance

    return result_dict


def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type

    """
    columns = X.columns if columns == None else columns
    X_test = X_test[columns]

    # to set up scoring parameters
    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,
                            'catboost_metric_name': 'AUC',
                            'sklearn_scoring_function': metrics.roc_auc_score},
                    }

    result_dict = {}

    # out-of-fold predictions on train data
    oof = np.zeros((len(X), len(set(y.values))))

    # averaged predictions on train data
    prediction = np.zeros((len(X_test), oof.shape[1]))

    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMClassifier(
                **params, n_estimators=n_estimators, n_jobs=-1)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)
                                ], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            y_pred_valid = model.predict_proba(X_valid)
            y_pred = model.predict_proba(
                X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(
                data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(
                data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist,
                              early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(
                X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(
                X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](
                y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')

            y_pred = model.predict_proba(X_test)

        if model_type == 'cat':
            model = CatBoostClassifier(iterations=n_estimators, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
                                       loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                      cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        oof[valid_index] = y_pred_valid
        scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](
            y_valid, y_pred_valid[:, 1]))

        prediction += y_pred

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat(
                [feature_importance, fold_importance], axis=0)

    prediction /= folds.n_splits

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(
        np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= folds.n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(
                cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(
                by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')

            result_dict['feature_importance'] = feature_importance

    return result_dict


# setting up altair
workaround = prepare_altair()
HTML("".join((
    "<script>",
    workaround,
    "</script>",
)))

# %%
DATA_PATH = Path('champs-scalar-coupling')
train = pd.read_csv(DATA_PATH / 'giba_dataset' / 'train.csv')
test = pd.read_csv(DATA_PATH / 'giba_dataset' / 'test.csv')
sub = pd.read_csv(DATA_PATH / 'sample_submission.csv')

# %%
n_estimators_default = 8000
params = {'num_leaves': 255,
          'feature_fraction': 0.5,
          'objective': 'huber',
          'learning_rate': 0.15,
          "bagging_seed": 11,
          "metric": 'mae',
          'bagging_fraction': 0.75,
          'bagging_freq': 1,
          'boosting_type': 'gbdt',
          'alpha': 0.85
          }
# %%
n_estimators_default = 8000
params = {'num_leaves': 50,
          'min_child_samples': 79,
          'min_data_in_leaf': 100,
          'objective': 'regression',
          'max_depth': 9,
          'learning_rate': 0.2,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1,
          'reg_lambda': 0.3,
          'colsample_bytree': 1.0
          }
# %%

n_fold = 10

X = train.drop(['scalar_coupling_constant'], axis=1)
X = X.reset_index().rename(columns={'index': 'id'})
X_test = test.drop(['scalar_coupling_constant'], axis=1)
X_test = X_test.reset_index().rename(columns={'index': 'id'})
y = train['scalar_coupling_constant']
X_short = pd.DataFrame({'ind': list(X.index), 'type': X['type'].values, 'oof': [
                       0] * len(X), 'target': y.values})
X_short_test = pd.DataFrame({'ind': list(
    X_test.index), 'type': X_test['type'].values, 'prediction': [0] * len(X_test)})
qm9 = pd.read_pickle(DATA_PATH / 'qm9' / 'data.covs.pickle').drop(
    ['id', 'type', 'scalar_coupling_constant'], axis=1)
X = X.merge(qm9, on=['molecule_name', 'atom_index_0',
                     'atom_index_1'], how='left')
X_test = X_test.merge(
    qm9, on=['molecule_name', 'atom_index_0', 'atom_index_1'], how='left')
# %%
# X_bond = pd.read_pickle(DATA_PATH/'dataset'/'R3_bond'/'train_R3_bond.pickle')
# X_test_bond = pd.read_pickle(DATA_PATH/'dataset'/'R3_bond'/'train_R3_bond.pickle')
# good_columns = ['type','id','molecule_name','atom_index_0','atom_index_1',
#  'bond_lengths_mean_y',
#  'bond_lengths_median_y',
#  'bond_lengths_std_y',
#  'bond_lengths_mean_x',
#  'molecule_atom_index_0_dist_min_div',
#  'molecule_atom_index_0_dist_std_div',
#  'molecule_atom_index_0_dist_mean',
#  'molecule_atom_index_0_dist_max',
#  'molecule_atom_index_1_dist_std_diff',
#  'molecule_type_dist_min',
#  'molecule_atom_index_0_y_1_mean_div',
#  'molecule_type_dist_std',
#  'molecule_atom_index_0_y_1_std',
#  'molecule_dist_mean',
#  'molecule_atom_index_0_dist_std_diff',
#  'molecule_atom_index_0_dist_std',
#  'molecule_atom_index_0_x_1_std',
#  'molecule_type_dist_std_diff',
#  'molecule_type_0_dist_std',
#  'molecule_atom_index_0_dist_mean_diff',
#  'molecule_atom_index_1_dist_min_div',
#  'molecule_atom_index_1_dist_mean_diff',
#  'molecule_type_dist_mean_div',
#  'molecule_dist_max',
#  'molecule_atom_index_0_dist_mean_div',
#  'molecule_atom_index_0_z_1_std',
#  'molecule_atom_index_1_dist_mean_div',
#  'molecule_atom_index_1_dist_min_diff',
#  'molecule_atom_index_1_dist_mean',
#  'molecule_atom_index_1_dist_min',
#  'molecule_atom_index_1_dist_max',
#  'molecule_type_0_dist_std_diff',
#  'molecule_atom_index_0_dist_min_diff',
#  'molecule_type_dist_mean_diff',
#  'molecule_atom_index_0_y_1_max',
#  'molecule_atom_index_0_y_1_mean_diff',
#  'molecule_atom_1_dist_std_diff',
#  'molecule_atom_index_0_y_1_mean',
#  'molecule_atom_1_dist_std',
#  'molecule_type_dist_max']
# X_bond = X_bond[good_columns]
# X_test_bond = X_test_bond[good_columns]
# X = X.merge(X_bond,on=['type','id','molecule_name','atom_index_0','atom_index_1'],how='left')
# X_test = X_test.merge(X_test_bond,on=['type','id','molecule_name','atom_index_0','atom_index_1'],how='left')
# %%
n_estimators_default = 8000
n_fold = 10
res = []

for t in ['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']:
    params = {'num_leaves': 255,
              'feature_fraction': 0.5,
              'objective': 'huber',
              'learning_rate': 0.15,
              "bagging_seed": 11,
              "metric": 'mae',
              'bagging_fraction': 0.75,
              'bagging_freq': 1,
              'boosting_type': 'gbdt',
              'alpha': 0.85
              }
    print(f'Training of type {t}')
    if t == '1JHC':
        for n_bond in range(2, 5):
            print(f'Training of type {t}, bonds {n_bond}')
            X_1JHC = pd.read_csv('champs-scalar-coupling/neighborhood/1JHC/train_1JHC_{}_bond.csv'.format(
                n_bond)).drop(['type', 'scalar_coupling_constant', 'dist'], axis=1)
            X_t = X.merge(
                X_1JHC, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_train = X_t['id']
            X_t = X_t.drop(
                ['atom_index_0', 'atom_index_1', 'id', 'type'], axis=1)
            y_t = X_short.loc[index_train, 'target']

            X_1JHC_test = pd.read_csv(
                'champs-scalar-coupling/neighborhood/1JHC/test_1JHC_{}_bond.csv'.format(n_bond)).drop(['type', 'dist'], axis=1)
            X_test_t = X_test.merge(
                X_1JHC_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_test = X_test_t['id']
            X_test_t = X_test_t.drop(
                ['atom_index_0', 'atom_index_1', 'id', 'type'], axis=1)
            X_t_train_test = pd.get_dummies(pd.concat([X_t, X_test_t]), columns=[
                                            'atom_{}'.format(n) for n in range(2, n_bond + 1)])
            X_t_train_test['type'] = 0

            X_t = X_t_train_test[:len(X_t)]
            X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
            folds = GroupKFold(n_splits=n_fold)
            groups = X_t['molecule_name']
            X_t = X_t.drop('molecule_name', axis=1)
            folds.get_n_splits(X_t, y_t, groups)

            result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                      verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
            X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
            X_short_test.loc[index_test,
                             'prediction'] = result_dict_lgb3['prediction']
            res.append((t, n_bond, result_dict_lgb3))

    elif t == '1JHN':
        for n_bond in range(2, 5):
            print(f'Training of type {t}, bonds {n_bond}')
            X_1JHN = pd.read_csv('champs-scalar-coupling/neighborhood/1JHN/train_1JHN_{}_bond.csv'.format(
                n_bond)).drop(['type', 'scalar_coupling_constant', 'dist'], axis=1)
            X_t = X.merge(
                X_1JHN, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_train = X_t['id']
            X_t = X_t.drop(['atom_index_0', 'atom_index_1', 'id'], axis=1)
            y_t = X_short.loc[index_train, 'target']

            X_1JHN_test = pd.read_csv(
                'champs-scalar-coupling/neighborhood/1JHN/test_1JHN_{}_bond.csv'.format(n_bond)).drop(['type', 'dist'], axis=1)
            X_test_t = X_test.merge(
                X_1JHN_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_test = X_test_t['id']
            X_test_t = X_test_t.drop(
                ['atom_index_0', 'atom_index_1', 'id'], axis=1)
            X_t_train_test = pd.get_dummies(pd.concat([X_t, X_test_t]), columns=[
                                            'atom_{}'.format(n) for n in range(2, n_bond + 1)])
            X_t_train_test['type'] = 0

            X_t = X_t_train_test[:len(X_t)]
            X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
            folds = GroupKFold(n_splits=n_fold)
            groups = X_t['molecule_name']
            X_t = X_t.drop('molecule_name', axis=1)
            folds.get_n_splits(X_t, y_t, groups)

            result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                      verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
            X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
            X_short_test.loc[index_test,
                             'prediction'] = result_dict_lgb3['prediction']
            res.append((t, n_bond, result_dict_lgb3))

    elif t in ['3JHC', '3JHH', '3JHN']:
        if t == '3JHH':
            n_estimators_default = 8000
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        elif t == '3JHN':
            params = {'num_leaves': 500,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        thisType = t
        X_3 = pd.read_csv('champs-scalar-coupling/neighborhood/{}/train_{}_v1.csv'.format(
            thisType, thisType)).drop(['type', 'scalar_coupling_constant'], axis=1)
        X_t = X.merge(
            X_3, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        index_train = X_t['id']
        X_t = X_t.drop(['atom_index_0', 'atom_index_1',
                        'atom_index_A', 'atom_index_B', 'id', 'type'], axis=1)
        y_t = X_short.loc[index_train, 'target']

        X_3_test = pd.read_csv('champs-scalar-coupling/neighborhood/{}/test_{}_v1.csv'.format(
            thisType, thisType)).drop(['type'], axis=1)
        X_test_t = X_test.merge(
            X_3_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        index_test = X_test_t['id']
        X_test_t = X_test_t.drop(
            ['atom_index_0', 'atom_index_1', 'atom_index_A', 'atom_index_B', 'id', 'type'], axis=1)
        X_t_train_test = pd.get_dummies(
            pd.concat([X_t, X_test_t]), columns=['atom_A', 'atom_B'])
        X_t_train_test['type'] = 0

        X_t = X_t_train_test[:len(X_t)]
        X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
        folds = GroupKFold(n_splits=n_fold)
        groups = X_t['molecule_name']
        X_t = X_t.drop('molecule_name', axis=1)
        folds.get_n_splits(X_t, y_t, groups)

        result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                  verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
        X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
        X_short_test.loc[index_test,
                         'prediction'] = result_dict_lgb3['prediction']
        res.append((t, result_dict_lgb3))

    elif t in ['2JHC', '2JHH', '2JHN']:
        if t == '2JHN':
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 30,
                      'learning_rate': 0.1,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        elif t == '2JHH':
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        elif t == '2JHC':
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        X_2 = pd.read_csv('champs-scalar-coupling/neighborhood/2J/train_2J_v1.csv').drop(
            ['type', 'scalar_coupling_constant'], axis=1)
        X_t = X.merge(
            X_2, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        X_t = X_t[X_t['type'] == t]
        index_train = X_t['id']
        X_t = X_t.drop(['atom_index_0', 'atom_index_1',
                        'atom_index_A', 'id', 'type'], axis=1)
        y_t = X_short.loc[index_train, 'target']

        X_2_test = pd.read_csv(
            'champs-scalar-coupling/neighborhood/2J/test_2J_v1.csv').drop(['type'], axis=1)
        X_test_t = X_test.merge(
            X_2_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        X_test_t = X_test_t[X_test_t['type'] == t]
        index_test = X_test_t['id']
        X_test_t = X_test_t.drop(
            ['atom_index_0', 'atom_index_1', 'atom_index_A', 'id', 'type'], axis=1)
        X_t_train_test = pd.get_dummies(
            pd.concat([X_t, X_test_t]), columns=['atom_A'])
        X_t_train_test['type'] = 0

        X_t = X_t_train_test[:len(X_t)]
        X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
        folds = GroupKFold(n_splits=n_fold)
        groups = X_t['molecule_name']
        X_t = X_t.drop('molecule_name', axis=1)
        folds.get_n_splits(X_t, y_t, groups)

        result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                  verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
        X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
        X_short_test.loc[index_test,
                         'prediction'] = result_dict_lgb3['prediction']
        res.append((t, result_dict_lgb3))
# %%
res.append({'train': X_short, 'test': X_short_test})
gc.collect()
sub['scalar_coupling_constant'] = X_short_test['prediction']
sub.to_csv('output/submission_qm9_10folds_param_tune.csv', index=False)
sub.head()

with open('output/pickle/res_dict_qm9_10folds_param_tune.pickle', 'wb') as f:
    pickle.dump(res, f)


# fc
# %%
X_short = pd.DataFrame({'ind': list(X.index), 'type': X['type'].values, 'oof': [
                       0] * len(X), 'target': df['fc'].values})
n_estimators_default = 8000
n_fold = 5
res = []

for t in ['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']:
    params = {'num_leaves': 255,
              'feature_fraction': 0.5,
              'objective': 'huber',
              'learning_rate': 0.15,
              "bagging_seed": 11,
              "metric": 'mae',
              'bagging_fraction': 0.75,
              'bagging_freq': 1,
              'boosting_type': 'gbdt',
              'alpha': 0.85
              }
    print(f'Training of type {t}')
    if t == '1JHC':
        for n_bond in range(2, 5):
            print(f'Training of type {t}, bonds {n_bond}')
            X_1JHC = pd.read_csv('champs-scalar-coupling/neighborhood/1JHC/train_1JHC_{}_bond.csv'.format(
                n_bond)).drop(['type', 'scalar_coupling_constant', 'dist'], axis=1)
            X_t = X.merge(
                X_1JHC, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_train = X_t['id']
            X_t = X_t.drop(
                ['atom_index_0', 'atom_index_1', 'id', 'type'], axis=1)
            y_t = X_short.loc[index_train, 'target']

            X_1JHC_test = pd.read_csv(
                'champs-scalar-coupling/neighborhood/1JHC/test_1JHC_{}_bond.csv'.format(n_bond)).drop(['type', 'dist'], axis=1)
            X_test_t = X_test.merge(
                X_1JHC_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_test = X_test_t['id']
            X_test_t = X_test_t.drop(
                ['atom_index_0', 'atom_index_1', 'id', 'type'], axis=1)
            X_t_train_test = pd.get_dummies(pd.concat([X_t, X_test_t]), columns=[
                                            'atom_{}'.format(n) for n in range(2, n_bond + 1)])
            X_t_train_test['type'] = 0

            X_t = X_t_train_test[:len(X_t)]
            X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
            folds = GroupKFold(n_splits=n_fold)
            groups = X_t['molecule_name']
            X_t = X_t.drop('molecule_name', axis=1)
            folds.get_n_splits(X_t, y_t, groups)

            result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                      verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
            X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
            X_short_test.loc[index_test,
                             'prediction'] = result_dict_lgb3['prediction']
            res.append((t, n_bond, result_dict_lgb3))

    elif t == '1JHN':
        for n_bond in range(2, 5):
            print(f'Training of type {t}, bonds {n_bond}')
            X_1JHN = pd.read_csv('champs-scalar-coupling/neighborhood/1JHN/train_1JHN_{}_bond.csv'.format(
                n_bond)).drop(['type', 'scalar_coupling_constant', 'dist'], axis=1)
            X_t = X.merge(
                X_1JHN, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_train = X_t['id']
            X_t = X_t.drop(['atom_index_0', 'atom_index_1', 'id'], axis=1)
            y_t = X_short.loc[index_train, 'target']

            X_1JHN_test = pd.read_csv(
                'champs-scalar-coupling/neighborhood/1JHN/test_1JHN_{}_bond.csv'.format(n_bond)).drop(['type', 'dist'], axis=1)
            X_test_t = X_test.merge(
                X_1JHN_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_test = X_test_t['id']
            X_test_t = X_test_t.drop(
                ['atom_index_0', 'atom_index_1', 'id'], axis=1)
            X_t_train_test = pd.get_dummies(pd.concat([X_t, X_test_t]), columns=[
                                            'atom_{}'.format(n) for n in range(2, n_bond + 1)])
            X_t_train_test['type'] = 0

            X_t = X_t_train_test[:len(X_t)]
            X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
            folds = GroupKFold(n_splits=n_fold)
            groups = X_t['molecule_name']
            X_t = X_t.drop('molecule_name', axis=1)
            folds.get_n_splits(X_t, y_t, groups)

            result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                      verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
            X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
            X_short_test.loc[index_test,
                             'prediction'] = result_dict_lgb3['prediction']
            res.append((t, n_bond, result_dict_lgb3))

    elif t in ['3JHC', '3JHH', '3JHN']:
        if t == '3JHH':
            n_estimators_default = 8000
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        elif t == '3JHN':
            params = {'num_leaves': 500,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        thisType = t
        X_3 = pd.read_csv('champs-scalar-coupling/neighborhood/{}/train_{}_v1.csv'.format(
            thisType, thisType)).drop(['type', 'scalar_coupling_constant'], axis=1)
        X_t = X.merge(
            X_3, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        index_train = X_t['id']
        X_t = X_t.drop(['atom_index_0', 'atom_index_1',
                        'atom_index_A', 'atom_index_B', 'id', 'type'], axis=1)
        y_t = X_short.loc[index_train, 'target']

        X_3_test = pd.read_csv('champs-scalar-coupling/neighborhood/{}/test_{}_v1.csv'.format(
            thisType, thisType)).drop(['type'], axis=1)
        X_test_t = X_test.merge(
            X_3_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        index_test = X_test_t['id']
        X_test_t = X_test_t.drop(
            ['atom_index_0', 'atom_index_1', 'atom_index_A', 'atom_index_B', 'id', 'type'], axis=1)
        X_t_train_test = pd.get_dummies(
            pd.concat([X_t, X_test_t]), columns=['atom_A', 'atom_B'])
        X_t_train_test['type'] = 0

        X_t = X_t_train_test[:len(X_t)]
        X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
        folds = GroupKFold(n_splits=n_fold)
        groups = X_t['molecule_name']
        X_t = X_t.drop('molecule_name', axis=1)
        folds.get_n_splits(X_t, y_t, groups)

        result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                  verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
        X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
        X_short_test.loc[index_test,
                         'prediction'] = result_dict_lgb3['prediction']
        res.append((t, result_dict_lgb3))

    elif t in ['2JHC', '2JHH', '2JHN']:
        if t == '2JHN':
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 30,
                      'learning_rate': 0.1,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        elif t == '2JHH':
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        elif t == '2JHC':
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        X_2 = pd.read_csv('champs-scalar-coupling/neighborhood/2J/train_2J_v1.csv').drop(
            ['type', 'scalar_coupling_constant'], axis=1)
        X_t = X.merge(
            X_2, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        X_t = X_t[X_t['type'] == t]
        index_train = X_t['id']
        X_t = X_t.drop(['atom_index_0', 'atom_index_1',
                        'atom_index_A', 'id', 'type'], axis=1)
        y_t = X_short.loc[index_train, 'target']

        X_2_test = pd.read_csv(
            'champs-scalar-coupling/neighborhood/2J/test_2J_v1.csv').drop(['type'], axis=1)
        X_test_t = X_test.merge(
            X_2_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        X_test_t = X_test_t[X_test_t['type'] == t]
        index_test = X_test_t['id']
        X_test_t = X_test_t.drop(
            ['atom_index_0', 'atom_index_1', 'atom_index_A', 'id', 'type'], axis=1)
        X_t_train_test = pd.get_dummies(
            pd.concat([X_t, X_test_t]), columns=['atom_A'])
        X_t_train_test['type'] = 0

        X_t = X_t_train_test[:len(X_t)]
        X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
        folds = GroupKFold(n_splits=n_fold)
        groups = X_t['molecule_name']
        X_t = X_t.drop('molecule_name', axis=1)
        folds.get_n_splits(X_t, y_t, groups)

        result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                  verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
        X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
        X_short_test.loc[index_test,
                         'prediction'] = result_dict_lgb3['prediction']
        res.append((t, result_dict_lgb3))
with open('output/pickle/res_dict_qm9_5folds_param_tune_fc.pickle', 'wb') as f:
    pickle.dump(res, f)
X_short.to_pickle('output/pickle/qm9_5folds_param_tune_oof_fc_train.pickle')
X_short_test.to_pickle(
    'output/pickle/qm9_5folds_param_tune_oof_fc_test.pickle')
# %%
X['oof_fc'] = X_short['oof']
X_test['oof_fc'] = X_short_test['prediction']
# %%
X_short['target'] = y.values
n_estimators_default = 8000
n_fold = 10
res = []

for t in ['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']:
    params = {'num_leaves': 255,
              'feature_fraction': 0.5,
              'objective': 'huber',
              'learning_rate': 0.15,
              "bagging_seed": 11,
              "metric": 'mae',
              'bagging_fraction': 0.75,
              'bagging_freq': 1,
              'boosting_type': 'gbdt',
              'alpha': 0.85
              }
    print(f'Training of type {t}')
    if t == '1JHC':
        for n_bond in range(2, 5):
            print(f'Training of type {t}, bonds {n_bond}')
            X_1JHC = pd.read_csv('champs-scalar-coupling/neighborhood/1JHC/train_1JHC_{}_bond.csv'.format(
                n_bond)).drop(['type', 'scalar_coupling_constant', 'dist'], axis=1)
            X_t = X.merge(
                X_1JHC, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_train = X_t['id']
            X_t = X_t.drop(
                ['atom_index_0', 'atom_index_1', 'id', 'type'], axis=1)
            y_t = X_short.loc[index_train, 'target']

            X_1JHC_test = pd.read_csv(
                'champs-scalar-coupling/neighborhood/1JHC/test_1JHC_{}_bond.csv'.format(n_bond)).drop(['type', 'dist'], axis=1)
            X_test_t = X_test.merge(
                X_1JHC_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_test = X_test_t['id']
            X_test_t = X_test_t.drop(
                ['atom_index_0', 'atom_index_1', 'id', 'type'], axis=1)
            X_t_train_test = pd.get_dummies(pd.concat([X_t, X_test_t]), columns=[
                                            'atom_{}'.format(n) for n in range(2, n_bond + 1)])
            X_t_train_test['type'] = 0

            X_t = X_t_train_test[:len(X_t)]
            X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
            folds = GroupKFold(n_splits=n_fold)
            groups = X_t['molecule_name']
            X_t = X_t.drop('molecule_name', axis=1)
            folds.get_n_splits(X_t, y_t, groups)

            result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                      verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
            X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
            X_short_test.loc[index_test,
                             'prediction'] = result_dict_lgb3['prediction']
            res.append((t, n_bond, result_dict_lgb3))

    elif t == '1JHN':
        for n_bond in range(2, 5):
            print(f'Training of type {t}, bonds {n_bond}')
            X_1JHN = pd.read_csv('champs-scalar-coupling/neighborhood/1JHN/train_1JHN_{}_bond.csv'.format(
                n_bond)).drop(['type', 'scalar_coupling_constant', 'dist'], axis=1)
            X_t = X.merge(
                X_1JHN, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_train = X_t['id']
            X_t = X_t.drop(['atom_index_0', 'atom_index_1', 'id'], axis=1)
            y_t = X_short.loc[index_train, 'target']

            X_1JHN_test = pd.read_csv(
                'champs-scalar-coupling/neighborhood/1JHN/test_1JHN_{}_bond.csv'.format(n_bond)).drop(['type', 'dist'], axis=1)
            X_test_t = X_test.merge(
                X_1JHN_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_test = X_test_t['id']
            X_test_t = X_test_t.drop(
                ['atom_index_0', 'atom_index_1', 'id'], axis=1)
            X_t_train_test = pd.get_dummies(pd.concat([X_t, X_test_t]), columns=[
                                            'atom_{}'.format(n) for n in range(2, n_bond + 1)])
            X_t_train_test['type'] = 0

            X_t = X_t_train_test[:len(X_t)]
            X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
            folds = GroupKFold(n_splits=n_fold)
            groups = X_t['molecule_name']
            X_t = X_t.drop('molecule_name', axis=1)
            folds.get_n_splits(X_t, y_t, groups)

            result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                      verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
            X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
            X_short_test.loc[index_test,
                             'prediction'] = result_dict_lgb3['prediction']
            res.append((t, n_bond, result_dict_lgb3))

    elif t in ['3JHC', '3JHH', '3JHN']:
        if t == '3JHH':
            n_estimators_default = 8000
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        elif t == '3JHN':
            params = {'num_leaves': 500,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        thisType = t
        X_3 = pd.read_csv('champs-scalar-coupling/neighborhood/{}/train_{}_v1.csv'.format(
            thisType, thisType)).drop(['type', 'scalar_coupling_constant'], axis=1)
        X_t = X.merge(
            X_3, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        index_train = X_t['id']
        X_t = X_t.drop(['atom_index_0', 'atom_index_1',
                        'atom_index_A', 'atom_index_B', 'id', 'type'], axis=1)
        y_t = X_short.loc[index_train, 'target']

        X_3_test = pd.read_csv('champs-scalar-coupling/neighborhood/{}/test_{}_v1.csv'.format(
            thisType, thisType)).drop(['type'], axis=1)
        X_test_t = X_test.merge(
            X_3_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        index_test = X_test_t['id']
        X_test_t = X_test_t.drop(
            ['atom_index_0', 'atom_index_1', 'atom_index_A', 'atom_index_B', 'id', 'type'], axis=1)
        X_t_train_test = pd.get_dummies(
            pd.concat([X_t, X_test_t]), columns=['atom_A', 'atom_B'])
        X_t_train_test['type'] = 0

        X_t = X_t_train_test[:len(X_t)]
        X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
        folds = GroupKFold(n_splits=n_fold)
        groups = X_t['molecule_name']
        X_t = X_t.drop('molecule_name', axis=1)
        folds.get_n_splits(X_t, y_t, groups)

        result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                  verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
        X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
        X_short_test.loc[index_test,
                         'prediction'] = result_dict_lgb3['prediction']
        res.append((t, result_dict_lgb3))

    elif t in ['2JHC', '2JHH', '2JHN']:
        if t == '2JHN':
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 30,
                      'learning_rate': 0.1,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        elif t == '2JHH':
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        elif t == '2JHC':
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        X_2 = pd.read_csv('champs-scalar-coupling/neighborhood/2J/train_2J_v1.csv').drop(
            ['type', 'scalar_coupling_constant'], axis=1)
        X_t = X.merge(
            X_2, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        X_t = X_t[X_t['type'] == t]
        index_train = X_t['id']
        X_t = X_t.drop(['atom_index_0', 'atom_index_1',
                        'atom_index_A', 'id', 'type'], axis=1)
        y_t = X_short.loc[index_train, 'target']

        X_2_test = pd.read_csv(
            'champs-scalar-coupling/neighborhood/2J/test_2J_v1.csv').drop(['type'], axis=1)
        X_test_t = X_test.merge(
            X_2_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        X_test_t = X_test_t[X_test_t['type'] == t]
        index_test = X_test_t['id']
        X_test_t = X_test_t.drop(
            ['atom_index_0', 'atom_index_1', 'atom_index_A', 'id', 'type'], axis=1)
        X_t_train_test = pd.get_dummies(
            pd.concat([X_t, X_test_t]), columns=['atom_A'])
        X_t_train_test['type'] = 0

        X_t = X_t_train_test[:len(X_t)]
        X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
        folds = GroupKFold(n_splits=n_fold)
        groups = X_t['molecule_name']
        X_t = X_t.drop('molecule_name', axis=1)
        folds.get_n_splits(X_t, y_t, groups)

        result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                  verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
        X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
        X_short_test.loc[index_test,
                         'prediction'] = result_dict_lgb3['prediction']
        res.append((t, result_dict_lgb3))
# %%
res.append({'train': X_short, 'test': X_short_test})
gc.collect()
sub['scalar_coupling_constant'] = X_short_test['prediction']
sub.to_csv('output/submission_qm9_10folds_param_tune_fc_stacking.csv', index=False)
sub.head()

with open('output/pickle/res_dict_qm9_10folds_param_tune_fc_stacking.pickle', 'wb') as f:
    pickle.dump(res, f)

# %%
d = res.copy()
top_features = {}
feature_num = 60

for x in d:
    if type(x) == dict:
        continue
    if x[0] == '1JHC':
        if '1JHC' not in top_features:
            top_features['1JHC'] = {}
        top_features['1JHC'][x[1]] = x[2]['feature_importance'].groupby('feature').mean().sort_values(
            by='importance', ascending=False)[:feature_num].reset_index()['feature'].tolist()
    elif x[0] == '1JHN':
        if '1JHN' not in top_features:
            top_features['1JHN'] = {}
        top_features['1JHN'][x[1]] = x[2]['feature_importance'].groupby('feature').mean().sort_values(
            by='importance', ascending=False)[:feature_num].reset_index()['feature'].tolist()
    else:
        top_features[x[0]] = x[1]['feature_importance'].groupby('feature').mean().sort_values(
            by='importance', ascending=False)[:feature_num].reset_index()['feature'].tolist()

n_estimators_default = 8000
n_fold = 10
res = []

for t in ['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']:
    params = {'num_leaves': 255,
              'feature_fraction': 0.5,
              'objective': 'huber',
              'learning_rate': 0.15,
              "bagging_seed": 11,
              "metric": 'mae',
              'bagging_fraction': 0.75,
              'bagging_freq': 1,
              'boosting_type': 'gbdt',
              'alpha': 0.85
              }
    print(f'Training of type {t}')
    if t == '1JHC':
        for n_bond in range(2, 5):
            print(f'Training of type {t}, bonds {n_bond}')
            X_1JHC = pd.read_csv('champs-scalar-coupling/neighborhood/1JHC/train_1JHC_{}_bond.csv'.format(
                n_bond)).drop(['type', 'scalar_coupling_constant', 'dist'], axis=1)
            X_t = X.merge(
                X_1JHC, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_train = X_t['id']
            X_t = X_t.drop(
                ['atom_index_0', 'atom_index_1', 'id', 'type'], axis=1)
            y_t = X_short.loc[index_train, 'target']

            X_1JHC_test = pd.read_csv(
                'champs-scalar-coupling/neighborhood/1JHC/test_1JHC_{}_bond.csv'.format(n_bond)).drop(['type', 'dist'], axis=1)
            X_test_t = X_test.merge(
                X_1JHC_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_test = X_test_t['id']
            X_test_t = X_test_t.drop(
                ['atom_index_0', 'atom_index_1', 'id', 'type'], axis=1)
            X_t_train_test = pd.get_dummies(pd.concat([X_t, X_test_t]), columns=[
                                            'atom_{}'.format(n) for n in range(2, n_bond + 1)])
            X_t_train_test['type'] = 0

            X_t = X_t_train_test[:len(X_t)]
            X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
            folds = GroupKFold(n_splits=n_fold)
            groups = X_t['molecule_name']
            X_t = X_t.drop('molecule_name', axis=1)
            folds.get_n_splits(X_t, y_t, groups)

            result_dict_lgb3 = train_model_regression(X=X_t[top_features[t][n_bond] + ['type']], X_test=X_test_t[top_features[t][n_bond] + ['type']], y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                      verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
            X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
            X_short_test.loc[index_test,
                             'prediction'] = result_dict_lgb3['prediction']
            res.append((t, n_bond, result_dict_lgb3))

    elif t == '1JHN':
        for n_bond in range(2, 5):
            print(f'Training of type {t}, bonds {n_bond}')
            X_1JHN = pd.read_csv('champs-scalar-coupling/neighborhood/1JHN/train_1JHN_{}_bond.csv'.format(
                n_bond)).drop(['type', 'scalar_coupling_constant', 'dist'], axis=1)
            X_t = X.merge(
                X_1JHN, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_train = X_t['id']
            X_t = X_t.drop(['atom_index_0', 'atom_index_1', 'id'], axis=1)
            y_t = X_short.loc[index_train, 'target']

            X_1JHN_test = pd.read_csv(
                'champs-scalar-coupling/neighborhood/1JHN/test_1JHN_{}_bond.csv'.format(n_bond)).drop(['type', 'dist'], axis=1)
            X_test_t = X_test.merge(
                X_1JHN_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
            index_test = X_test_t['id']
            X_test_t = X_test_t.drop(
                ['atom_index_0', 'atom_index_1', 'id'], axis=1)
            X_t_train_test = pd.get_dummies(pd.concat([X_t, X_test_t]), columns=[
                                            'atom_{}'.format(n) for n in range(2, n_bond + 1)])
            X_t_train_test['type'] = 0

            X_t = X_t_train_test[:len(X_t)]
            X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
            folds = GroupKFold(n_splits=n_fold)
            groups = X_t['molecule_name']
            X_t = X_t.drop('molecule_name', axis=1)
            folds.get_n_splits(X_t, y_t, groups)

            result_dict_lgb3 = train_model_regression(X=X_t[top_features[t][n_bond] + ['type']], X_test=X_test_t[top_features[t][n_bond] + ['type']], y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                      verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
            X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
            X_short_test.loc[index_test,
                             'prediction'] = result_dict_lgb3['prediction']
            res.append((t, n_bond, result_dict_lgb3))

    elif t in ['3JHC', '3JHH', '3JHN']:
        if t == '3JHH':
            n_estimators_default = 8000
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        elif t == '3JHN':
            params = {'num_leaves': 500,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        thisType = t
        X_3 = pd.read_csv('champs-scalar-coupling/neighborhood/{}/train_{}_v1.csv'.format(
            thisType, thisType)).drop(['type', 'scalar_coupling_constant'], axis=1)
        X_t = X.merge(
            X_3, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        index_train = X_t['id']
        X_t = X_t.drop(['atom_index_0', 'atom_index_1',
                        'atom_index_A', 'atom_index_B', 'id', 'type'], axis=1)
        y_t = X_short.loc[index_train, 'target']

        X_3_test = pd.read_csv('champs-scalar-coupling/neighborhood/{}/test_{}_v1.csv'.format(
            thisType, thisType)).drop(['type'], axis=1)
        X_test_t = X_test.merge(
            X_3_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        index_test = X_test_t['id']
        X_test_t = X_test_t.drop(
            ['atom_index_0', 'atom_index_1', 'atom_index_A', 'atom_index_B', 'id', 'type'], axis=1)
        X_t_train_test = pd.get_dummies(
            pd.concat([X_t, X_test_t]), columns=['atom_A', 'atom_B'])
        X_t_train_test['type'] = 0

        X_t = X_t_train_test[:len(X_t)]
        X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
        folds = GroupKFold(n_splits=n_fold)
        groups = X_t['molecule_name']
        X_t = X_t.drop('molecule_name', axis=1)
        folds.get_n_splits(X_t, y_t, groups)

        result_dict_lgb3 = train_model_regression(X=X_t[top_features[t] + ['type']], X_test=X_test_t[top_features[t] + ['type']], y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                  verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
        X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
        X_short_test.loc[index_test,
                         'prediction'] = result_dict_lgb3['prediction']
        res.append((t, result_dict_lgb3))

    elif t in ['2JHC', '2JHH', '2JHN']:
        if t == '2JHN':
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 30,
                      'learning_rate': 0.1,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        elif t == '2JHH':
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        elif t == '2JHC':
            params = {'num_leaves': 380,
                      'objective': 'huber',
                      'max_depth': 15,
                      'learning_rate': 0.12,
                      "boosting_type": "gbdt",
                      "subsample_freq": 1,
                      "subsample": 0.8,
                      "metric": 'mae',
                      "verbosity": -1,
                      'lambda_l1': 0.8,
                      'lambda_l2': 0.2,
                      'feature_fraction': 0.7,
                      }
        X_2 = pd.read_csv('champs-scalar-coupling/neighborhood/2J/train_2J_v1.csv').drop(
            ['type', 'scalar_coupling_constant'], axis=1)
        X_t = X.merge(
            X_2, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        X_t = X_t[X_t['type'] == t]
        index_train = X_t['id']
        X_t = X_t.drop(['atom_index_0', 'atom_index_1',
                        'atom_index_A', 'id', 'type'], axis=1)
        y_t = X_short.loc[index_train, 'target']

        X_2_test = pd.read_csv(
            'champs-scalar-coupling/neighborhood/2J/test_2J_v1.csv').drop(['type'], axis=1)
        X_test_t = X_test.merge(
            X_2_test, on=['atom_index_0', 'atom_index_1', 'molecule_name'], how='right')
        X_test_t = X_test_t[X_test_t['type'] == t]
        index_test = X_test_t['id']
        X_test_t = X_test_t.drop(
            ['atom_index_0', 'atom_index_1', 'atom_index_A', 'id', 'type'], axis=1)
        X_t_train_test = pd.get_dummies(
            pd.concat([X_t, X_test_t]), columns=['atom_A'])
        X_t_train_test['type'] = 0

        X_t = X_t_train_test[:len(X_t)]
        X_test_t = X_t_train_test[len(X_t):].drop('molecule_name', axis=1)
        folds = GroupKFold(n_splits=n_fold)
        groups = X_t['molecule_name']
        X_t = X_t.drop('molecule_name', axis=1)
        folds.get_n_splits(X_t, y_t, groups)

        result_dict_lgb3 = train_model_regression(X=X_t[top_features[t] + ['type']], X_test=X_test_t[top_features[t] + ['type']], y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
                                                  verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
        X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
        X_short_test.loc[index_test,
                         'prediction'] = result_dict_lgb3['prediction']
        res.append((t, result_dict_lgb3))
# %%
res.append({'train': X_short, 'test': X_short_test})
gc.collect()
sub['scalar_coupling_constant'] = X_short_test['prediction']
sub.to_csv(
    'output/submission_qm9_10folds_param_tune_fc_stacking_top_feature.csv', index=False)
sub.head()

with open('output/pickle/res_dict_qm9_10folds_param_tune_fc_stacking_top_feature.pickle', 'wb') as f:
    pickle.dump(res, f)


# %%
# X_short_test
# sub
# #%%
# # fc stacking
# df = pd.read_csv('champs-scalar-coupling/scalar_coupling_contributions.csv')
# X_short = pd.DataFrame({'ind': list(X.index), 'type': X['type'].values, 'oof': [0] * len(X), 'target': df['fc'].values})
# res = []
# n_fold = 3
# n_estimators_default = 4000
# for t in ['1JHC','1JHN','2JHC','2JHH','2JHN','3JHC','3JHH','3JHN']:
#     print(f'Training of type {t}')
#     if t == '1JHC':
#         for n_bond in range(2,5):
#             print(f'Training of type {t}, bonds {n_bond}')
#             X_1JHC = pd.read_csv('champs-scalar-coupling/neighborhood/1JHC/train_1JHC_{}_bond.csv'.format(n_bond)).drop(['type','scalar_coupling_constant','dist'],axis=1)
#             X_t = X.merge(X_1JHC,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#             index_train = X_t['id']
#             X_t = X_t.drop(['atom_index_0','atom_index_1','id','type'],axis=1)
#             y_t = X_short.loc[index_train, 'target']
#
#             X_1JHC_test = pd.read_csv('champs-scalar-coupling/neighborhood/1JHC/test_1JHC_{}_bond.csv'.format(n_bond)).drop(['type','dist'],axis=1)
#             X_test_t = X_test.merge(X_1JHC_test,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#             index_test = X_test_t['id']
#             X_test_t = X_test_t.drop(['atom_index_0','atom_index_1','id','type'],axis=1)
#             X_t_train_test = pd.get_dummies(pd.concat([X_t,X_test_t]),columns=['atom_{}'.format(n) for n in range(2,n_bond+1)])
#             X_t_train_test['type']=0
#
#             X_t = X_t_train_test[:len(X_t)]
#             X_test_t = X_t_train_test[len(X_t):].drop('molecule_name',axis=1)
#             folds = GroupKFold(n_splits=n_fold)
#             groups = X_t['molecule_name']
#             X_t = X_t.drop('molecule_name',axis=1)
#             folds.get_n_splits(X_t,y_t,groups)
#
#             result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
#                                                                   verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default,groups=groups)
#             X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
#             X_short_test.loc[index_test, 'prediction'] = result_dict_lgb3['prediction']
#             res.append((t,n_bond,result_dict_lgb3))
#
#     elif t == '1JHN':
#         for n_bond in range(2,5):
#             print(f'Training of type {t}, bonds {n_bond}')
#             X_1JHN = pd.read_csv('champs-scalar-coupling/neighborhood/1JHN/train_1JHN_{}_bond.csv'.format(n_bond)).drop(['type','scalar_coupling_constant','dist'],axis=1)
#             X_t = X.merge(X_1JHN,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#             index_train = X_t['id']
#             X_t = X_t.drop(['atom_index_0','atom_index_1','id'],axis=1)
#             y_t = X_short.loc[index_train, 'target']
#
#             X_1JHN_test = pd.read_csv('champs-scalar-coupling/neighborhood/1JHN/test_1JHN_{}_bond.csv'.format(n_bond)).drop(['type','dist'],axis=1)
#             X_test_t = X_test.merge(X_1JHN_test,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#             index_test = X_test_t['id']
#             X_test_t = X_test_t.drop(['atom_index_0','atom_index_1','id'],axis=1)
#             X_t_train_test = pd.get_dummies(pd.concat([X_t,X_test_t]),columns=['atom_{}'.format(n) for n in range(2,n_bond+1)])
#             X_t_train_test['type']=0
#
#             X_t = X_t_train_test[:len(X_t)]
#             X_test_t = X_t_train_test[len(X_t):].drop('molecule_name',axis=1)
#             folds = GroupKFold(n_splits=n_fold)
#             groups = X_t['molecule_name']
#             X_t = X_t.drop('molecule_name',axis=1)
#             folds.get_n_splits(X_t,y_t,groups)
#
#             result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
#                                                                   verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
#             X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
#             X_short_test.loc[index_test, 'prediction'] = result_dict_lgb3['prediction']
#             res.append((t,n_bond,result_dict_lgb3))
#
#     elif t in ['3JHC','3JHH','3JHN']:
#         thisType = t
#         X_3 = pd.read_csv('champs-scalar-coupling/neighborhood/{}/train_{}_v1.csv'.format(thisType,thisType)).drop(['type','scalar_coupling_constant'],axis=1)
#         X_t = X.merge(X_3,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#         index_train = X_t['id']
#         X_t = X_t.drop(['atom_index_0','atom_index_1','atom_index_A','atom_index_B','id','type'],axis=1)
#         y_t = X_short.loc[index_train, 'target']
#
#         X_3_test = pd.read_csv('champs-scalar-coupling/neighborhood/{}/test_{}_v1.csv'.format(thisType,thisType)).drop(['type'],axis=1)
#         X_test_t = X_test.merge(X_3_test,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#         index_test = X_test_t['id']
#         X_test_t = X_test_t.drop(['atom_index_0','atom_index_1','atom_index_A','atom_index_B','id','type'],axis=1)
#         X_t_train_test = pd.get_dummies(pd.concat([X_t,X_test_t]),columns=['atom_A','atom_B'])
#         X_t_train_test['type']=0
#
#         X_t = X_t_train_test[:len(X_t)]
#         X_test_t = X_t_train_test[len(X_t):].drop('molecule_name',axis=1)
#         folds = GroupKFold(n_splits=n_fold)
#         groups = X_t['molecule_name']
#         X_t = X_t.drop('molecule_name',axis=1)
#         folds.get_n_splits(X_t,y_t,groups)
#
#         result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
#                                                               verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
#         X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
#         X_short_test.loc[index_test, 'prediction'] = result_dict_lgb3['prediction']
#         res.append((t,result_dict_lgb3))
#
#     elif t in ['2JHC','2JHH','2JHN']:
#         X_2 = pd.read_csv('champs-scalar-coupling/neighborhood/2J/train_2J_v1.csv').drop(['type','scalar_coupling_constant'],axis=1)
#         X_t = X.merge(X_2,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#         X_t = X_t[X_t['type']==t]
#         index_train = X_t['id']
#         X_t = X_t.drop(['atom_index_0','atom_index_1','atom_index_A','id','type'],axis=1)
#         y_t = X_short.loc[index_train, 'target']
#
#         X_2_test = pd.read_csv('champs-scalar-coupling/neighborhood/2J/test_2J_v1.csv').drop(['type'],axis=1)
#         X_test_t = X_test.merge(X_2_test,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#         X_test_t = X_test_t[X_test_t['type']==t]
#         index_test = X_test_t['id']
#         X_test_t = X_test_t.drop(['atom_index_0','atom_index_1','atom_index_A','id','type'],axis=1)
#         X_t_train_test = pd.get_dummies(pd.concat([X_t,X_test_t]),columns=['atom_A'])
#         X_t_train_test['type']=0
#
#         X_t = X_t_train_test[:len(X_t)]
#         X_test_t = X_t_train_test[len(X_t):].drop('molecule_name',axis=1)
#         folds = GroupKFold(n_splits=n_fold)
#         groups = X_t['molecule_name']
#         X_t = X_t.drop('molecule_name',axis=1)
#         folds.get_n_splits(X_t,y_t,groups)
#
#         result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
#                                                               verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
#         X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
#         X_short_test.loc[index_test, 'prediction'] = result_dict_lgb3['prediction']
#         res.append((t,result_dict_lgb3))
# gc.collect()
# # sub['scalar_coupling_constant'] = X_short_test['prediction']
# # sub.to_csv('output/submission_giba_bonds_qm9_10folds_huber_R3.csv', index=False)
# # sub.head()
# #
#
# with open('output/pickle/res_digt_giba_bonds_qm9_10folds_huber_R3_oof_fc.pickle','wb') as f:
#     pickle.dump(d+[{'train':X_short,'test':X_short_test}],f)
#
# X['oof_fc'] = X_short['oof']
# X_test['oof_fc'] = X_short_test['prediction']
#
# d = res.copy()
# top_features = {}
# feature_num = 60
# for x in d:
#     if x[0] == '1JHC':
#         if '1JHC' not in top_features:
#             top_features['1JHC'] = {}
#         top_features['1JHC'][x[1]] = x[2]['feature_importance'].groupby('feature').mean().sort_values(by='importance',ascending=False)[:feature_num].reset_index()['feature'].tolist()+['oof_fc']
#     elif x[0] == '1JHN':
#         if '1JHN' not in top_features:
#             top_features['1JHN'] = {}
#         top_features['1JHN'][x[1]] = x[2]['feature_importance'].groupby('feature').mean().sort_values(by='importance',ascending=False)[:feature_num].reset_index()['feature'].tolist()+['oof_fc']
#     else:
#         top_features[x[0]] = x[1]['feature_importance'].groupby('feature').mean().sort_values(by='importance',ascending=False)[:feature_num].reset_index()['feature'].tolist()+['oof_fc']
#
# data = {}
# for x in d:
#     if x[0] == '1JHC':
#         if '1JHC' not in data:
#             data['1JHC'] = {}
#         data['1JHC'][x[1]] = x[2]
#     elif x[0] == '1JHN':
#         if '1JHN' not in data:
#             data['1JHN'] = {}
#         data['1JHN'][x[1]] = x[2]
#     else:
#         data[x[0]] = x[1]
# a = X_short.copy()
# b = X_short_test.copy()
# res = []
# n_fold = 5
# n_estimators_default = 8000
# X_short = pd.DataFrame({'ind': list(X.index), 'type': X['type'].values, 'oof': [0] * len(X), 'target': y.values})
# X_short_test = pd.DataFrame({'ind': list(X_test.index), 'type': X_test['type'].values, 'prediction': [0] * len(X_test)})
#
# for t in ['1JHC','1JHN','2JHC','2JHH','2JHN','3JHC','3JHH','3JHN']:
#     print(f'Training of type {t}')
#     if t == '1JHC':
#         for n_bond in range(2,5):
#             print(f'Training of type {t}, bonds {n_bond}')
#             X_1JHC = pd.read_csv('champs-scalar-coupling/neighborhood/1JHC/train_1JHC_{}_bond.csv'.format(n_bond)).drop(['type','scalar_coupling_constant','dist'],axis=1)
#             X_t = X.merge(X_1JHC,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#             index_train = X_t['id']
#             X_t = X_t.drop(['atom_index_0','atom_index_1','id','type'],axis=1)
#             y_t = X_short.loc[index_train, 'target']
#
#             X_1JHC_test = pd.read_csv('champs-scalar-coupling/neighborhood/1JHC/test_1JHC_{}_bond.csv'.format(n_bond)).drop(['type','dist'],axis=1)
#             X_test_t = X_test.merge(X_1JHC_test,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#             index_test = X_test_t['id']
#             X_test_t = X_test_t.drop(['atom_index_0','atom_index_1','id','type'],axis=1)
#             X_t_train_test = pd.get_dummies(pd.concat([X_t,X_test_t]),columns=['atom_{}'.format(n) for n in range(2,n_bond+1)])
#             X_t_train_test['type']=0
#
#             X_t = X_t_train_test[:len(X_t)]
#             X_test_t = X_t_train_test[len(X_t):].drop('molecule_name',axis=1)
#             folds = GroupKFold(n_splits=n_fold)
#             groups = X_t['molecule_name']
#             X_t = X_t.drop('molecule_name',axis=1)
#             folds.get_n_splits(X_t,y_t,groups)
#
#             result_dict_lgb3 = train_model_regression(X=X_t[top_features[t][n_bond]+['type']], X_test=X_test_t[top_features[t][n_bond]+['type']], y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
#                                                                   verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default,groups=groups)
#             X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
#             X_short_test.loc[index_test, 'prediction'] = result_dict_lgb3['prediction']
#             res.append((t,n_bond,result_dict_lgb3))
#
#     elif t == '1JHN':
#         for n_bond in range(2,5):
#             print(f'Training of type {t}, bonds {n_bond}')
#             X_1JHN = pd.read_csv('champs-scalar-coupling/neighborhood/1JHN/train_1JHN_{}_bond.csv'.format(n_bond)).drop(['type','scalar_coupling_constant','dist'],axis=1)
#             X_t = X.merge(X_1JHN,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#             index_train = X_t['id']
#             X_t = X_t.drop(['atom_index_0','atom_index_1','id'],axis=1)
#             y_t = X_short.loc[index_train, 'target']
#
#             X_1JHN_test = pd.read_csv('champs-scalar-coupling/neighborhood/1JHN/test_1JHN_{}_bond.csv'.format(n_bond)).drop(['type','dist'],axis=1)
#             X_test_t = X_test.merge(X_1JHN_test,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#             index_test = X_test_t['id']
#             X_test_t = X_test_t.drop(['atom_index_0','atom_index_1','id'],axis=1)
#             X_t_train_test = pd.get_dummies(pd.concat([X_t,X_test_t]),columns=['atom_{}'.format(n) for n in range(2,n_bond+1)])
#             X_t_train_test['type']=0
#
#             X_t = X_t_train_test[:len(X_t)]
#             X_test_t = X_t_train_test[len(X_t):].drop('molecule_name',axis=1)
#             folds = GroupKFold(n_splits=n_fold)
#             groups = X_t['molecule_name']
#             X_t = X_t.drop('molecule_name',axis=1)
#             folds.get_n_splits(X_t,y_t,groups)
#
#             result_dict_lgb3 = train_model_regression(X=X_t[top_features[t][n_bond]+['type']], X_test=X_test_t[top_features[t][n_bond]+['type']], y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
#                                                                   verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
#             X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
#             X_short_test.loc[index_test, 'prediction'] = result_dict_lgb3['prediction']
#             res.append((t,n_bond,result_dict_lgb3))
#
#     elif t in ['3JHC','3JHH','3JHN']:
#         thisType = t
#         X_3 = pd.read_csv('champs-scalar-coupling/neighborhood/{}/train_{}_v1.csv'.format(thisType,thisType)).drop(['type','scalar_coupling_constant'],axis=1)
#         X_t = X.merge(X_3,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#         index_train = X_t['id']
#         X_t = X_t.drop(['atom_index_0','atom_index_1','atom_index_A','atom_index_B','id','type'],axis=1)
#         y_t = X_short.loc[index_train, 'target']
#
#         X_3_test = pd.read_csv('champs-scalar-coupling/neighborhood/{}/test_{}_v1.csv'.format(thisType,thisType)).drop(['type'],axis=1)
#         X_test_t = X_test.merge(X_3_test,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#         index_test = X_test_t['id']
#         X_test_t = X_test_t.drop(['atom_index_0','atom_index_1','atom_index_A','atom_index_B','id','type'],axis=1)
#         X_t_train_test = pd.get_dummies(pd.concat([X_t,X_test_t]),columns=['atom_A','atom_B'])
#         X_t_train_test['type']=0
#
#         X_t = X_t_train_test[:len(X_t)]
#         X_test_t = X_t_train_test[len(X_t):].drop('molecule_name',axis=1)
#         folds = GroupKFold(n_splits=n_fold)
#         groups = X_t['molecule_name']
#         X_t = X_t.drop('molecule_name',axis=1)
#         folds.get_n_splits(X_t,y_t,groups)
#
#         result_dict_lgb3 = train_model_regression(X=X_t[top_features[t]+['type']], X_test=X_test_t[top_features[t]+['type']], y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
#                                                               verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
#         X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
#         X_short_test.loc[index_test, 'prediction'] = result_dict_lgb3['prediction']
#         res.append((t,result_dict_lgb3))
#
#     elif t in ['2JHC','2JHH','2JHN']:
#         X_2 = pd.read_csv('champs-scalar-coupling/neighborhood/2J/train_2J_v1.csv').drop(['type','scalar_coupling_constant'],axis=1)
#         X_t = X.merge(X_2,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#         X_t = X_t[X_t['type']==t]
#         index_train = X_t['id']
#         X_t = X_t.drop(['atom_index_0','atom_index_1','atom_index_A','id','type'],axis=1)
#         y_t = X_short.loc[index_train, 'target']
#
#         X_2_test = pd.read_csv('champs-scalar-coupling/neighborhood/2J/test_2J_v1.csv').drop(['type'],axis=1)
#         X_test_t = X_test.merge(X_2_test,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#         X_test_t = X_test_t[X_test_t['type']==t]
#         index_test = X_test_t['id']
#         X_test_t = X_test_t.drop(['atom_index_0','atom_index_1','atom_index_A','id','type'],axis=1)
#         X_t_train_test = pd.get_dummies(pd.concat([X_t,X_test_t]),columns=['atom_A'])
#         X_t_train_test['type']=0
#
#         X_t = X_t_train_test[:len(X_t)]
#         X_test_t = X_t_train_test[len(X_t):].drop('molecule_name',axis=1)
#         folds = GroupKFold(n_splits=n_fold)
#         groups = X_t['molecule_name']
#         X_t = X_t.drop('molecule_name',axis=1)
#         folds.get_n_splits(X_t,y_t,groups)
#
#         result_dict_lgb3 = train_model_regression(X=X_t[top_features[t]+['type']], X_test=X_test_t[top_features[t]+['type']], y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,
#                                                               verbose=1000, early_stopping_rounds=200, n_estimators=n_estimators_default, groups=groups)
#         X_short.loc[index_train, 'oof'] = result_dict_lgb3['oof']
#         X_short_test.loc[index_test, 'prediction'] = result_dict_lgb3['prediction']
#         res.append((t,result_dict_lgb3))
#     res.append({'train':X_short,'test':X_short_test})
#
#
#
#
#
#
# d
# d[type]
# i = 0
# for t in ['1JHC','1JHN','2JHC','2JHH','2JHN','3JHC','3JHH','3JHN']:
#     print(f'Training of type {t}')
#     if t == '1JHC':
#         for n_bond in range(2,5):
#             print(f'Training of type {t}, bonds {n_bond}')
#             X_1JHC = pd.read_csv('champs-scalar-coupling/neighborhood/1JHC/train_1JHC_{}_bond.csv'.format(n_bond)).drop(['type','scalar_coupling_constant','dist'],axis=1)
#             X_t = X.merge(X_1JHC,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#             index_train = X_t['id']
#             X_1JHC_test = pd.read_csv('champs-scalar-coupling/neighborhood/1JHC/test_1JHC_{}_bond.csv'.format(n_bond)).drop(['type','dist'],axis=1)
#             X_test_t = X_test.merge(X_1JHC_test,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#             index_test = X_test_t['id']
#
#             X_short.loc[index_train, 'oof'] = data[t][n_bond]['oof']
#             X_short_test.loc[index_test, 'prediction'] = data[t][n_bond]['prediction']
#
#     elif t == '1JHN':
#         for n_bond in range(2,5):
#             print(f'Training of type {t}, bonds {n_bond}')
#             X_1JHN = pd.read_csv('champs-scalar-coupling/neighborhood/1JHN/train_1JHN_{}_bond.csv'.format(n_bond)).drop(['type','scalar_coupling_constant','dist'],axis=1)
#             X_t = X.merge(X_1JHN,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#             index_train = X_t['id']
#
#             X_1JHN_test = pd.read_csv('champs-scalar-coupling/neighborhood/1JHN/test_1JHN_{}_bond.csv'.format(n_bond)).drop(['type','dist'],axis=1)
#             X_test_t = X_test.merge(X_1JHN_test,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#             index_test = X_test_t['id']
#
#             X_short.loc[index_train, 'oof'] = data[t][n_bond]['oof']
#             X_short_test.loc[index_test, 'prediction'] = data[t][n_bond]['prediction']
#
#
#
#     elif t in ['3JHC','3JHH','3JHN']:
#         thisType = t
#         X_3 = pd.read_csv('champs-scalar-coupling/neighborhood/{}/train_{}_v1.csv'.format(thisType,thisType)).drop(['type','scalar_coupling_constant'],axis=1)
#         X_t = X.merge(X_3,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#         index_train = X_t['id']
#
#
#         X_3_test = pd.read_csv('champs-scalar-coupling/neighborhood/{}/test_{}_v1.csv'.format(thisType,thisType)).drop(['type'],axis=1)
#         X_test_t = X_test.merge(X_3_test,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#         index_test = X_test_t['id']
#
#
#         X_short.loc[index_train, 'oof'] = data[t]['oof']
#         X_short_test.loc[index_test, 'prediction'] = data[t]['prediction']
#         res.append((t,result_dict_lgb3))
#
#     elif t in ['2JHC','2JHH','2JHN']:
#         X_2 = pd.read_csv('champs-scalar-coupling/neighborhood/2J/train_2J_v1.csv').drop(['type','scalar_coupling_constant'],axis=1)
#         X_t = X.merge(X_2,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#         X_t = X_t[X_t['type']==t]
#         index_train = X_t['id']
#
#
#         X_2_test = pd.read_csv('champs-scalar-coupling/neighborhood/2J/test_2J_v1.csv').drop(['type'],axis=1)
#         X_test_t = X_test.merge(X_2_test,on=['atom_index_0','atom_index_1','molecule_name'],how='right')
#         X_test_t = X_test_t[X_test_t['type']==t]
#         index_test = X_test_t['id']
#
#         X_short.loc[index_train, 'oof'] = data[t]['oof']
#         X_short_test.loc[index_test, 'prediction'] = data[t]['prediction']
#
# res.append({'train':X_short,'test':X_short_test})
# gc.collect()
# sub['scalar_coupling_constant'] = X_short_test['prediction']
# sub.to_csv('output/submission_5folds_fc_stacking.csv', index=False)
# sub.head()
#
# with open('output/pickle/res_dict_5folds_fc_stacking.pickle','wb') as f:
#     pickle.dump(res,f)


X[['id', 'atom_index_0', 'atom_index_1', 'molecule_name', 'type', 'oof_fc']
  ].to_pickle('output/pickle/train_oof_fc_ver1.pickle')
X_test[['id', 'atom_index_0', 'atom_index_1', 'molecule_name', 'type',
        'oof_fc']].to_pickle('output/pickle/test_oof_fc_ver1.pickle')


# 2JHN
n_estimators_default = 8000
params = {'num_leaves': 380,
          'objective': 'huber',
          'max_depth': 30,
          'learning_rate': 0.1,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.8,
          "metric": 'mae',
          "verbosity": -1,
          'lambda_l1': 0.8,
          'lambda_l2': 0.2,
          'feature_fraction': 0.7,
          }

# 2JHH
n_estimators_default = 8000
params = {'num_leaves': 380,
          'objective': 'huber',
          'max_depth': 15,
          'learning_rate': 0.12,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.8,
          "metric": 'mae',
          "verbosity": -1,
          'lambda_l1': 0.8,
          'lambda_l2': 0.2,
          'feature_fraction': 0.7,
          }

# 2JHC
n_estimators_default = 8000
params = {'num_leaves': 380,
          'objective': 'huber',
          'max_depth': 15,
          'learning_rate': 0.12,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.8,
          "metric": 'mae',
          "verbosity": -1,
          'lambda_l1': 0.8,
          'lambda_l2': 0.2,
          'feature_fraction': 0.7,
          }

# 3JHH
n_estimators_default = 8000
params = {'num_leaves': 380,
          'objective': 'huber',
          'max_depth': 15,
          'learning_rate': 0.12,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.8,
          "metric": 'mae',
          "verbosity": -1,
          'lambda_l1': 0.8,
          'lambda_l2': 0.2,
          'feature_fraction': 0.7,
          }

# 3JHN
n_estimators_default = 8000
params = {'num_leaves': 500,
          'objective': 'huber',
          'max_depth': 15,
          'learning_rate': 0.12,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.8,
          "metric": 'mae',
          "verbosity": -1,
          'lambda_l1': 0.8,
          'lambda_l2': 0.2,
          'feature_fraction': 0.7,
          }
