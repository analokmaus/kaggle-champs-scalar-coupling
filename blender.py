import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
from tqdm import tqdm

from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split

#%%
DATA_PATH = Path('/Volumes/T5/CHAMPS')
CHARM_PATH = DATA_PATH/'charm_dataset'
RESULTS_PATH = Path('results')

train = pd.read_csv(DATA_PATH/'train.csv')
test = pd.read_csv(DATA_PATH/'test.csv')
sub = pd.read_csv(DATA_PATH/'sample_submission.csv')
res_paths = list(RESULTS_PATH.glob('*.pkl'))
print(train.shape, test.shape)
train_idx = {x:train.loc[train['type'] == x, 'id'].values
             for x in train['type'].unique()}
test_idx = {x:test.loc[test['type'] == x, 'id'].values
             for x in test['type'].unique()}
for ctype in train_idx.keys():
    print(ctype, len(train_idx[ctype]))
for ctype in test_idx.keys():
    print(ctype, len(test_idx[ctype]))


def clean_separate_1J(ctype, bond, r):
    print(f'Processing {ctype} {bond}')
    all_oof = np.zeros(len(train), dtype=np.float16)
    all_pred = np.zeros(len(test), dtype=np.float16)
    train_features = pd.read_csv(CHARM_PATH/'neighborhood'/f'{ctype}'/f'train_{ctype}_{bond}_bond.csv', usecols=[0,1,2])
    test_features = pd.read_csv(CHARM_PATH/'neighborhood'/f'{ctype}'/f'test_{ctype}_{bond}_bond.csv', usecols=[0,1,2])
    _train_idx = train.merge(train_features, on=['molecule_name', 'atom_index_0', 'atom_index_1'], how='right')['id'].values
    _test_idx = test.merge(test_features, on=['molecule_name', 'atom_index_0', 'atom_index_1'], how='right')['id'].values
    _test_idx = _test_idx - 4658147
    assert len(_train_idx) == len(r['oof'])
    assert len(_test_idx) == len(r['prediction'])
    all_oof[_train_idx] = r['oof']
    all_pred[_test_idx] = r['prediction']
    return all_oof, all_pred

def clean_result(ctype, res):
    print(f'Processing {ctype}')
    all_oof = np.zeros(len(train), dtype=np.float16)
    all_pred = np.zeros(len(test), dtype=np.float16)
    _train_idx = train_idx[ctype]
    _test_idx = test_idx[ctype]
    _test_idx = _test_idx - 4658147
    assert len(_train_idx) == len(res['oof'])
    assert len(_test_idx) == len(res['prediction'])
    all_oof[_train_idx] = res['oof']
    all_pred[_test_idx] = res['prediction']
    return all_oof, all_pred


#%%
oof_dict = {}
pred_dict = {}
def parse_result(path):
    res = pickle.load(open(path, 'rb'))
    oof_frags = []
    pred_frags = []
    for res_obj in res:
        if not isinstance(res_obj, tuple):
            print('skipped')
            continue
        if len(res_obj) == 2:
            oof_frag, pred_frag = clean_result(*res_obj)
            oof_frags.append(oof_frag)
            pred_frags.append(pred_frag)
        elif len(res_obj) == 3:
            oof_frag, pred_frag = clean_separate_1J(*res_obj)
            oof_frags.append(oof_frag)
            pred_frags.append(pred_frag)
        else:
            raise ValueError('wrong structure!')
    oof = np.array(oof_frags).sum(axis=0)
    pred = np.array(pred_frags).sum(axis=0)
    return oof, pred

base_oof, base_pred = parse_result(RESULTS_PATH/'res_dict_giba_R3_10folds.pickle')
for ctype in ['1JHC', '1JHN', '2J', '3JHC', '3JHN', '3JHH']:
    for surfix in ['', '_DNN']:
        _ctype = ctype + surfix
        res_path = RESULTS_PATH/f'{_ctype}.pkl'
        if not res_path.exists():
            print(str(res_path), 'not found.')
            continue
        oof_dict[_ctype], pred_dict[_ctype] = parse_result(RESULTS_PATH/f'{_ctype}.pkl')


#%%
X = pd.DataFrame(base_oof, columns=['charmq'])
X_pred = pd.DataFrame(base_pred, columns=['charmq'])
for ctype in oof_dict.keys():
    X[ctype] = oof_dict[ctype]
    X_pred[ctype] = pred_dict[ctype]
X['new'] = X[['1JHC', '1JHN', '2J', '3JHC', '3JHN', '3JHH']].sum(axis=1)
X_pred['new'] = X_pred[['1JHC', '1JHN', '2J', '3JHC', '3JHN', '3JHH']].sum(axis=1)
X['target'] = train['scalar_coupling_constant'].values
X.head(10)

def evaluate(col, ctypes):
    maes = []
    for ctype in ctypes:
        _X = X.loc[train_idx[ctype], col]
        _y = train.loc[train_idx[ctype], 'scalar_coupling_constant']
        mae = MAE(_y, _X)
        logmae = np.log(mae)
        maes.append(logmae)
        print('logMAE for {:.6}: {:.6} ({:.6})'.format(ctype, logmae, mae))
    print(f'overall logMAE {np.mean(maes)}')

def compare(cols, ctypes):
    idx = np.concatenate([train_idx[x] for x in ctypes])
    return X.loc[idx, cols]

compare(['1JHN', '1JHN_DNN', 'target'], ['1JHN'])
evaluate('charmq', train_idx.keys())
evaluate('new', train_idx.keys())


#%%
def make_X_stack(x, cols, idx):
    X_train, X_valid, y_train, y_valid = train_test_split(
        x.loc[idx, cols], x.loc[idx, 'target'],
        test_size=0.2, random_state=2019)
    return X_train, X_valid, y_train, y_valid


def auto_stack(cols, ctypes):
    _train_idx = np.concatenate([train_idx[x] for x in ctypes])
    _test_idx = np.concatenate([test_idx[x] - 4658147 for x in ctypes])

    X_train, X_valid, y_train, y_valid = make_X_stack(X, cols, _train_idx)

    ridge = Ridge()
    ridge.fit(X_train, y_train)

    print('before stacking')
    for col in X_train.columns:
        mae = MAE(X_train[col], y_train)
        logmae = np.log(mae)
        print(col, f'\t{logmae:.6}({mae:.6})')

    mae = MAE(ridge.predict(X_valid), y_valid)
    logmae = np.log(mae)
    print('-> after stacking', f'\t{logmae:.6}({mae:.6})')

    stack_oof = ridge.predict(X.loc[_train_idx, cols])
    stack_pred = ridge.predict(X_pred.loc[_test_idx, cols])

    return stack_oof, stack_pred

oof_1jhn, pred_1jhn = auto_stack(['1JHN', '1JHN_DNN'], ['1JHN'])
oof_1jhc, pred_1jhc = auto_stack(['1JHC', '1JHC_DNN'], ['1JHC'])
oof_2jhc, pred_2jhc = auto_stack(['2J', 'charmq'], ['2JHC'])
oof_2jhh, pred_2jhh = auto_stack(['2J', 'charmq'], ['2JHH'])


#%% neural stacking
# train_idx.keys()
for ctype in ['1JHN']:
    print('now cooking', ctype)
    train_nn = pd.read_csv(RESULTS_PATH/'schnet_15'/f'schnet_{ctype}_train.csv', index_col=0)
    valid_nn = pd.read_csv(RESULTS_PATH/'schnet_15'/f'schnet_{ctype}_valid.csv', index_col=0)
    test_nn = pd.read_csv(RESULTS_PATH/'schnet_15'/f'schnet_{ctype}_test.csv', index_col=0)

    target_col = ['charmq', 'new'] + [ctype+'_DNN'] if ctype+'_DNN' in X.columns else []
    X_train, y_train = X.loc[train_nn.index, target_col], X.loc[train_nn.index, 'target']
    X_valid, y_valid = X.loc[valid_nn.index, target_col], X.loc[valid_nn.index, 'target']
    X_test= X_pred.loc[test_nn.index - 4658147, target_col]

    X_train['schnet_15'] = train_nn['scalar_coupling_constant']
    X_valid['schnet_15'] = valid_nn['scalar_coupling_constant']
    X_test['schnet_15'] = test_nn['scalar_coupling_constant']

    display(X_train.head())

    ridge = Ridge()
    ridge.fit(X_train, y_train)

    print('before stacking')
    for col in X_train.columns:
        mae = MAE(X_train[col], y_train)
        logmae = np.log(mae)
        print(col, f'\t{logmae:.6}({mae:.6})')

    mae = MAE(ridge.predict(X_valid), y_valid)
    logmae = np.log(mae)
    print('-> after stacking', f'\t{logmae:.6}({mae:.6})')

np.max(y_train)
#%%
def rewrite(sub, pred, ctypes):
    _test_idx = np.concatenate([test_idx[x] - 4658147 for x in ctypes])
    assert len(_test_idx) == len(pred)
    sub.loc[_test_idx, 'scalar_coupling_constant'] = pred
    return sub
# best_sub = rewrite(best_sub, stack_pred, ['1JHN'])


# # 2243 config
# best_sub = pd.read_csv(RESULTS_PATH/'2184.csv', index_col=0)
# for ctype in ['1JHC', '2JHC', '2JHH', '3JHC']:
#     new = pd.read_csv(RESULTS_PATH/'schnet_abci'/f'{ctype}.csv', index_col=0)
#     schnet_vals = new['scalar_coupling_constant'].values
#     origin_vals = best_sub.loc[new.index, 'scalar_coupling_constant'].values
#     update_vals = 0.5 * schnet_vals + 0.5 * origin_vals
#     print(ctype, origin_vals,
#           '->', update_vals)
#     best_sub.loc[new.index, 'scalar_coupling_constant'] = update_vals

# # 2362config
# best_sub = pd.read_csv(RESULTS_PATH/'2243.csv', index_col=0)
# for ctype in ['1JHC', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']:
#     NEW_RATIO = 0.6
#     new = pd.read_csv(RESULTS_PATH/'schnet_tyamaguchi'/f'{ctype}.csv', index_col=0)
#     new_vals = new['scalar_coupling_constant'].values
#     origin_vals = best_sub.loc[new.index, 'scalar_coupling_constant'].values
#     update_vals = NEW_RATIO * new_vals + (1 - NEW_RATIO) * origin_vals
#     print(ctype, origin_vals,
#           '->', update_vals)
#     best_sub.loc[new.index, 'scalar_coupling_constant'] = update_vals

# # 2386config
# best_sub = pd.read_csv(RESULTS_PATH/'2362.csv', index_col=0)
# for ctype in ['ALL']:
#     NEW_RATIO = 1 / 9
#     new = pd.read_csv(RESULTS_PATH/'schnet_tyamaguchi'/f'{ctype}.csv', index_col=0)
#     new_vals = new['scalar_coupling_constant'].values
#     origin_vals = best_sub.loc[new.index, 'scalar_coupling_constant'].values
#     update_vals = NEW_RATIO * new_vals + (1 - NEW_RATIO) * origin_vals
#     # print(new_vals)
#     print(ctype, origin_vals,
#           '->', update_vals)
#     best_sub.loc[new.index, 'scalar_coupling_constant'] = update_vals

# 2396config
# best_sub = pd.read_csv(RESULTS_PATH/'2386.csv', index_col=0)
# for ctype in ['1JHC', '1JHN', '2JHH', '3JHC', '3JHH']:
#     NEW_RATIO = 0.4
#     new = pd.read_csv(RESULTS_PATH/'schnet_new0825'/f'{ctype}.csv', index_col=0)
#     new_vals = new['scalar_coupling_constant'].values
#     origin_vals = best_sub.loc[new.index, 'scalar_coupling_constant'].values
#     update_vals = NEW_RATIO * new_vals + (1 - NEW_RATIO) * origin_vals
#     # print(new_vals)
#     print(ctype, origin_vals,
#           '->', update_vals)
#     best_sub.loc[new.index, 'scalar_coupling_constant'] = update_vals

# # 2416config
# best_sub = pd.read_csv(RESULTS_PATH/'2396.csv', index_col=0)
# for ctype in ['ALL']:
#     NEW_RATIO = 1/9
#     new = pd.read_csv(RESULTS_PATH/'schnet_new0825'/f'{ctype}.csv', index_col=0)
#     new_vals = new['scalar_coupling_constant'].values
#     origin_vals = best_sub.loc[new.index, 'scalar_coupling_constant'].values
#     update_vals = NEW_RATIO * new_vals + (1 - NEW_RATIO) * origin_vals
#     # print(new_vals)
#     print(ctype, origin_vals,
#           '->', update_vals)
#     best_sub.loc[new.index, 'scalar_coupling_constant'] = update_vals

# # 2411config
# best_sub = pd.read_csv(RESULTS_PATH/'2416.csv', index_col=0)
# NEW_RATIO = 4/5
# new_all = pd.read_csv(RESULTS_PATH/'schnet_new0825'/f'ALL.csv', index_col=0)
# new = new_all.loc[test_idx['1JHC']]
# new_vals = new['scalar_coupling_constant'].values
# origin_vals = best_sub.loc[new.index, 'scalar_coupling_constant'].values
# update_vals = NEW_RATIO * new_vals + (1 - NEW_RATIO) * origin_vals
# print(ctype, origin_vals,
#       '->', update_vals)
# best_sub.loc[new.index, 'scalar_coupling_constant'] = update_vals

# # 2416config
# best_sub = pd.read_csv(RESULTS_PATH/'2396.csv', index_col=0)
# for ctype in ['ALL']:
#     NEW_RATIO = 1/9
#     new = pd.read_csv(RESULTS_PATH/'schnet_new0825'/f'{ctype}.csv', index_col=0)
#     new_vals = new['scalar_coupling_constant'].values
#     origin_vals = best_sub.loc[new.index, 'scalar_coupling_constant'].values
#     update_vals = NEW_RATIO * new_vals + (1 - NEW_RATIO) * origin_vals
#     # print(new_vals)
#     print(ctype, origin_vals,
#           '->', update_vals)
#     best_sub.loc[new.index, 'scalar_coupling_constant'] = update_vals

# # 2459config
# best_sub = pd.read_csv(RESULTS_PATH/'2416.csv', index_col=0)
# for ctype in ['1JHC', '2JHC', '2JHN', '3JHC', '3JHH', '3JHN']:
#     NEW_RATIO = 0.3
#     new = pd.read_csv(RESULTS_PATH/'schnet_depth10'/f'{ctype}.csv', index_col=1)
#     new_vals = new['scalar_coupling_constant'].values
#     origin_vals = best_sub.loc[new.index, 'scalar_coupling_constant'].values
#     update_vals = NEW_RATIO * new_vals + (1 - NEW_RATIO) * origin_vals
#     # print(new_vals)
#     print(ctype, origin_vals,
#           '->', update_vals)
#     best_sub.loc[new.index, 'scalar_coupling_constant'] = update_vals

# # 2463config
# best_sub = pd.read_csv(RESULTS_PATH/'2459.csv', index_col=0)
# for ctype in train_idx.keys():
#     NEW_RATIO = 0.2
#     new = pd.read_csv(RESULTS_PATH/'schnet_each_best'/f'{ctype}.csv', index_col=0)
#     if new.shape[1] == 2:
#         new = pd.read_csv(RESULTS_PATH/'schnet_each_best'/f'{ctype}.csv', index_col=1)
#     new_vals = new['scalar_coupling_constant'].values
#     origin_vals = best_sub.loc[new.index, 'scalar_coupling_constant'].values
#     update_vals = NEW_RATIO * new_vals + (1 - NEW_RATIO) * origin_vals
#     # print(new_vals)
#     print(ctype, origin_vals,
#           '->', update_vals)
#     best_sub.loc[new.index, 'scalar_coupling_constant'] = update_vals

# # 2495config
# best_sub = pd.read_csv(RESULTS_PATH/'2463.csv', index_col=0)
# for ctype in ['1JHC', '1JHN', '2JHC', '2JHH', '3JHC', '3JHN']:
#     NEW_RATIO = 0.3
#     new = pd.read_csv(RESULTS_PATH/'schnet_15'/f'schnet_{ctype}.csv', index_col=0)
#     if new.shape[1] == 2:
#         new = pd.read_csv(RESULTS_PATH/'schnet_15'/f'schnet_{ctype}.csv', index_col=1)
#     new_vals = new['scalar_coupling_constant'].values
#     origin_vals = best_sub.loc[new.index, 'scalar_coupling_constant'].values
#     update_vals = NEW_RATIO * new_vals + (1 - NEW_RATIO) * origin_vals
#     # print(new_vals)
#     print(ctype, origin_vals,
#           '->', update_vals)
#     best_sub.loc[new.index, 'scalar_coupling_constant'] = update_vals

# 2428config
# best_sub = pd.read_csv(RESULTS_PATH/'2495.csv', index_col=0)
# for ctype in train_idx.keys():
#     NEW_RATIO = 1/9
#     new = pd.read_csv(RESULTS_PATH/'weavenet'/f'weavenet_{ctype}.csv', index_col=0)
#     if new.shape[1] == 2:
#         new = pd.read_csv(RESULTS_PATH/'weavenet'/f'weavenet_{ctype}.csv', index_col=1)
#     new_vals = new['scalar_coupling_constant'].values
#     origin_vals = best_sub.loc[new.index, 'scalar_coupling_constant'].values
#     update_vals = NEW_RATIO * new_vals + (1 - NEW_RATIO) * origin_vals
#     print(ctype, origin_vals,
#           '->', update_vals)
#     best_sub.loc[new.index, 'scalar_coupling_constant'] = update_vals

# # 2490config
# best_sub = pd.read_csv(RESULTS_PATH/'2495.csv', index_col=0)
# NEW_RATIO = 1/9
# new = pd.read_csv(RESULTS_PATH/'best_public.csv', index_col=0)
# new_vals = new['scalar_coupling_constant'].values
# origin_vals = best_sub.loc[new.index, 'scalar_coupling_constant'].values
# update_vals = NEW_RATIO * new_vals + (1 - NEW_RATIO) * origin_vals
# print(origin_vals,
#       '->', update_vals)
# best_sub.loc[new.index, 'scalar_coupling_constant'] = update_vals

#2522config
best_sub = pd.read_csv(RESULTS_PATH/'2495.csv', index_col=0)
for ctype in train_idx.keys():
    NEW_RATIO = 0.2
    try:
        new = pd.read_csv(RESULTS_PATH/'schnet_20'/f'schnet_{ctype}.csv', index_col=0)
    except:
        print('skipped', ctype)
        continue
    if new.shape[1] == 2:
        new = pd.read_csv(RESULTS_PATH/'schnet_20'/f'schnet_{ctype}.csv', index_col=1)
    new_vals = new['scalar_coupling_constant'].values
    origin_vals = best_sub.loc[new.index, 'scalar_coupling_constant'].values
    update_vals = NEW_RATIO * new_vals + (1 - NEW_RATIO) * origin_vals
    # print(new_vals)
    print(ctype, origin_vals,
          '->', update_vals)
    print('diff_mean', np.mean(update_vals - origin_vals))
    best_sub.loc[new.index, 'scalar_coupling_constant'] = update_vals

# # 2522config
# best_sub = pd.read_csv(RESULTS_PATH/'2522.csv', index_col=0)
# for ctype in ['all']:
#     NEW_RATIO = 1/9
#     new = pd.read_csv(RESULTS_PATH/'schnet_last'/f'schnet_{ctype}.csv', index_col=0)
#     new_vals = new['scalar_coupling_constant'].values
#     origin_vals = best_sub.loc[new.index, 'scalar_coupling_constant'].values
#     update_vals = NEW_RATIO * new_vals + (1 - NEW_RATIO) * origin_vals
#     # print(new_vals)
#     print(ctype, origin_vals,
#           '->', update_vals)
#     print('diff_mean', np.mean(update_vals - origin_vals))
#     best_sub.loc[new.index, 'scalar_coupling_constant'] = update_vals


best_sub.to_csv('2495_8_(schnet_20)_2.csv', index=True)
