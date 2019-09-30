from common import *

from pathlib import Path
import pickle
import math
import random
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


import os
import shutil
import tempfile
import time
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold
from sklearn import metrics
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import chainer
import chainer_chemistry
from chainer.datasets.dict_dataset import DictDataset
from chainer import reporter
from chainer import functions as F
from chainer import links as L
from chainer_chemistry.links import SchNetUpdate
from chainer_chemistry.links import GraphLinear, GraphBatchNormalization
from chainer.iterators import OrderSampler
from chainer import optimizers
from chainer import training
from chainer.dataset import to_device
from chainer.training.extensions import Evaluator
from chainer import cuda
from chainer.training import make_extension
from chainer.training import triggers

import sklearn
from scipy.spatial import distance

# '''
DATA_PATH = Path('/Volumes/T5/CHAMPS/')
RESULT_PATH = 'results/chainer'
CTYPE = '3JHN'
BATCH_SIZE = 16
EARLY_STOPPING_ROUNDS = 10
NUM_LAYER = 10
# '''


#%%
def load_dataset(couple_type):

    train = pd.merge(pd.read_csv(DATA_PATH/'train.csv'),
                     pd.read_csv(DATA_PATH/'scalar_coupling_contributions.csv'))
    if couple_type != 'all':
        train = train[train['type']==couple_type]

    test = pd.read_csv(DATA_PATH/'test.csv')
    if couple_type != 'all':
        test = test[test['type']==couple_type]

    counts = train['molecule_name'].value_counts()
    moles = list(counts.index)

    random.seed(2019)
    random.shuffle(moles)

    num_train = int(len(moles) * 0.9)
    train_moles = sorted(moles[:num_train])
    valid_moles = sorted(moles[num_train:])
    test_moles = sorted(list(set(test['molecule_name'])))

    valid = train.query('molecule_name not in @train_moles')
    train = train.query('molecule_name in @train_moles')

    train.sort_values('id', inplace=True)
    valid.sort_values('id', inplace=True)
    test.sort_values('id', inplace=True)

    return train, valid, test, train_moles, valid_moles, test_moles

class Graph:

    def __init__(self, points_df, list_atoms, feature_col):

        self.points = points_df[['x', 'y', 'z']].values
        self.features = points_df[feature_col].values.astype(np.float32)

        self._dists = distance.cdist(self.points, self.points)

        self.adj = self._dists < 1.5
        self.num_nodes = len(points_df)

        self.atoms = points_df['atom']
        dict_atoms = {at: i for i, at in enumerate(list_atoms)}

        atom_index = [dict_atoms[atom] for atom in self.atoms]
        one_hot = np.identity(len(dict_atoms))[atom_index]

        bond = np.sum(self.adj, 1) - 1
        bonds = np.identity(len(dict_atoms))[bond - 1]

        self._array = np.concatenate([one_hot, bonds, self.features], axis=1).astype(np.float32)

    @property
    def input_array(self):
        return self._array

    @property
    def dists(self):
        return self._dists.astype(np.float32)

class SchNetUpdateBN(SchNetUpdate):

    def __init__(self, *args, **kwargs):
        super(SchNetUpdateBN, self).__init__(*args, **kwargs)
        with self.init_scope():
            self.bn = GraphBatchNormalization(args[0])

    def __call__(self, h, adj, **kwargs):
        v = self.linear[0](h)
        v = self.cfconv(v, adj)
        v = self.linear[1](v)
        v = F.softplus(v)
        v = self.linear[2](v)
        return h + self.bn(v)

class SchNet(chainer.Chain):

    def __init__(self, num_layer=3):
        super(SchNet, self).__init__()

        self.num_layer = num_layer

        with self.init_scope():
            self.gn = GraphLinear(512)
            for l in range(self.num_layer):
                self.add_link('sch{}'.format(l), SchNetUpdateBN(512))

            self.interaction1 = L.Linear(512)
            self.interaction2 = L.Linear(512)
            self.interaction3 = L.Linear(512)
            self.interaction4 = L.Linear(512)
            self.interaction5 = L.Linear(512)
            self.interaction6 = L.Linear(4)

    def __call__(self, input_array, dists, pairs_features, targets):

        out = self.predict(input_array, dists, pairs_features)
        loss = F.mean_absolute_error(out, targets)
        reporter.report({'loss': loss}, self)
        return loss

    def predict(self, input_array, dists, pairs_features, **kwargs):

        h = self.gn(input_array)

        for l in range(self.num_layer):
            h = self['sch{}'.format(l)](h, dists)

        h = F.concat((h, input_array), axis=2)

        concat = F.concat([
            h[pairs_features[:, 0].astype(np.int32), pairs_features[:, 1].astype(np.int32), :],
            h[pairs_features[:, 0].astype(np.int32), pairs_features[:, 2].astype(np.int32), :],
            F.expand_dims(dists[pairs_features[:, 0].astype(np.int32),
                                pairs_features[:, 1].astype(np.int32),
                                pairs_features[:, 2].astype(np.int32)], 1),
            pairs_features[:,3:]
        ], axis=1)

        h1 = F.leaky_relu(self.interaction1(concat))
        h2 = F.leaky_relu(self.interaction2(h1))
        h3 = F.leaky_relu(self.interaction3(h2))
        h4 = F.leaky_relu(self.interaction4(h3))
        h5 = F.leaky_relu(self.interaction5(h4))
        out = self.interaction6(h5)

        return out

class SameSizeSampler(OrderSampler):

    def __init__(self, structures_groups, moles, batch_size,
                 random_state=None, use_remainder=False):

        self.structures_groups = structures_groups
        self.moles = moles
        self.batch_size = batch_size
        if random_state is None:
            random_state = np.random.random.__self__
        self._random = random_state
        self.use_remainder = use_remainder

    def __call__(self, current_order, current_position):

        batches = list()

        atom_counts = pd.DataFrame()
        atom_counts['mol_index'] = np.arange(len(self.moles))
        atom_counts['molecular_name'] = self.moles
        atom_counts['num_atom'] = [len(self.structures_groups.get_group(mol))
                                   for mol in self.moles]

        num_atom_counts = atom_counts['num_atom'].value_counts()

        for count, num_mol in num_atom_counts.to_dict().items():
            if self.use_remainder:
                num_batch_for_this = -(-num_mol // self.batch_size)
            else:
                num_batch_for_this = num_mol // self.batch_size

            target_mols = atom_counts.query('num_atom==@count')['mol_index'].values
            random.shuffle(target_mols)

            devider = np.arange(0, len(target_mols), self.batch_size)
            devider = np.append(devider, 99999)

            if self.use_remainder:
                target_mols = np.append(
                    target_mols,
                    np.repeat(target_mols[-1], -len(target_mols) % self.batch_size))

            for b in range(num_batch_for_this):
                batches.append(target_mols[devider[b]:devider[b + 1]])

        random.shuffle(batches)
        batches = np.concatenate(batches).astype(np.int32)

        return batches

def coupling_converter(batch, device):

    list_array = list()
    list_dists = list()
    list_targets = list()
    list_pairs = list()

    with_target = 'fc' in batch[0]['targets'].columns

    for i, d in enumerate(batch):
        list_array.append(d['graphs'].input_array)
        list_dists.append(d['graphs'].dists)
        if with_target:
            list_targets.append(
                d['targets'][['fc', 'sd', 'pso', 'dso']].values.astype(np.float32))

        sample_index = np.full((len(d['targets']), 1), i)
        atom_index = d['targets'][['atom_index_0', 'atom_index_1']].values
        couple_features = d['couples'].values[:,3:]
        list_pairs.append(np.concatenate([sample_index, atom_index, couple_features], axis=1))

    input_array = to_device(device, np.stack(list_array))
    dists = to_device(device, np.stack(list_dists))
    pairs_features = to_device(device, np.concatenate(list_pairs).astype(np.float32))

    array = {'input_array': input_array, 'dists': dists, 'pairs_features': pairs_features}

    if with_target:
        array['targets'] = to_device(device, np.concatenate(list_targets))

    return array

class TypeWiseEvaluator(Evaluator):

    def __init__(self, iterator, target, converter, device, name,
                 is_validate=False, is_submit=False):

        super(TypeWiseEvaluator, self).__init__(
            iterator, target, converter=converter, device=device)

        self.is_validate = is_validate
        self.is_submit = is_submit
        self.name = name

    def calc_score(self, df_truth, pred):

        target_types = list(set(df_truth['type']))

        diff = df_truth['scalar_coupling_constant'] - pred

        scores = 0
        metrics = {}

        for target_type in target_types:

            target_pair = df_truth['type'] == target_type
            score_exp = np.mean(np.abs(diff[target_pair]))
            scores += np.log(score_exp)

            metrics[target_type] = scores

        metrics['ALL_LogMAE'] = scores / len(target_types)

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(metrics, self._targets['main'])

        return observation

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self._targets['main']

        iterator.reset()
        it = iterator

        y_total = []
        t_total = []

        for batch in it:
            in_arrays = self.converter(batch, self.device)
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                y = eval_func.predict(**in_arrays)

            y_data = cuda.to_cpu(y.data)
            y_total.append(y_data)
            t_total.extend([d['targets'] for d in batch])

        df_truth = pd.concat(t_total, axis=0)
        y_pred = np.sum(np.concatenate(y_total), axis=1)

        if self.is_submit:
            submit = pd.DataFrame()
            submit['id'] = df_truth['id']
            submit['scalar_coupling_constant'] = y_pred
            submit.drop_duplicates(subset='id', inplace=True)
            submit.sort_values('id', inplace=True)
            submit.to_csv(f'schnet_{CTYPE}.csv', index=False)

        if self.is_validate:
            return self.calc_score(df_truth, y_pred)

        return {}

def _snapshot_object(trainer, target, filename, savefun):
    fd, tmppath = tempfile.mkstemp()
    try:
        savefun(tmppath, target)
    except Exception:
        os.close(fd)
        os.remove(tmppath)
        raise
    os.close(fd)
    shutil.move(tmppath, filename)

class SaveRestore(chainer.training.extension.Extension):

    """Trainer extension to save a snapshot and restore it at the end of
    training.

    Typical usage is:

    .. code-block:: python

        trainer.extend(
            SaveRestore(),
            trigger=chainer.training.triggers.MinValueTrigger('validation/main/loss'))

    which save will save snapshots and apply (pseudo-) early stopping by
    loading the snapshot with the best validation loss.

    Args:
        filename (str): Name of the file into which the object is serialized.
            It can be a format string, where the trainer object is passed to
            the :meth:`str.format` method. For example,
            ``'snapshot_{.updater.iteration}'`` is converted to
            ``'snapshot_10000'`` at the 10,000th iteration.
            Or you can give name without formatter, which will overwrite the
            saved object on each call, thus only keeping the best model on
            the disk.
            Or you can give None, in which case the object is saved to
            a temporaly path and deleted at the end of the training.
        savefun: Function to save the object. It takes two arguments: the
            output file path and the object to serialize.
        loadfun: Function to load the object. It takes two arguments: the
            file path and the object to deserialize.
    """
    priority = -100

    def __init__(self, filename='snapshot_iter_{.updater.iteration}',
                 savefun=chainer.serializers.npz.save_npz,
                 loadfun=chainer.serializers.npz.load_npz):
        super(SaveRestore, self).__init__()
        self._savefun = savefun
        self._loadfun = loadfun
        self._saved_iteration = None
        self._keep_snapshot = filename is not None
        self._filename = filename or 'saverestore' + str(hash(random.random()))

    def __call__(self, trainer):
        fn = self._filename.format(trainer)
        self._saved_path = os.path.join(trainer.out, fn)
        if not os.path.exists(trainer.out):
            os.makedirs(trainer.out)
        _snapshot_object(trainer, trainer, self._saved_path, self._savefun)
        self._saved_iteration = trainer.updater.iteration
        self._trainer = trainer  # get referencee to trainer

    def finalize(self):
        if self._saved_iteration is not None:
            print('Loading model from %d iteration' % self._saved_iteration)
            self._loadfun(self._saved_path, self._trainer)
        else:
            print('Warning: SaveRestore was never triggered')
        if not self._keep_snapshot and os.path.exists(self._saved_path):
            os.remove(self._saved_path)


def stop_train_mode(trigger):
    @make_extension(trigger=trigger)
    def _stop_train_mode(_):
        chainer.config.train = False
    return _stop_train_mode


def predict_iter(iter, model):
    y_total = []
    t_total = []

    iter.reset()
    for batch in iter:
        in_arrays = coupling_converter(batch, 0)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            y = model.predict(**in_arrays)

        y_data = cuda.to_cpu(y.data)
        y_total.append(y_data)
        t_total.extend([d['targets'] for d in batch])

    df_truth = pd.concat(t_total, axis=0)
    y_pred = np.sum(np.concatenate(y_total), axis=1)

    submit = pd.DataFrame()
    submit['id'] = df_truth['id']
    submit['scalar_coupling_constant'] = y_pred
    submit.drop_duplicates(subset='id', inplace=True)
    submit.sort_values('id', inplace=True)
    return submit


#%%
# def main():
#%% Load datasets
train, valid, test, train_moles, valid_moles, test_moles = load_dataset(CTYPE)

train_gp = train.groupby('molecule_name')
valid_gp = valid.groupby('molecule_name')
test_gp = test.groupby('molecule_name')

#%%
structures = pd.read_csv(DATA_PATH/'structures.csv')

giba_features = pd.read_csv(DATA_PATH/'unified-features'/'giba_features.csv', index_col=0)
structures = pd.merge(structures,giba_features.drop(['atom_name','x','y','z'],axis=1),on=['molecule_name','atom_index'])
norm_col = [col for col in structures.columns if col not in ['molecule_name','atom_index','atom','x','y','z']]
structures[norm_col] = (structures[norm_col]-structures[norm_col].mean())/structures[norm_col].std()
structures = structures.fillna(0)
structures_groups = structures.groupby('molecule_name')

#%%
if CTYPE != 'all':
    train_couple = pd.read_csv(DATA_PATH/'typewise-dataset'/'kuma_dataset'/'kuma_dataset'/'train'/'{}_full.csv'.format(CTYPE),index_col=0)
else:
    train_couple = pd.read_csv(DATA_PATH/'typewise-dataset'/'kuma_dataset'/'kuma_dataset'/'train_all.csv',index_col=0)
train_couple = reduce_mem_usage(train_couple)
train_couple = train_couple.drop(['id','scalar_coupling_constant','type'],axis=1)
if CTYPE != 'all':
    test_couple = pd.read_csv(DATA_PATH/'typewise-dataset'/'kuma_dataset'/'kuma_dataset'/'test'/'{}_full.csv'.format(CTYPE),index_col=0)
else:
    test_couple = pd.read_csv(DATA_PATH/'typewise-dataset'/'kuma_dataset'/'kuma_dataset'/'test_all.csv',index_col=0)
test_couple = reduce_mem_usage(test_couple)
test_couple = test_couple.drop(['id','type'],axis=1)

couples = pd.concat([train_couple, test_couple])

del train_couple, test_couple

couples_norm_col = [col for col in couples.columns if col not in ['atom_index_0','atom_index_1','molecule_name','type']]

for col in couples_norm_col:
    if couples[col].dtype==np.dtype('O'):
        couples = pd.get_dummies(couples,columns=[col])
    else:
        couples[col] = (couples[col]-couples[col].mean())/couples[col].std()

couples = couples.fillna(0)
couples = couples.replace(np.inf, 0)
couples = couples.replace(-np.inf, 0)
couples_groups = couples.groupby('molecule_name')

#%% Make graphs
feature_col = [col for col in structures.columns if col not in ['molecule_name','atom_index','atom']]

list_atoms = list(set(structures['atom']))
print('list of atoms')
print(list_atoms)

train_graphs = list()
train_targets = list()
train_couples = list()
print('preprocess training molecules ...')
for mole in tqdm(train_moles):
    train_graphs.append(Graph(structures_groups.get_group(mole), list_atoms, feature_col))
    train_targets.append(train_gp.get_group(mole))
    train_couples.append(couples_groups.get_group(mole))

valid_graphs = list()
valid_targets = list()
valid_couples = list()
print('preprocess validation molecules ...')
for mole in tqdm(valid_moles):
    valid_graphs.append(Graph(structures_groups.get_group(mole), list_atoms, feature_col))
    valid_targets.append(valid_gp.get_group(mole))
    valid_couples.append(couples_groups.get_group(mole))

test_graphs = list()
test_targets = list()
test_couples = list()
print('preprocess test molecules ...')
for mole in tqdm(test_moles):
    test_graphs.append(Graph(structures_groups.get_group(mole), list_atoms, feature_col))
    test_targets.append(test_gp.get_group(mole))
    test_couples.append(couples_groups.get_group(mole))


#%% Make datasets
train_dataset = DictDataset(graphs=train_graphs, targets=train_targets, couples=train_couples)
valid_dataset = DictDataset(graphs=valid_graphs, targets=valid_targets, couples=valid_couples)
test_dataset = DictDataset(graphs=test_graphs, targets=test_targets, couples=test_couples)


#%% Build Model
model = SchNet(num_layer=NUM_LAYER)
# model.to_gpu(device=0)


#%% Sampler
train_sampler = SameSizeSampler(structures_groups, train_moles, BATCH_SIZE)
valid_sampler = SameSizeSampler(structures_groups, valid_moles, BATCH_SIZE,
                                use_remainder=True)
test_sampler = SameSizeSampler(structures_groups, test_moles, BATCH_SIZE,
                               use_remainder=True)


#%% Iterator, Optimizer
train_iter = chainer.iterators.SerialIterator(
    train_dataset, BATCH_SIZE, repeat=False, order_sampler=train_sampler)

valid_iter = chainer.iterators.SerialIterator(
    valid_dataset, BATCH_SIZE, repeat=False, order_sampler=valid_sampler)

test_iter = chainer.iterators.SerialIterator(
    test_dataset, BATCH_SIZE, repeat=False, order_sampler=test_sampler)


#%% Predict
chainer.config.train = False
# snapshot_path = f'results/chainer/best_epoch_{CTYPE}'
snapshot_path = f'../results/chainer/best_epoch_{CTYPE}'
chainer.serializers.npz.load_npz(snapshot_path, model, 'updater/model:main/')
train_oof = predict_iter(train_iter, model)
valid_oof = predict_iter(valid_iter, model)
test_oof = predict_iter(test_iter, model)

train_oof.to_csv(f'results/schnet_{CTYPE}_train.csv', index=False)
valid_oof.to_csv(f'results/schnet_{CTYPE}_valid.csv', index=False)
test_oof.to_csv(f'results/schnet_{CTYPE}_test.csv', index=False)

main()
if __name__ == "__main__":
    #%%
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='~/kaggle-data/champs/',
                        help="root path to data")
    parser.add_argument("--num_layer", type=int, default=5,
                        help="depth of schnet")
    parser.add_argument("--epoch", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="number of batch size")
    parser.add_argument("--types", type=str, default='1JHN',
                        help="which type to train (default: 1JHN)")
    parser.add_argument("--multi_gpu", action='store_true',
                        help="whether to use multiple gpus")
    parser.add_argument("--early_stopping", type=int, default=10,
                        help="early stopping rounds")

    opt = parser.parse_args()
    print(opt)
    EPOCH = opt.epoch
    DATA_PATH = Path(opt.data)
    RESULT_PATH = 'results/chainer'
    Path(RESULT_PATH).mkdir(exist_ok=True)
    CTYPE = opt.types
    BATCH_SIZE = opt.batch_size
    EARLY_STOPPING_ROUNDS = opt.early_stopping
    NUM_LAYER = opt.num_layer

    for CTYPE in ['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN', 'all']:
        print('starting', CTYPE)
        main()
