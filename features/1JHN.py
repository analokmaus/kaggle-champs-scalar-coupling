import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
from pathlib import Path
import multiprocessing


DATA_PATH = Path('../data')
(DATA_PATH/'neighborhood'/'1JHN').mkdir()
NH_PATH = DATA_PATH / 'neighborhood'
FEATURE_COLS = ['len1', 'ring', 'aroma',
                'nh_count', 'combination_atom',#'Cs', 'Hs', 'Ns', 'Os', 'Fs',
                'ang2_mean', 'ang2_max', 'ang2_var', 'ang2_atom',
                'len2_mean', 'len2_min', 'len2_var', 'len2_atom',
                'nh_ring', 'nh_aroma',
                'near_count', 'near_atom', 'near_dist']


atom_name = {
    'C': 0,
    'H': 1,
    'N': 2,
    'O': 3,
    'F': 4
}


mol_structure = pickle.load(open(DATA_PATH / 'mol_features.pkl', 'rb'))
train = pd.read_csv(DATA_PATH / 'train.csv')
test = pd.read_csv(DATA_PATH / 'test.csv')
train = train[train['type'] == '1JHN']
test = test[test['type'] == '1JHN']
print(train.shape, test.shape)
train.head()


'''
GENERATE FEATURES
'''
def cosine(vec1, vec2):
    return np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)

def get_features(molecule_name, a1, a2):

    '''
    INPUT
        a1: idx of H
        a2: idx of C
    OUTPUT

    '''
    nh1_features = []
    nh2_features = []
    nnh_features = []

    # features
    atoms = mol_structure[molecule_name]['atom_type']
    coords = mol_structure[molecule_name]['coord']
    rings = mol_structure[molecule_name]['ring']
    bonds = mol_structure[molecule_name]['bond']
    bonds_ = mol_structure[molecule_name]['bond'][:, [1, 0, 2, 3]] # inversed bond
    bonds = np.concatenate([bonds, bonds_])
    all_vecs = coords - coords[a1]
    all_dists = np.linalg.norm(all_vecs, axis=1) # distances from a1

    # neighborhood 1
    nh1_idx = (bonds[:, 0] == a2) & (bonds[:, 1] == a1)
    nh1_vec = coords[a1] - coords[a2]
    if nh1_idx.sum() == 0: #dsgdb9nsd_059827
        nh1_features += [all_dists[a2]]
    else:
        nh1_features += [*bonds[nh1_idx, 3]] # bond length
    nh1_features += [*rings[a2]]# atom isring, atom isaroma

    # neightborhood 2
    nh2_idx = (bonds[:, 0] == a2) ^ nh1_idx
    nh2_features += [nh2_idx.sum()] # nh2 atom count
    nh2_atms = np.zeros(5, dtype=np.int8) # atom count 2
    nh2_angs = []
    nh2_lens = []
    nh2_rngs = []
    nh2_atom_idx = []
    for idx in np.where(nh2_idx)[0]: # iterate indice
        nh2 = int(bonds[idx, 1]) # atom index
        nh2_vec = coords[nh2] - coords[a2]
        nh2_len = np.linalg.norm(nh2_vec)
        nh2_cos = cosine(nh1_vec, nh2_vec)
        nh2_atom_idx.append(nh2)
        nh2_lens.append(nh2_len)
        nh2_angs.append(nh2_cos)
        nh2_rngs.append(rings[nh2])
        nh2_atms[atoms[nh2]] += 1

    # atom counter
    nh2_atms = int(''.join((nh2_atms+1).astype(str)))
    nh2_features += [nh2_atms]

    # mean, max, variance of angles, atom with sharpest angle
    nh2_features += [np.mean(nh2_angs), np.max(nh2_angs),
                     np.var(nh2_angs), atoms[nh2_atom_idx[np.argmax(nh2_angs)]]]
    # mean, min, variance of lengths, closest atom
    nh2_features += [np.mean(nh2_lens), np.min(nh2_lens),
                     np.var(nh2_lens), atoms[nh2_atom_idx[np.argmin(nh2_lens)]]]
    # count of atoms in ring, aromatic
    nh2_features += list(np.sum(nh2_rngs, axis=0))

    # non neighborhood
    near_count = (all_dists <= 3.0).sum()
    nnh_features += [near_count]
    for idx in np.argsort(all_dists)[:near_count]: # looking for hydrogen bond
        a = atoms[idx]
        dist = all_dists[idx]
        if a >= 2: # F, O, N
            nnh_features += [a, dist]
            break
    else:
        nnh_features += [0, 3.0]

    return (*nh1_features, *nh2_features, *nnh_features)

get_features('dsgdb9nsd_000002',1,0)

'''
EXPORT
'''
res = []
t_res = []
# for mname, a0, a1 in tqdm(train[['molecule_name', 'atom_index_0', 'atom_index_1']].values):
#     res.append(get_features(mname, a0, a1))

n_cpu = multiprocessing.cpu_count()

def wrapper(args):
    return get_features(*args)

with multiprocessing.Pool(n_cpu) as p:
    n = train.shape[0]
    with tqdm(total=n) as pbar:
        for line in p.imap(wrapper,
                           train[['molecule_name', 'atom_index_0', 'atom_index_1']].values):
            res.append(line)
            pbar.update()

with multiprocessing.Pool(n_cpu) as p:
    n = test.shape[0]
    with tqdm(total=n) as pbar:
        for line in p.imap(wrapper,
                           test[['molecule_name', 'atom_index_0', 'atom_index_1']].values):
            t_res.append(line)
            pbar.update()


res = pd.DataFrame(res)
res.index = train.index
res.columns = FEATURE_COLS
res.head()
res.to_csv('../data/neighborhood/1JHN/unified_features.csv')

t_res = pd.DataFrame(t_res)
t_res.index = test.index
t_res.columns = FEATURE_COLS
t_res.shape
t_res.to_csv('../data/neighborhood/1JHN/unified_features_test.csv')
