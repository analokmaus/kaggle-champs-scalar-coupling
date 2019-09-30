import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
import gc
import multiprocessing
tqdm.pandas()


DATA_PATH = Path('../data')
OUTPUT_PATH = DATA_PATH / 'giba_dataset'


# %%
# Loading Data
train = pd.read_csv(DATA_PATH / 'train.csv', index_col=0)
test = pd.read_csv(DATA_PATH / 'test.csv', index_col=0)
mol_structures = pickle.load(open(DATA_PATH / 'mol_features.pkl', 'rb'))
train.head()
test.head()
train.shape, test.shape


def cosine(vec1, vec2):
    return np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)


data = []

for mname, props in tqdm(mol_structures.items()):
    atoms = props['atom_type']
    coords = props['coord']
    bonds = props['bond']
    bonds_ = bonds[:, [1, 0, 2, 3]]  # inversed bond
    bonds = np.concatenate([bonds, bonds_])

    for i, a1 in enumerate(atoms):
        d = {x: [] for x in range(5)}
        misc = [mname, i, a1, *coords[i]]
        cou = []
        vdw = []
        ykw = []
        bnd = []

        for ii, a2 in enumerate(atoms):  # get distances to atoms by type
            if i == ii:
                continue
            d[a2].append(np.linalg.norm(coords[ii] - coords[i]))

        for a in range(5):
            cou.append(sum([1 / r for r in d[a]]))
            vdw.append(sum([1 / r**3 for r in d[a]]))
            ykw.append(sum([np.exp(-r) / r for r in d[a]]))

        bond_filter = bonds[:, 0] == i
        bnd.append(bond_filter.sum())  # bond counts
        bond_atms = []
        bond_coss = []
        bond_lens = []
        for a2 in np.where(bond_filter)[0]:
            ii = int(bonds[a2, 1])  # atom index
            atm = str(atoms[ii] + 1)
            vec = coords[ii] - coords[i]
            cos = cosine([1, 0, 0], vec)
            len = np.linalg.norm(vec)
            bond_atms.append(atm)
            bond_coss.append(cos)
            bond_lens.append(len)
        bond_atms = ''.join(bond_atms)
        bnd += [bond_atms, np.mean(bond_coss), np.var(bond_coss),
                np.mean(bond_lens), np.var(bond_lens)]
        data.append([*misc, *cou, *vdw, *ykw, *bnd])
