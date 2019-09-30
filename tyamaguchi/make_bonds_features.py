# %%
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
%matplotlib inline

KAGGLE_DIR = 'data'  # 'champs-scalar-coupling'

# Atom level properties
MULLIKEN_CHARGES_CSV = os.path.join(KAGGLE_DIR, 'mulliken_charges.csv')
SCALAR_COUPLING_CONTRIBUTIONS_CSV = os.path.join(
    KAGGLE_DIR, 'scalar_coupling_contributions.csv')
MAGNETIC_SHIELDING_TENSORS_CSV = os.path.join(
    KAGGLE_DIR, 'magnetic_shielding_tensors.csv')
STRUCTURES_CSV = os.path.join(KAGGLE_DIR, 'structures.csv')
STRUCTURES_TEST_CSV = os.path.join(KAGGLE_DIR, 'structures_test.csv')

# Molecule level properties
POTENTIAL_ENERGY_CSV = os.path.join(KAGGLE_DIR, 'potential_energy.csv')
DIPOLE_MOMENTS_CSV = os.path.join(KAGGLE_DIR, 'dipole_moments.csv')

# Atom-Atom interactions
TRAIN_CSV = os.path.join(KAGGLE_DIR, 'train.csv')
TEST_CSV = os.path.join(KAGGLE_DIR, 'test.csv')


# %%
atomic_radius = {'H': 0.38, 'C': 0.77, 'N': 0.85,
                 'O': 0.83, 'F': 0.71}  # Without fudge factor

# 1J,2Jでは0.05, 3Jでは0.1
fudge_factor = 0.05
atomic_radius = {k: v + fudge_factor for k, v in atomic_radius.items()}
print(atomic_radius)

electronegativity = {'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98}

structures = pd.read_csv(STRUCTURES_CSV, dtype={'atom_index': np.int8})
atoms = structures['atom'].values
atoms_en = [electronegativity[x] for x in tqdm(atoms)]
atoms_rad = [atomic_radius[x] for x in tqdm(atoms)]
structures['EN'] = atoms_en
structures['rad'] = atoms_rad

display(structures.head())

# %%
i_atom = structures['atom_index'].values
p = structures[['x', 'y', 'z']].values
p_compare = p
m = structures['molecule_name'].values
m_compare = m
r = structures['rad'].values
r_compare = r

source_row = np.arange(len(structures))
max_atoms = 29

bonds = np.zeros((len(structures) + 1, max_atoms + 1), dtype=np.int8)
bond_dists = np.zeros((len(structures) + 1, max_atoms + 1), dtype=np.float32)

print('Calculating bonds')

for i in tqdm(range(max_atoms - 1)):
    p_compare = np.roll(p_compare, -1, axis=0)
    m_compare = np.roll(m_compare, -1, axis=0)
    r_compare = np.roll(r_compare, -1, axis=0)

    # Are we still comparing atoms in the same molecule?
    mask = np.where(m == m_compare, 1, 0)
    dists = np.linalg.norm(p - p_compare, axis=1) * mask
    r_bond = r + r_compare

    bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

    source_row = source_row
    # Note: Will be out of bounds of bonds array for some values of i
    target_row = source_row + i + 1
    target_row = np.where(np.logical_or(target_row > len(structures), mask == 0), len(
        structures), target_row)  # If invalid target, write to dummy row

    source_atom = i_atom
    # Note: Will be out of bounds of bonds array for some values of i
    target_atom = i_atom + i + 1
    target_atom = np.where(np.logical_or(target_atom > max_atoms, mask == 0),
                           max_atoms, target_atom)  # If invalid target, write to dummy col

    bonds[(source_row, target_atom)] = bond
    bonds[(target_row, source_atom)] = bond
    bond_dists[(source_row, target_atom)] = dists
    bond_dists[(target_row, source_atom)] = dists

bonds = np.delete(bonds, axis=0, obj=-1)  # Delete dummy row
bonds = np.delete(bonds, axis=1, obj=-1)  # Delete dummy col
bond_dists = np.delete(bond_dists, axis=0, obj=-1)  # Delete dummy row
bond_dists = np.delete(bond_dists, axis=1, obj=-1)  # Delete dummy col

print('Counting and condensing bonds')

bonds_numeric = [[i for i, x in enumerate(row) if x] for row in tqdm(bonds)]
bond_lengths = [[dist for i, dist in enumerate(
    row) if i in bonds_numeric[j]] for j, row in enumerate(tqdm(bond_dists))]
n_bonds = [len(x) for x in bonds_numeric]

#bond_data = {'bond_' + str(i):col for i, col in enumerate(np.transpose(bonds))}
#bond_data.update({'bonds_numeric':bonds_numeric, 'n_bonds':n_bonds})

bond_data = {'bonds': bonds_numeric,
             'n_bonds': n_bonds, 'bond_lengths': bond_lengths}
bond_df = pd.DataFrame(bond_data)
structures = structures.join(bond_df)
display(structures.head(20))
with open('data/structures_simple.pkl', 'wb') as f:
    pickle.dump(structures, f)
with open('data/structures_simple.pkl', 'rb') as f:
    structures = pickle.load(f)


# %%
# 1JHC
train = pd.read_csv(TRAIN_CSV, index_col=0)
train = train[train['type'] == '1JHC']
train = train.merge(structures, left_on=[
                    'molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])
train = train.drop(['EN', 'rad', 'atom_index'], axis=1)
train2 = train[train['n_bonds'] == 2]
train3 = train[train['n_bonds'] == 3]
train4 = train[train['n_bonds'] == 4]
train2['scalar_coupling_constant'].mean(), train3['scalar_coupling_constant'].mean(
), train4['scalar_coupling_constant'].mean()
# (196.25652434942228, 124.60926891959014, 91.02942034187265)
structures = pd.read_csv(STRUCTURES_CSV, dtype={'atom_index': np.int8})
structures_d = {}
for thisIter in tqdm(structures.iterrows(), total=structures.shape[0]):
    molecule = thisIter[1]['molecule_name']
    atom = thisIter[1]['atom']
    atom_index = thisIter[1]['atom_index']
    if molecule not in structures_d:
        structures_d[molecule] = {}
    structures_d[molecule][atom_index] = [
        atom, thisIter[1]['x'], thisIter[1]['y'], thisIter[1]['z']]

structures_d['dsgdb9nsd_028960']


def add_neighborhood_features(df, n_bonds):
    data = []
    for thisIter in tqdm(df.iterrows(), total=df.shape[0]):
        h = thisIter[1]['atom_index_0']
        dists = [[thisIter[1]['bonds'][i], thisIter[1]['bond_lengths'][i]]
                 for i in range(n_bonds) if thisIter[1]['bonds'][i] != h]
        dists.sort(key=lambda x: x[1])
        x1, y1, z1 = thisIter[1]['x'], thisIter[1]['y'], thisIter[1]['z']
        d = structures_d[thisIter[1]['molecule_name']]
        x0, y0, z0 = d[thisIter[1]['atom_index_0']][1:]
        thisData = [((x0 - x1)**2 + (y0 - y1)**2 + (z0 - z1)**2)**0.5]
        atom = []
        cos = []
        for index, dist in dists:
            atom.append(d[index][0])
            x, y, z = d[index][1:]
            cos.append(((x - x1) * (x0 - x1) + (y - y1) * (y0 - y1)
                        + (z - z1) * (z0 - z1)) / (dist * thisData[0]))
        dists = [dist for index, dist in dists]
        thisData = thisData + atom + dists + cos
        data.append(thisData)
    cols = ['dist']
    res = df.copy()
    for i in range(2, n_bonds + 1):
        cols.append('atom_{}'.format(i))
    for i in range(2, n_bonds + 1):
        cols.append('dist_{}'.format(i))
    for i in range(2, n_bonds + 1):
        cols.append('cos_{}'.format(i))
    for col in cols:
        res[col] = 0
    res[cols] = data
    res = res.drop(['atom', 'bonds', 'n_bonds',
                    'bond_lengths', 'x', 'y', 'z'], axis=1)
    return res


train4_nh = add_neighborhood_features(train4, 4)
train3_nh = add_neighborhood_features(train3, 3)
train2_nh = add_neighborhood_features(train2, 2)
train4_nh.to_csv(os.path.join(
    KAGGLE_DIR, 'neighborhood/1JHC/train_1JHC_4_bond.csv'), index=False)
train3_nh.to_csv(os.path.join(
    KAGGLE_DIR, 'neighborhood/1JHC/train_1JHC_3_bond.csv'), index=False)
train2_nh.to_csv(os.path.join(
    KAGGLE_DIR, 'neighborhood/1JHC/train_1JHC_2_bond.csv'), index=False)

# %%
# 1JHN
train = pd.read_csv(TRAIN_CSV, index_col=0)
train = train[train['type'] == '1JHN']
train = train.merge(structures, left_on=[
                    'molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])
train = train.drop(['EN', 'rad', 'atom_index'], axis=1)
train2 = train[train['n_bonds'] == 2]
train3 = train[train['n_bonds'] == 3]
train4 = train[train['n_bonds'] == 4]
train2['scalar_coupling_constant'].mean(), train3['scalar_coupling_constant'].mean(
), train4['scalar_coupling_constant'].mean()
# (34.64526708410899, 49.54152316814112, 44.34582113207547)
train4
train4_nh = add_neighborhood_features(train4, 4)
train3_nh = add_neighborhood_features(train3, 3)
train2_nh = add_neighborhood_features(train2, 2)

train4_nh.to_csv(os.path.join(
    KAGGLE_DIR, 'neighborhood/1JHN/train_1JHN_4_bond.csv'), index=False)
train3_nh.to_csv(os.path.join(
    KAGGLE_DIR, 'neighborhood/1JHN/train_1JHN_3_bond.csv'), index=False)
train2_nh.to_csv(os.path.join(
    KAGGLE_DIR, 'neighborhood/1JHN/train_1JHN_2_bond.csv'), index=False)

# %%
# test
atomic_radius = {'H': 0.38, 'C': 0.77, 'N': 0.75,
                 'O': 0.73, 'F': 0.71}  # Without fudge factor

fudge_factor = 0.05
atomic_radius = {k: v + fudge_factor for k, v in atomic_radius.items()}
print(atomic_radius)

electronegativity = {'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98}

structures = pd.read_csv(STRUCTURES_TEST_CSV, dtype={'atom_index': np.int8})

atoms = structures['atom'].values
atoms_en = [electronegativity[x] for x in tqdm(atoms)]
atoms_rad = [atomic_radius[x] for x in tqdm(atoms)]

structures['EN'] = atoms_en
structures['rad'] = atoms_rad

display(structures.head())

# %%
i_atom = structures['atom_index'].values
p = structures[['x', 'y', 'z']].values
p_compare = p
m = structures['molecule_name'].values
m_compare = m
r = structures['rad'].values
r_compare = r

source_row = np.arange(len(structures))
max_atoms = 29

bonds = np.zeros((len(structures) + 1, max_atoms + 1), dtype=np.int8)
bond_dists = np.zeros((len(structures) + 1, max_atoms + 1), dtype=np.float32)

print('Calculating bonds')

for i in tqdm(range(max_atoms - 1)):
    p_compare = np.roll(p_compare, -1, axis=0)
    m_compare = np.roll(m_compare, -1, axis=0)
    r_compare = np.roll(r_compare, -1, axis=0)

    # Are we still comparing atoms in the same molecule?
    mask = np.where(m == m_compare, 1, 0)
    dists = np.linalg.norm(p - p_compare, axis=1) * mask
    r_bond = r + r_compare

    bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

    source_row = source_row
    # Note: Will be out of bounds of bonds array for some values of i
    target_row = source_row + i + 1
    target_row = np.where(np.logical_or(target_row > len(structures), mask == 0), len(
        structures), target_row)  # If invalid target, write to dummy row

    source_atom = i_atom
    # Note: Will be out of bounds of bonds array for some values of i
    target_atom = i_atom + i + 1
    target_atom = np.where(np.logical_or(target_atom > max_atoms, mask == 0),
                           max_atoms, target_atom)  # If invalid target, write to dummy col

    bonds[(source_row, target_atom)] = bond
    bonds[(target_row, source_atom)] = bond
    bond_dists[(source_row, target_atom)] = dists
    bond_dists[(target_row, source_atom)] = dists

bonds = np.delete(bonds, axis=0, obj=-1)  # Delete dummy row
bonds = np.delete(bonds, axis=1, obj=-1)  # Delete dummy col
bond_dists = np.delete(bond_dists, axis=0, obj=-1)  # Delete dummy row
bond_dists = np.delete(bond_dists, axis=1, obj=-1)  # Delete dummy col

print('Counting and condensing bonds')

bonds_numeric = [[i for i, x in enumerate(row) if x] for row in tqdm(bonds)]
bond_lengths = [[dist for i, dist in enumerate(
    row) if i in bonds_numeric[j]] for j, row in enumerate(tqdm(bond_dists))]
n_bonds = [len(x) for x in bonds_numeric]

#bond_data = {'bond_' + str(i):col for i, col in enumerate(np.transpose(bonds))}
#bond_data.update({'bonds_numeric':bonds_numeric, 'n_bonds':n_bonds})

bond_data = {'bonds': bonds_numeric,
             'n_bonds': n_bonds, 'bond_lengths': bond_lengths}
bond_df = pd.DataFrame(bond_data)
structures = structures.join(bond_df)
display(structures.head(20))

structures
# %%
# 1JHC
test = pd.read_csv(TEST_CSV, index_col=0)
test = test[test['type'] == '1JHC']
test = test.merge(structures, left_on=['molecule_name', 'atom_index_1'], right_on=[
                  'molecule_name', 'atom_index'])
test = test.drop(['EN', 'rad', 'atom_index'], axis=1)
test2 = test[test['n_bonds'] == 2]
test3 = test[test['n_bonds'] == 3]
test4 = test[test['n_bonds'] == 4]
structures = pd.read_csv(STRUCTURES_TEST_CSV, dtype={'atom_index': np.int8})
structures_d = {}
for thisIter in tqdm(structures.iterrows()):
    molecule = thisIter[1]['molecule_name']
    atom = thisIter[1]['atom']
    atom_index = thisIter[1]['atom_index']
    if molecule not in structures_d:
        structures_d[molecule] = {}
    structures_d[molecule][atom_index] = [
        atom, thisIter[1]['x'], thisIter[1]['y'], thisIter[1]['z']]


def add_neighborhood_features(df, n_bonds):
    data = []
    for thisIter in tqdm(df.iterrows()):
        h = thisIter[1]['atom_index_0']
        dists = [[thisIter[1]['bonds'][i], thisIter[1]['bond_lengths'][i]]
                 for i in range(n_bonds) if thisIter[1]['bonds'][i] != h]
        dists.sort(key=lambda x: x[1])
        x1, y1, z1 = thisIter[1]['x'], thisIter[1]['y'], thisIter[1]['z']
        d = structures_d[thisIter[1]['molecule_name']]
        x0, y0, z0 = d[thisIter[1]['atom_index_0']][1:]
        thisData = [((x0 - x1)**2 + (y0 - y1)**2 + (z0 - z1)**2)**0.5]
        atom = []
        cos = []
        for index, dist in dists:
            atom.append(d[index][0])
            x, y, z = d[index][1:]
            cos.append(((x - x1) * (x0 - x1) + (y - y1) * (y0 - y1)
                        + (z - z1) * (z0 - z1)) / (dist * thisData[0]))
        dists = [dist for index, dist in dists]
        thisData = thisData + atom + dists + cos
        data.append(thisData)
    cols = ['dist']
    res = df.copy()
    for i in range(2, n_bonds + 1):
        cols.append('atom_{}'.format(i))
    for i in range(2, n_bonds + 1):
        cols.append('dist_{}'.format(i))
    for i in range(2, n_bonds + 1):
        cols.append('cos_{}'.format(i))
    for col in cols:
        res[col] = 0
    res[cols] = data
    res = res.drop(['atom', 'bonds', 'n_bonds',
                    'bond_lengths', 'x', 'y', 'z'], axis=1)
    return res


test4
test4_nh = add_neighborhood_features(test4, 4)
test3_nh = add_neighborhood_features(test3, 3)
test2_nh = add_neighborhood_features(test2, 2)
test4_nh.to_csv(os.path.join(
    KAGGLE_DIR, 'neighborhood/1JHC/test_1JHC_4_bond.csv'), index=False)
test3_nh.to_csv(os.path.join(
    KAGGLE_DIR, 'neighborhood/1JHC/test_1JHC_3_bond.csv'), index=False)
test2_nh.to_csv(os.path.join(
    KAGGLE_DIR, 'neighborhood/1JHC/test_1JHC_2_bond.csv'), index=False)

# %%
# 1JHN
test = pd.read_csv(TEST_CSV, index_col=0)
test = test[test['type'] == '1JHN']
test = test.merge(structures, left_on=['molecule_name', 'atom_index_1'], right_on=[
                  'molecule_name', 'atom_index'])
test = test.drop(['EN', 'rad', 'atom_index'], axis=1)
test2 = test[test['n_bonds'] == 2]
test3 = test[test['n_bonds'] == 3]
test4 = test[test['n_bonds'] == 4]

test4_nh = add_neighborhood_features(test4, 4)
test3_nh = add_neighborhood_features(test3, 3)
test2_nh = add_neighborhood_features(test2, 2)
test4_nh.to_csv(os.path.join(
    KAGGLE_DIR, 'neighborhood/1JHN/test_1JHN_4_bond.csv'), index=False)
test3_nh.to_csv(os.path.join(
    KAGGLE_DIR, 'neighborhood/1JHN/test_1JHN_3_bond.csv'), index=False)
test2_nh.to_csv(os.path.join(
    KAGGLE_DIR, 'neighborhood/1JHN/test_1JHN_2_bond.csv'), index=False)
train_nh

# %%
# 3JHC
train = pd.read_csv(TRAIN_CSV, index_col=0)
train = train[train['type'] == '3JHH']

train = train.merge(structures, left_on=[
                    'molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])
train = train.drop(['EN', 'rad', 'atom', 'atom_index'], axis=1)
train = train.merge(structures, left_on=['molecule_name', 'atom_index_1'], right_on=[
                    'molecule_name', 'atom_index'], suffixes=['_0', '_1'])
train = train.drop(['EN', 'rad', 'atom', 'atom_index'], axis=1)
train['atom_index_A'] = train['bonds_0'].apply(lambda x: x[0])
train = train.drop(['bonds_0', 'n_bonds_0', 'bond_lengths_0'], axis=1)
df = train.merge(structures, left_on=['molecule_name', 'atom_index_A'], right_on=[
                 'molecule_name', 'atom_index'])
df = df.drop(['EN', 'rad', 'atom_index'], axis=1)
df = df.rename(columns={'atom': 'atom_A', 'x': 'x_A', 'y': 'y_A', 'z': 'z_A',
                        'bonds': 'bonds_A', 'n_bonds': 'n_bonds_A', 'bond_lengths': 'bond_lengths_A'})

s = []
s1 = df['bonds_1'].apply(lambda x: set(x))
sA = df['bonds_A'].apply(lambda x: set(x))
for i in tqdm(range(len(df))):
    s.append(s1[i] & sA[i])
s = [list(x) for x in s]
df['atom_B_indices'] = s
df['atom_B_num'] = df['atom_B_indices'].apply(lambda x: len(x))
df['atom_B_num'].value_counts()

df = df.drop(['bonds_1', 'bond_lengths_1', 'bond_lengths_A'], axis=1)
atom_B_index = []
for thisIter in tqdm(df.iterrows()):
    if thisIter[1]['atom_B_num'] != 0:
        atom_B_index.append(thisIter[1]['atom_B_indices'][0])
    else:
        atom_B_index.append(thisIter[1]['bonds_A'][0])
df['atom_index_B'] = atom_B_index
df = df.drop(['bonds_A', 'atom_B_indices'], axis=1)
df = df.merge(structures, left_on=['molecule_name', 'atom_index_B'], right_on=[
              'molecule_name', 'atom_index'])
df = df.drop(['EN', 'rad', 'bonds', 'bond_lengths', 'atom_index'], axis=1)
df = df.rename(columns={'atom': 'atom_B', 'x': 'x_B',
                        'y': 'y_B', 'z': 'z_B', 'n_bonds': 'n_bonds_B'})


def add_angle_dist_features(df):
    data = []
    for thisIter in tqdm(df.iterrows()):
        thisIter = thisIter[1]
        r0 = np.array([thisIter['x_0'], thisIter['y_0'], thisIter['z_0']])
        r1 = np.array([thisIter['x_1'], thisIter['y_1'], thisIter['z_1']])
        rA = np.array([thisIter['x_A'], thisIter['y_A'], thisIter['z_A']])
        rB = np.array([thisIter['x_B'], thisIter['y_B'], thisIter['z_B']])
        r0AB = np.cross(r0 - rA, rB - rA)
        r0AB /= np.linalg.norm(r0AB)
        r1AB = np.cross(r1 - rB, rA - rB)
        r1AB /= np.linalg.norm(r1AB)
        d0A = np.linalg.norm(rA - r0)
        d1B = np.linalg.norm(rB - r1)
        dAB = np.linalg.norm(rB - rA)
        d0B = np.linalg.norm(rB - r0)
        d1A = np.linalg.norm(rA - r1)
        d01 = np.linalg.norm(r1 - r0)
        cosA = np.dot(r0 - rA, rB - rA) / (d0A * dAB)
        cosB = np.dot(r1 - rB, rA - rB) / (dAB * d1B)
        cosAB = np.dot(r0 - rA, r1 - rB) / (d0A * d1B)
        cosD = np.dot(r0AB, r1AB)
        data.append([d0A, d1B, dAB, d0B, d1A, d01, cosA, cosB, cosAB, cosD])

    cols = ['d0A', 'd1B', 'dAB', 'd0B', 'd1A', 'd01',
            'cosA', 'cosB', 'cos0A_1B', 'cos_Dihedral']
    res = df.copy()
    for col in cols:
        res[col] = 0
    res[cols] = data
    res = res.drop(['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1',
                    'x_A', 'y_A', 'z_A', 'x_B', 'y_B', 'z_B'], axis=1)
    return res


df = add_angle_dist_features(df)

df = pd.read_csv(os.path.join(
    KAGGLE_DIR, 'neighborhood/3JHH/train_3JHH_v1.csv'))
# df.to_csv(os.path.join(KAGGLE_DIR,'neighborhood/3JHH/train_3JHH_v1.csv'),index=False)

pd.get_dummies(df, columns=['atom_A', 'atom_B'])
# %%
# test
# 3JHC
test = pd.read_csv(TEST_CSV, index_col=0)
test = test[test['type'] == '3JHC']

test = test.merge(structures, left_on=['molecule_name', 'atom_index_0'], right_on=[
                  'molecule_name', 'atom_index'])
test = test.drop(['EN', 'rad', 'atom', 'atom_index'], axis=1)
test = test.merge(structures, left_on=['molecule_name', 'atom_index_1'], right_on=[
                  'molecule_name', 'atom_index'], suffixes=['_0', '_1'])
test = test.drop(['EN', 'rad', 'atom', 'atom_index'], axis=1)
test['atom_index_A'] = test['bonds_0'].apply(lambda x: x[0])
test = test.drop(['bonds_0', 'n_bonds_0', 'bond_lengths_0'], axis=1)
df = test.merge(structures, left_on=['molecule_name', 'atom_index_A'], right_on=[
                'molecule_name', 'atom_index'])
df = df.drop(['EN', 'rad', 'atom_index'], axis=1)
df = df.rename(columns={'atom': 'atom_A', 'x': 'x_A', 'y': 'y_A', 'z': 'z_A',
                        'bonds': 'bonds_A', 'n_bonds': 'n_bonds_A', 'bond_lengths': 'bond_lengths_A'})

s = []
s1 = df['bonds_1'].apply(lambda x: set(x))
sA = df['bonds_A'].apply(lambda x: set(x))
for i in tqdm(range(len(df))):
    s.append(s1[i] & sA[i])
s = [list(x) for x in s]
df['atom_B_indices'] = s
df['atom_B_num'] = df['atom_B_indices'].apply(lambda x: len(x))
df['atom_B_num'].value_counts()

df = df.drop(['bonds_1', 'bond_lengths_1', 'bond_lengths_A'], axis=1)
atom_B_index = []
for thisIter in tqdm(df.iterrows()):
    if thisIter[1]['atom_B_num'] != 0:
        atom_B_index.append(thisIter[1]['atom_B_indices'][0])
    else:
        atom_B_index.append(thisIter[1]['bonds_A'][0])
df['atom_index_B'] = atom_B_index
df = df.drop(['bonds_A', 'atom_B_indices'], axis=1)
df = df.merge(structures, left_on=['molecule_name', 'atom_index_B'], right_on=[
              'molecule_name', 'atom_index'])
df = df.drop(['EN', 'rad', 'bonds', 'bond_lengths', 'atom_index'], axis=1)
df = df.rename(columns={'atom': 'atom_B', 'x': 'x_B',
                        'y': 'y_B', 'z': 'z_B', 'n_bonds': 'n_bonds_B'})


def add_angle_dist_features(df):
    data = []
    for thisIter in tqdm(df.iterrows()):
        thisIter = thisIter[1]
        r0 = np.array([thisIter['x_0'], thisIter['y_0'], thisIter['z_0']])
        r1 = np.array([thisIter['x_1'], thisIter['y_1'], thisIter['z_1']])
        rA = np.array([thisIter['x_A'], thisIter['y_A'], thisIter['z_A']])
        rB = np.array([thisIter['x_B'], thisIter['y_B'], thisIter['z_B']])
        r0AB = np.cross(r0 - rA, rB - rA)
        r0AB /= np.linalg.norm(r0AB)
        r1AB = np.cross(r1 - rB, rA - rB)
        r1AB /= np.linalg.norm(r1AB)
        d0A = np.linalg.norm(rA - r0)
        d1B = np.linalg.norm(rB - r1)
        dAB = np.linalg.norm(rB - rA)
        d0B = np.linalg.norm(rB - r0)
        d1A = np.linalg.norm(rA - r1)
        d01 = np.linalg.norm(r1 - r0)
        cosA = np.dot(r0 - rA, rB - rA) / (d0A * dAB)
        cosB = np.dot(r1 - rB, rA - rB) / (dAB * d1B)
        cosAB = np.dot(r0 - rA, r1 - rB) / (d0A * d1B)
        cosD = np.dot(r0AB, r1AB)
        data.append([d0A, d1B, dAB, d0B, d1A, d01, cosA, cosB, cosAB, cosD])

    cols = ['d0A', 'd1B', 'dAB', 'd0B', 'd1A', 'd01',
            'cosA', 'cosB', 'cos0A_1B', 'cos_Dihedral']
    res = df.copy()
    for col in cols:
        res[col] = 0
    res[cols] = data
    res = res.drop(['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1',
                    'x_A', 'y_A', 'z_A', 'x_B', 'y_B', 'z_B'], axis=1)
    return res


df = add_angle_dist_features(df)
df.to_csv(os.path.join(
    KAGGLE_DIR, 'neighborhood/3JHC/test_3JHC_v1.csv'), index=False)

# %%
# train
# 2J
train = pd.read_csv(TRAIN_CSV, index_col=0)
train = train[train['type'].isin(['2JHC', '2JHH', '2JHN'])]
train = train.merge(structures, left_on=[
                    'molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])
train = train.drop(['EN', 'rad', 'atom', 'atom_index',
                    'n_bonds', 'bond_lengths'], axis=1)
train['bonds'] = train['bonds'].apply(lambda x: x[0])
train = train.rename(columns={'bonds': 'atom_index_A'})
train = train.merge(structures, left_on=['molecule_name', 'atom_index_1'], right_on=[
                    'molecule_name', 'atom_index'], suffixes=['_0', '_1'])
train = train.drop(['EN', 'rad', 'atom', 'atom_index',
                    'bonds', 'bond_lengths'], axis=1)
train = train.merge(structures, left_on=['molecule_name', 'atom_index_A'], right_on=[
                    'molecule_name', 'atom_index'], suffixes=['_1', '_A'])
train = train.drop(
    ['EN', 'rad', 'atom_index', 'bonds', 'bond_lengths'], axis=1)
train = train.rename(
    columns={'atom': 'atom_A', 'x': 'x_A', 'y': 'y_A', 'z': 'z_A'})


def add_angle_dist_features(df):
    data = []
    for thisIter in tqdm(df.iterrows()):
        thisIter = thisIter[1]
        r0 = np.array([thisIter['x_0'], thisIter['y_0'], thisIter['z_0']])
        r1 = np.array([thisIter['x_1'], thisIter['y_1'], thisIter['z_1']])
        rA = np.array([thisIter['x_A'], thisIter['y_A'], thisIter['z_A']])
        d0A = np.linalg.norm(rA - r0)
        d1A = np.linalg.norm(rA - r1)
        d01 = np.linalg.norm(r1 - r0)
        cosA = np.dot(r0 - rA, r1 - rA) / (d0A * d1A)
        cos0 = np.dot(rA - r0, r1 - r0) / (d0A * d01)
        cos1 = np.dot(rA - r1, r0 - r1) / (d1A * d01)
        data.append([d0A, d1A, d01, cosA, cos0, cos1])

    cols = ['d0A', 'd1A', 'd01', 'cosA', 'cos0', 'cos1']
    res = df.copy()
    for col in cols:
        res[col] = 0
    res[cols] = data
    res = res.drop(['x_0', 'y_0', 'z_0', 'x_1', 'y_1',
                    'z_1', 'x_A', 'y_A', 'z_A'], axis=1)
    return res


df = add_angle_dist_features(train)
df.to_csv(os.path.join(KAGGLE_DIR, 'neighborhood/2J/train_2J_v1.csv'), index=False)

# %%
# test
# 2J
test = pd.read_csv(TEST_CSV, index_col=0)
test = test[test['type'].isin(['2JHC', '2JHH', '2JHN'])]
test = test.merge(structures, left_on=['molecule_name', 'atom_index_0'], right_on=[
                  'molecule_name', 'atom_index'])
test = test.drop(['EN', 'rad', 'atom', 'atom_index',
                  'n_bonds', 'bond_lengths'], axis=1)
test['bonds'] = test['bonds'].apply(lambda x: x[0])
test = test.rename(columns={'bonds': 'atom_index_A'})
test = test.merge(structures, left_on=['molecule_name', 'atom_index_1'], right_on=[
                  'molecule_name', 'atom_index'], suffixes=['_0', '_1'])
test = test.drop(['EN', 'rad', 'atom', 'atom_index',
                  'bonds', 'bond_lengths'], axis=1)
test = test.merge(structures, left_on=['molecule_name', 'atom_index_A'], right_on=[
                  'molecule_name', 'atom_index'], suffixes=['_1', '_A'])
test = test.drop(['EN', 'rad', 'atom_index', 'bonds', 'bond_lengths'], axis=1)
test = test.rename(
    columns={'atom': 'atom_A', 'x': 'x_A', 'y': 'y_A', 'z': 'z_A'})


def add_angle_dist_features(df):
    data = []
    for thisIter in tqdm(df.iterrows()):
        thisIter = thisIter[1]
        r0 = np.array([thisIter['x_0'], thisIter['y_0'], thisIter['z_0']])
        r1 = np.array([thisIter['x_1'], thisIter['y_1'], thisIter['z_1']])
        rA = np.array([thisIter['x_A'], thisIter['y_A'], thisIter['z_A']])
        d0A = np.linalg.norm(rA - r0)
        d1A = np.linalg.norm(rA - r1)
        d01 = np.linalg.norm(r1 - r0)
        cosA = np.dot(r0 - rA, r1 - rA) / (d0A * d1A)
        cos0 = np.dot(rA - r0, r1 - r0) / (d0A * d01)
        cos1 = np.dot(rA - r1, r0 - r1) / (d1A * d01)
        data.append([d0A, d1A, d01, cosA, cos0, cos1])

    cols = ['d0A', 'd1A', 'd01', 'cosA', 'cos0', 'cos1']
    res = df.copy()
    for col in cols:
        res[col] = 0
    res[cols] = data
    res = res.drop(['x_0', 'y_0', 'z_0', 'x_1', 'y_1',
                    'z_1', 'x_A', 'y_A', 'z_A'], axis=1)
    return res


df = add_angle_dist_features(test)
df.to_csv(os.path.join(KAGGLE_DIR, 'neighborhood/2J/test_2J_v1.csv'), index=False)
df
