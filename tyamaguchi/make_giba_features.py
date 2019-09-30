import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
import gc
tqdm.pandas()

DATA_PATH = Path('../data')
OUTPUT_PATH = DATA_PATH/'giba_dataset'


#%%
# Loading Data
train = pd.read_csv(DATA_PATH/'train.csv',index_col=0)
test = pd.read_csv(DATA_PATH/'test.csv',index_col=0)
mol_structures = pickle.load(open(DATA_PATH/'mol_features.pkl', 'rb'))
train.head()
test.head()
train.shape, test.shape

#%%
# Merge atom positions
structures = pd.read_csv(DATA_PATH/'structures.csv')
structures.head()

train = train.merge(structures, how='left',
                    left_on=['molecule_name','atom_index_0'],
                    right_on=['molecule_name','atom_index']).drop('atom_index',axis=1)
train = train.merge(structures, how='left', suffixes=['_0','_1'],
                    left_on=['molecule_name','atom_index_1'],
                    right_on=['molecule_name','atom_index']).drop('atom_index',axis=1)
test = test.merge(structures, how='left',
                  left_on=['molecule_name','atom_index_0'],
                  right_on=['molecule_name','atom_index']).drop('atom_index',axis=1)
test = test.merge(structures, how='left', suffixes=['_0','_1'],
                  left_on=['molecule_name','atom_index_1'],
                  right_on=['molecule_name','atom_index']).drop('atom_index',axis=1)

#%%
# Encode atoms and links
train_test = pd.concat([train,test])
train_test = train_test.iloc[0:10000]

# N: unique num of coupled atoms
train_test = train_test.merge(
    train_test.groupby(by=['molecule_name','atom_index_0']).progress_apply(
        lambda x:len(set(x['atom_index_1']))).reset_index().rename(
            columns={0:'N1'}),on=['molecule_name','atom_index_0'], how='left')
train_test = train_test.merge(
    train_test.groupby(by=['molecule_name','atom_index_1']).progress_apply(
        lambda x:len(set(x['atom_index_0']))).reset_index().rename(
            columns={0:'N2'}),on=['molecule_name','atom_index_1'], how='left')


# V: which coupling do each atom has
dt0 = train_test.groupby(
    by=['molecule_name','atom_index_0']).progress_apply(
        lambda x:' '.join(sorted(x['type']))).reset_index().rename(columns={0:'V1'})

dt1 = train_test.groupby(
    by=['molecule_name','atom_index_1']).progress_apply(
        lambda x:' '.join(sorted(x['type']))).reset_index().rename(columns={0:'V1'})

link0 = sorted(list(set(dt0['V1'].values)))
link0 = [[link0[i],i] for i in range(len(link0))]
link0 = pd.DataFrame(link0,columns=['V1','link0'])
dt0 = dt0.merge(link0)

link1 = sorted(list(set(dt1['V1'].values)))
link1 = [[link1[i],i] for i in range(len(link1))]
link1 = pd.DataFrame(link1,columns=['V1','link1'])
dt1 = dt1.merge(link1)


train_test = train_test.merge(
    dt0[['molecule_name','atom_index_0','link0']],
    on=['molecule_name','atom_index_0'],how='left')
del(dt0); gc.collect()

train_test = train_test.merge(
    dt1[['molecule_name','atom_index_1','link1']],
    on=['molecule_name','atom_index_1'],how='left')
del(dt1); gc.collect()

train_test['links'] = train_test['link0'].astype(str) + ' ' + train_test['link1'].astype(str)

train_test = train_test.merge(
    train_test['links'].value_counts().reset_index().rename( # count encoding
        columns={'index':'links','links':'linkN'}),on='links',how='left').drop('links',axis=1)

#%%
# Distances
train_test['dist_xyz'] = np.sqrt((train_test['x_0']-train_test['x_1'])**2+(train_test['y_0']-train_test['y_1'])**2+(train_test['z_0']-train_test['z_1'])**2)
# inv_dist: inverse of sum of inversed distances from coupled atoms
train_test=train_test.merge(
    train_test.groupby(by=['molecule_name','atom_index_0']).progress_apply(
        lambda x:1/(1/x['dist_xyz']**3).sum()).reset_index().rename(
            columns={0:'inv_dist0'}),on=['molecule_name','atom_index_0'],how='left')
train_test=train_test.merge(train_test.groupby(
    by=['molecule_name','atom_index_1']).progress_apply(
        lambda x:1/(1/x['dist_xyz']**3).sum()).reset_index().rename(
            columns={0:'inv_dist1'}),on=['molecule_name','atom_index_1'],how='left')
train_test['inv_distP'] = train_test['inv_dist0']*train_test['inv_dist1']/(train_test['inv_dist0']+train_test['inv_dist1'])
gc.collect()
train_test
#%%
# Some distance/radius based features
atomic_radius = pd.DataFrame({'atom':['H','C','N','O','F'],'R':[0.38,0.77,0.75,0.73,0.71],'E':[2.2,2.55,3.04,3.44,3.98]})
train_test = train_test.merge(atomic_radius.rename(columns={'atom':'atom_0','R':'R0','E':'E0'}),how='left')
train_test = train_test.merge(atomic_radius.rename(columns={'atom':'atom_1','R':'R1','E':'E1'}),how='left')
# Coupled
train_test = train_test.merge(
    train_test.groupby(by=['molecule_name','atom_index_0']).progress_apply(
        lambda x:1/(1/(x['dist_xyz']-x['R0']-x['R1'])**2).sum()).reset_index().rename(
            columns={0:'inv_dist0R'}),on=['molecule_name','atom_index_0'],how='left')
train_test = train_test.merge(
    train_test.groupby(by=['molecule_name','atom_index_1']).progress_apply(
        lambda x:1/(1/(x['dist_xyz']-x['R0']-x['R1'])**2).sum()).reset_index().rename(
            columns={0:'inv_dist1R'}),on=['molecule_name','atom_index_1'],how='left')
train_test['inv_distPR'] = train_test['inv_dist0R']*train_test['inv_dist1R']/(train_test['inv_dist0R']+train_test['inv_dist1R'])
train_test.to_csv(OUTPUT_PATH/'train_test.csv',index=False)
# train_test = pd.read_csv(OUTPUT_PATH/'train_test.csv')
gc.collect()
train_test=train_test.merge(
    train_test.groupby(by=['molecule_name','atom_index_0']).progress_apply(
        lambda x:1/(1/(x['dist_xyz']*(0.5*x['E0']+0.5*x['E1']))**2).sum()).reset_index().rename(
            columns={0:'inv_dist0E'}),on=['molecule_name','atom_index_0'],how='left')
train_test=train_test.merge(
    train_test.groupby(by=['molecule_name','atom_index_1']).progress_apply(
        lambda x:1/(1/(x['dist_xyz']*(0.5*x['E0']+0.5*x['E1']))**2).sum()).reset_index().rename(
            columns={0:'inv_dist1E'}),on=['molecule_name','atom_index_1'],how='left')
train_test['inv_distPE'] = train_test['inv_dist0E']*train_test['inv_dist1E']/(train_test['inv_dist0E']+train_test['inv_dist1E'])
train_test.to_pickle(OUTPUT_PATH/'train_test.pickle')


linkM0 = train_test[['inv_dist0','link0']].groupby('link0').mean().reset_index().rename(columns={'inv_dist0':'linkM0'})
train_test = train_test.merge(linkM0,on='link0',how='left')
train_test['linkM0'] = train_test['linkM0']-train_test['inv_dist0']
linkM1 = train_test[['inv_dist1','link1']].groupby('link1').mean().reset_index().rename(columns={'inv_dist1':'linkM1'})
train_test = train_test.merge(linkM1,on='link1',how='left')
train_test['linkM1'] = train_test['linkM1']-train_test['inv_dist1']

# Simple Aggregations
Aggregations0 = train_test[['molecule_name','atom_index_0','dist_xyz']]
Aggregations0 = Aggregations0.merge(train_test[['molecule_name','atom_index_0','dist_xyz']].groupby(by=['molecule_name','atom_index_0']).mean().reset_index().rename(columns={'dist_xyz':'mean_molecule_atom_0_dist_xyz'}),on=['molecule_name','atom_index_0'],how='left')
Aggregations0 = Aggregations0.merge(train_test[['molecule_name','atom_index_0','dist_xyz']].groupby(by=['molecule_name','atom_index_0']).std().fillna(0).reset_index().rename(columns={'dist_xyz':'std_molecule_atom_0_dist_xyz'}),on=['molecule_name','atom_index_0'],how='left')
Aggregations0 = Aggregations0.merge(train_test[['molecule_name','atom_index_0','dist_xyz']].groupby(by=['molecule_name','atom_index_0']).max().reset_index().rename(columns={'dist_xyz':'max_molecule_atom_0_dist_xyz'}),on=['molecule_name','atom_index_0'],how='left')
Aggregations0 = Aggregations0.merge(train_test[['molecule_name','atom_index_0','dist_xyz']].groupby(by=['molecule_name','atom_index_0']).min().reset_index().rename(columns={'dist_xyz':'min_molecule_atom_0_dist_xyz'}),on=['molecule_name','atom_index_0'],how='left')
Aggregations0 = Aggregations0.drop('dist_xyz',axis=1).drop_duplicates().reset_index(drop=True)
Aggregations1 = train_test[['molecule_name','atom_index_1','dist_xyz']]
Aggregations1 = Aggregations1.merge(train_test[['molecule_name','atom_index_1','dist_xyz']].groupby(by=['molecule_name','atom_index_1']).mean().reset_index().rename(columns={'dist_xyz':'mean_molecule_atom_1_dist_xyz'}),on=['molecule_name','atom_index_1'],how='left')
Aggregations1 = Aggregations1.merge(train_test[['molecule_name','atom_index_1','dist_xyz']].groupby(by=['molecule_name','atom_index_1']).std().fillna(0).reset_index().rename(columns={'dist_xyz':'std_molecule_atom_1_dist_xyz'}),on=['molecule_name','atom_index_1'],how='left')
Aggregations1 = Aggregations1.merge(train_test[['molecule_name','atom_index_1','dist_xyz']].groupby(by=['molecule_name','atom_index_1']).max().reset_index().rename(columns={'dist_xyz':'max_molecule_atom_1_dist_xyz'}),on=['molecule_name','atom_index_1'],how='left')
Aggregations1 = Aggregations1.merge(train_test[['molecule_name','atom_index_1','dist_xyz']].groupby(by=['molecule_name','atom_index_1']).min().reset_index().rename(columns={'dist_xyz':'min_molecule_atom_1_dist_xyz'}),on=['molecule_name','atom_index_1'],how='left')
Aggregations1 = Aggregations1.drop('dist_xyz',axis=1).drop_duplicates().reset_index(drop=True)

train_test = train_test.merge(Aggregations0,on=['molecule_name','atom_index_0'],how='left')
train_test = train_test.merge(Aggregations1,on=['molecule_name','atom_index_1'],how='left')


# Coulomb, Van der Walls & Yukawa interactions
structures = structures.copy()
structures.head(10)

data = []
for thisID,thisMolecule in tqdm(structures.groupby('molecule_name')):
    thisMolecule = thisMolecule.values
    for i in range(len(thisMolecule)):
        d = {'H':[],'C':[],'N':[],'O':[],'F':[]} # all distances grouped by atom type
        r = thisMolecule[i][3:]
        for j in range(len(thisMolecule)):
            if j == i:
                continue
            atom = thisMolecule[j][2]
            d[atom].append(np.linalg.norm(r-thisMolecule[j][3:]))
        thisData = [v for v in thisMolecule[i]]
        for key in d: # coulomb
            thisData.append(sum([1/r for r in d[key]]))
        for key in d: # vander
            thisData.append(sum([1/r**3 for r in d[key]]))
        for key in d: # yukawa
            thisData.append(sum([np.exp(-r) for r in d[key]]))
        data.append(thisData)

structures_pot = pd.DataFrame(data=data,columns=['molecule_name','atom_index','atom','x','y','z','coulomb_H','coulomb_C','coulomb_N','coulomb_O','coulomb_F','vander_H','vander_C','vander_N','vander_O','vander_F','yukawa_H','yukawa_C','yukawa_N','yukawa_O','yukawa_F'])
structures_pot.to_pickle(OUTPUT_PATH/'structures_pot.pickle')

train_test = train_test.merge(structures_pot.rename(columns={'atom_index':'atom_index_0'}),on=['molecule_name','atom_index_0'],how='left')
train_test = train_test.merge(structures_pot.rename(columns={'atom_index':'atom_index_1'}),on=['molecule_name','atom_index_1'],how='left',suffixes=['_0','_1'])

train_test.to_csv(OUTPUT_PATH/'train_test.csv',index=False)

# Counts
counts = train_test[['atom_0','atom_1','atom_index_0','atom_index_1','molecule_name','dist_xyz']]

# atom_0 = train_test['atom_0'].T.iloc[0].T
# atom_1 = train_test['atom_1'].T.iloc[0].T
# x_0 = train_test['x_0'].T.iloc[0].T
# y_0 = train_test['y_0'].T.iloc[0].T
# z_0 = train_test['z_0'].T.iloc[0].T
# x_1 = train_test['x_1'].T.iloc[0].T
# y_1 = train_test['y_1'].T.iloc[0].T
# z_1 = train_test['z_1'].T.iloc[0].T
#
# train_test = train_test.drop(['atom_0','atom_1','x_0','y_0','z_0','x_1','y_1','z_1'],axis=1)
# train_test['atom_0'] = atom_0
# train_test['atom_1'] = atom_1
# train_test['x_0'] = x_0
# train_test['y_0'] = y_0
# train_test['z_0'] = z_0
# train_test['x_1'] = x_1
# train_test['y_1'] = y_1
# train_test['z_1'] = z_1

counts
gc.collect()

data = []
done = 0
for thisMolecule in counts.groupby(by=['molecule_name','atom_index_0']):
    molecule_name = thisMolecule[0][0]
    atom_index_0 = thisMolecule[0][1]
    thisMolecule = thisMolecule[1].values
    d_n = {'C':0,'H':0,'N':0}
    d_d = {'C':0,'H':0,'N':0}
    for i in range(len(thisMolecule)):
        atom_1 = thisMolecule[i][1]
        dist = thisMolecule[i][5]
        d_n[atom_1] += 1
        d_d[atom_1] += dist
    thisData = [molecule_name,atom_index_0]
    for key in d_n:
        if d_n[key] == 0:
            thisData.append(np.nan)
        else:
            thisData.append(d_n[key])
    for key in d_n:
        if d_n[key] == 0:
            thisData.append(np.nan)
        else:
            thisData.append(d_d[key]/d_n[key])
    data.append(thisData)
    done += 1
    if done%10000==0:
        print('{} record done'.format(done))

counts0 = pd.DataFrame(data=data,columns=['molecule_name','atom_index_0','countC0','countH0','countN0','distC0','distH0','distN0'])


data = []
done = 0
for thisMolecule in counts.groupby(by=['molecule_name','atom_index_1']):
    molecule_name = thisMolecule[0][0]
    atom_index_1 = thisMolecule[0][1]
    thisMolecule = thisMolecule[1].values
    d_n = {'C':0,'H':0,'N':0}
    d_d = {'C':0,'H':0,'N':0}
    for i in range(len(thisMolecule)):
        atom_0 = thisMolecule[i][0]
        dist = thisMolecule[i][5]
        d_n[atom_0] += 1
        d_d[atom_0] += dist
    thisData = [molecule_name,atom_index_1]
    for key in d_n:
        if d_n[key] == 0:
            thisData.append(np.nan)
        else:
            thisData.append(d_n[key])
    for key in d_n:
        if d_n[key] == 0:
            thisData.append(np.nan)
        else:
            thisData.append(d_d[key]/d_n[key])
    data.append(thisData)
    done += 1
    if done%10000==0:
        print('{} record done'.format(done))

counts1 = pd.DataFrame(data=data,columns=['molecule_name','atom_index_1','countC1','countH1','countN1','distC1','distH1','distN1'])

train_test = train_test.merge(counts0,on=['molecule_name','atom_index_0'],how='left')
train_test = train_test.merge(counts1,on=['molecule_name','atom_index_1'],how='left')
train_test.to_csv(OUTPUT_PATH/'train_test.csv',index=False)

# Distance features by atom type
#%%
train_test = pd.read_csv(OUTPUT_PATH/'train_test.csv')
data = []
done = 0
for thisMolecule in train_test[['molecule_name','atom_index_0','atom_index_1','atom_1','dist_xyz']].groupby(by=['molecule_name','atom_index_0']):
    molecule_name = thisMolecule[0][0]
    atom_index_0 = thisMolecule[0][1]
    thisMolecule = thisMolecule[1].values
    for i in range(len(thisMolecule)):
        d = {'H':[],'C':[],'N':[]}
        atom_index_1 = thisMolecule[i][2]
        thisData = [molecule_name,atom_index_0,atom_index_1]
        for j in range(len(thisMolecule)):
            if j == i:
                continue
            atom_1 = thisMolecule[j][3]
            d[atom_1].append(thisMolecule[j][4])
        adH = sorted(d['H'])[:4]
        adC = sorted(d['C'])[:4]
        adN = sorted(d['N'])[:4]
        adH += [np.nan]*(4-len(adH))
        adC += [np.nan]*(4-len(adC))
        adN += [np.nan]*(4-len(adN))
        thisData += adH+adC+adN
        data.append(thisData)
    done += 1
    if done%10000==0:
        print('{} record done'.format(done))

adHCN = pd.DataFrame(data=data,columns=['molecule_name','atom_index_0','atom_index_1','adH1','adH2','adH3','adH4','adC1','adC2','adC3','adC4','adN1','adN2','adN3','adN4'])
train_test = train_test.merge(adHCN,on=['molecule_name','atom_index_0','atom_index_1'],how='left')

data = []
for thisMolecule in structures[['molecule_name','atom']].groupby('molecule_name'):
    d = {'C':0,'H':0,'N':0,'F':0,'O':0}
    thisData = [thisMolecule[0]]
    thisMolecule = thisMolecule[1].values
    for i in range(len(thisMolecule)):
        d[thisMolecule[i][1]] += 1
    for key in d:
        thisData.append(d[key])
    data.append(thisData)

Natom = pd.DataFrame(data=data,columns=['molecule_name','NC','NH','NN','NF','NO'])
train_test = train_test.merge(Natom,on='molecule_name',how='left')
train_test = train_test.drop(['x_0','y_0','z_0','x_1','y_1','z_1','atom_0','atom_1','R0','E0','R1','E1'],axis=1)

train = train_test[:len(train)]
test = train_test[len(train):]
train.to_csv(OUTPUT_PATH/'train.csv',index=False)
test.to_csv(OUTPUT_PATH/'test.csv',index=False)
