import numpy as np
import pandas as pd

# from rdkit import Chem
# from rdkit.Chem import Descriptors

def FeatureNormalization(mol_graphs, feat_name):
    features = [getattr(g, feat_name) for g in mol_graphs]
    features_mean = np.mean(features)
    features_std = np.std(features)

    for g in mol_graphs:
        val = getattr(g, feat_name)
        if features_std == 0:
            setattr(g, feat_name, 0)
        else:
            setattr(g, feat_name, (val - features_mean) / features_std)

def Z_Score(X):
    if len(X.shape) == 1:
        means = np.mean(X)
        stds = np.std(X)

        for i in range(0, X.shape[0]):
            if stds == 0:
                X[i] = 0
            else:
                X[i] = (X[i] - means) / stds
    else:
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)

        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[1]):
                if stds[j] == 0:
                    X[i, j] = 0
                else:
                    X[i, j] = (X[i, j] - means[j]) / stds[j]
    return X

def adj_mat_to_edges(adj_mat):
    edges = []

    for i in range(0, adj_mat.shape[0]):
        for j in range(0, adj_mat.shape[1]):
            if adj_mat[i, j] == 1:
                edges.append((i, j))

    return edges

def atoms_to_symbols(atoms):
    return [atom.GetSymbol() for atom in atoms]

def weight_reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()