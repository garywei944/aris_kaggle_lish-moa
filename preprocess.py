import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

from MLSMOTE import MLSMOTE

import logging
from os import path
import pickle


def preprocess(df):
    df['cp_type'] = df['cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df['cp_time'] = df['cp_time'].map({24: 1, 48: 2, 72: 3})
    df['cp_dose'] = df['cp_dose'].map({'D1': 0, 'D2': 1})
    return df


def load_and_process():
    df_train = pd.read_csv('data/train_features.csv', index_col=0)
    df_test = pd.read_csv('data/test_features.csv', index_col=0)
    df_target_s = pd.read_csv('data/train_targets_scored.csv', index_col=0)

    X_train = preprocess(df_train)
    X_test = preprocess(df_test)

    return X_train, X_test, df_target_s


def pca(X_train, X_test, n_gene=200, n_cell=50):
    genes = [col for col in X_train.columns if col.startswith('g-')]
    cells = [col for col in X_train.columns if col.startswith('c-')]

    # PCA genes
    data_genes = pd.concat([pd.DataFrame(X_train[genes]), pd.DataFrame(X_test[genes])])
    data_genes_pca = PCA(n_components=n_gene, random_state=42).fit_transform(data_genes)

    train_gene_pca = data_genes_pca[:X_train.shape[0]]
    test_gene_pca = data_genes_pca[-X_test.shape[0]:]

    train_gene_pca = pd.DataFrame(train_gene_pca, columns=[f'pca_G-{i}' for i in range(n_gene)])
    test_gene_pca = pd.DataFrame(test_gene_pca, columns=[f'pca_G-{i}' for i in range(n_gene)])

    # PCA cells
    data_cells = pd.concat([pd.DataFrame(X_train[cells]), pd.DataFrame(X_test[cells])])
    data_cells_pca = PCA(n_components=n_cell, random_state=42).fit_transform(data_cells)

    train_cells_pca = data_cells_pca[:X_train.shape[0]]
    test_cells_pca = data_cells_pca[-X_test.shape[0]:]

    train_cells_pca = pd.DataFrame(train_cells_pca, columns=[f'pca_C-{i}' for i in range(n_cell)])
    test_cells_pca = pd.DataFrame(test_cells_pca, columns=[f'pca_C-{i}' for i in range(n_cell)])

    # Generate new training and test data
    train_features = pd.concat((train_gene_pca, train_cells_pca), axis=1)
    test_features = pd.concat((test_gene_pca, test_cells_pca), axis=1)
    return train_features, test_features


# Deprecated, failed since it generates all label 0
def mlsmote(X_train, y_train, n_sample, neigh=5):
    if path.exists('over_sampling.pkl'):
        return pickle.load(open('over_sampling.pkl', 'rb'))
    else:
        X_train_os, y_train_os = MLSMOTE(X_train, y_train, X_train.shape[0])
        pickle.dump((X_train_os, y_train_os), open('over_sampling.pkl', 'wb'))
        return X_train_os, y_train_os
