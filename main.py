import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, log_loss
import warnings
import logging

from time import time


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def preprocess(df):
    df['cp_type'] = df['cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df['cp_time'] = df['cp_time'].map({24: 1, 48: 2, 72: 3})
    df['cp_dose'] = df['cp_dose'].map({'D1': 0, 'D2': 1})
    return df


def main():
    start_time = time()

    # Load Data
    df_train = pd.read_csv('data/train_features.csv')
    df_test = pd.read_csv('data/test_features.csv')
    df_target_s = pd.read_csv('data/train_targets_scored.csv')
    submission = pd.read_csv('data/sample_submission.csv')

    # Preprocess Data
    X_train = preprocess(df_train)
    X_test = preprocess(df_test)

    # PCA on Genes
    n_comp = 500

    genes = [col for col in X_train.columns if col.startswith('g-')]
    cells = [col for col in X_train.columns if col.startswith('c-')]

    data_genes = pd.concat([pd.DataFrame(X_train[genes]), pd.DataFrame(X_test[genes])])
    data_genes_pca = PCA(n_components=n_comp, random_state=42).fit_transform(data_genes)

    train_gene_pca = data_genes_pca[:X_train.shape[0]]
    test_gene_pca = data_genes_pca[-X_test.shape[0]:]

    train_gene_pca = pd.DataFrame(train_gene_pca, columns=[f'pca_G-{i}' for i in range(n_comp)])
    test_gene_pca = pd.DataFrame(test_gene_pca, columns=[f'pca_G-{i}' for i in range(n_comp)])

    # PCA on Cells
    n_comp = 50

    data_cells = pd.concat([pd.DataFrame(X_train[cells]), pd.DataFrame(X_test[cells])])
    data_cells_pca = PCA(n_components=n_comp, random_state=42).fit_transform(data_cells)

    train_cells_pca = data_cells_pca[:X_train.shape[0]]
    test_cells_pca = data_cells_pca[-X_test.shape[0]:]

    train_cells_pca = pd.DataFrame(train_cells_pca, columns=[f'pca_C-{i}' for i in range(n_comp)])
    test_cells_pca = pd.DataFrame(test_cells_pca, columns=[f'pca_C-{i}' for i in range(n_comp)])

    train_features = pd.concat((train_gene_pca, train_cells_pca), axis=1)
    test_features = pd.concat((test_gene_pca, test_cells_pca), axis=1)

    df_target_s = df_target_s.drop(['sig_id'], axis=1)

    # Linear Model - Logistic Regression
    kf = KFold(n_splits=3)
    kf.get_n_splits(train_features)

    log_Loss = 0.0
    y_pred, y_val_ = None, None
    for train_index, test_index in kf.split(train_features):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train_, X_val_ = train_features.iloc[train_index], train_features.iloc[test_index]
        y_train_, y_val_ = df_target_s.iloc[train_index], df_target_s.iloc[test_index]

        clf = OneVsRestClassifier(LogisticRegression())
        clf.fit(X_train_, y_train_)
        y_pred = clf.predict(X_val_)
        break

    for c in range(y_pred.shape[1]):
        y_pred_col = y_pred[:, c]
        y_val_col = y_val_.iloc[:, c]
        log_Loss += log_loss(y_val_col, y_pred_col)

    print(log_Loss)
    print("Used {:2f}s.".format(time() - start_time))


if __name__ == '__main__':
    main()
