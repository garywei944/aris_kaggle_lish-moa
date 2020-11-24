import pandas as pd
import numpy as np

import logging
import sys
from config import config

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier

from preprocess import load_and_process, pca, mlsmote
from metrics import eval_model
from MultiOutputWithSampling import MultiOutputWithSampling


def main():
    X_train, X_test, y_train = load_and_process()

    # Without PCA
    eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4), n_jobs=-1), X_train, y_train,
               id_='lr_no_sampling')
    eval_model(MultiOutputWithSampling(LogisticRegression(max_iter=1e4)), X_train,
               y_train, id_='lr_separate_sampling')
    eval_model(MultiOutputWithSampling(
        RandomForestClassifier(n_estimators=200, max_depth=10, random_state=43, min_samples_split=10)),
        X_train, y_train, id_='rf_200_10_43_10_separate_sampling')
    eval_model(MLPClassifier(random_state=1, max_iter=1500), X_train, y_train, id_='mlp_1')

    # Tuning on PCA
    for n_genes in [100, 200, 400, 500]:
        for n_cells in [25, 50, 75]:
            train_features, test_features = pca(X_train, X_test, n_genes, n_cells)
            eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4), n_jobs=-1), train_features, y_train,
                       id_='{}_{}_lr_no_sampling'.format(n_genes, n_cells))
            eval_model(MultiOutputWithSampling(LogisticRegression(max_iter=1e4)), train_features,
                       y_train, id_='{}_{}_lr_separate_sampling'.format(n_genes, n_cells))
            eval_model(MultiOutputWithSampling(
                RandomForestClassifier(n_estimators=200, max_depth=10, random_state=43, min_samples_split=10)),
                train_features, y_train, id_='{}_{}_rf_200_10_43_10_separate_sampling'.format(n_genes, n_cells))

    # Failed since the generated data only contains label 0
    # # Oversampling on the entire dataset with MLSMOTE
    # X_train_os, y_train_os = mlsmote(train_features, y_train, train_features.shape[0])
    # eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4), n_jobs=-1), X_train_os, y_train_os,
    #            id_='lr_complete_sampling')


if __name__ == '__main__':
    config()
    main()
