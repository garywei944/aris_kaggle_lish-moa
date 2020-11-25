import pandas as pd
import numpy as np

import logging
import sys

from config import config

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.neural_network import MLPClassifier

from preprocess import load_and_process, pca, mlsmote
from metrics import eval_model
from MultiOutputWithSampling import MultiOutputWithSampling


def main():
    X_train, X_test, y_train = load_and_process()

    # Tuning PCA
    for n_genes in [100, 200, 400, 500]:
        for n_cells in [25, 50, 75]:
            train_features, test_features = pca(X_train, X_test, n_genes, n_cells)
            # Ridge no sampling
            eval_model(MultiOutputRegressor(Ridge(alpha=1.0), n_jobs=-1), train_features, y_train,
                       id_='ridge_{}_{}_a_1.00_lr_no_sampling'.format(n_genes, n_cells))

    # Use 100, 25 as best PCA
    train_features, test_features = pca(X_train, X_test, 100, 25)
    # Ridge separate sampling
    eval_model(MultiOutputWithSampling(Ridge(alpha=1.0)), train_features, y_train,
               id_='ridge_100_25_a_1.00_lr_separate_sampling')


if __name__ == '__main__':
    config()
    main()
