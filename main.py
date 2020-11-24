import pandas as pd
import numpy as np

import logging
import sys
from config import config

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from preprocess import load_and_process, pca, mlsmote
from metrics import eval_model
from MultiOutputWithSampling import MultiOutputWithSampling


def main():
    X_train, X_test, y_train = load_and_process()

    # PCA with 200 genes and 50 cells
    train_features, test_features = pca(X_train, X_test)

    # Linear Model
    eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4), n_jobs=-1), train_features, y_train,
               id_='lr_no_sampling')
    eval_model(MultiOutputWithSampling(LogisticRegression(max_iter=1e4)), train_features,
               y_train, id_='lr_separate_sampling')
    eval_model(MultiOutputWithSampling(
        RandomForestClassifier(n_estimators=200, max_depth=10, random_state=43, min_samples_split=10)), train_features,
        y_train, id_='rf_200_10_43_10_no_sampling')

    # PCA with 500 genes and 75 cells
    # TODO

    # Failed since the generated data only contains label 0
    # # Oversampling on the entire dataset with MLSMOTE
    # X_train_os, y_train_os = mlsmote(train_features, y_train, train_features.shape[0])
    # eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4), n_jobs=-1), X_train_os, y_train_os,
    #            id_='lr_complete_sampling')


if __name__ == '__main__':
    config()
    main()
