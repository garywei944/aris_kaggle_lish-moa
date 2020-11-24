import pandas as pd
import numpy as np

import logging
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from preprocess import load_and_process, pca
from metrics import eval_model
from MultiOutputWithSampling import MultiOutputWithSampling


def main():
    X_train, X_test, y_train = load_and_process()
    train_features, test_features = pca(X_train, X_test)

    # Linear Model
    eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4), n_jobs=-1), train_features, y_train,
               id_='lr_no_sampling')
    eval_model(MultiOutputWithSampling(LogisticRegression(max_iter=1e4)), train_features,
               y_train, id_='demo_lr')
    eval_model(MultiOutputWithSampling(
        RandomForestClassifier(n_estimators=200, max_depth=10, random_state=43, min_samples_split=10)), train_features,
        y_train, id_='rf_200_10_43_10_no_sampling')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    main()
