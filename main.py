import pandas as pd
import numpy as np

import logging
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from preprocess import load_and_process, pca
from metrics import eval_model


def main():
    X_train, X_test, y_train = load_and_process()
    train_features, test_features = pca(X_train, X_test)

    # Linear Model
    eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4), n_jobs=-1), train_features, y_train,
               id_='lr_no_sampling')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    main()
