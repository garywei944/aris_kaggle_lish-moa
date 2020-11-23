import pandas as pd
import numpy as np

import logging
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from preprocess import load_and_process, pca
from metrics import eval_model


def main():
    X_train, X_test, y_train = load_and_process()
    train_features, test_features = pca(X_train, X_test)

    # Linear Model
    eval_model(OneVsRestClassifier(LogisticRegression()), train_features, y_train)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    main()
