import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, log_loss, f1_score

import pickle
from time import time
from pathlib import Path
from os import path


def avg_log_loss(y_true, y_pred):
    r = 0.0

    # Add a dummy prediction to y_true and y_pred in case some label has all 0's
    y_true = np.vstack((y_true, np.ones((1, y_true.shape[1]))))
    y_pred = np.vstack((y_pred, np.ones((1, y_pred.shape[1]))))

    v = y_true.shape[1]
    for i in range(v):
        r += log_loss(y_true[:, i], y_pred[:, i])
    return r / v


def eval_model(model, X_train, y_train, id_=None):
    start_time = time()
    print('*' * 20)
    print("Evaluating model {}".format(model))

    output = None
    if id_:
        output = Path('output') / id_
        output.mkdir(parents=True, exist_ok=True)
        if path.exists(output / 'val.pkl'):
            score, y_pred_, y_val_ = pickle.load(open(output / 'val.pkl', 'rb'))
            print("Load result from disk, the Average Log Loss is {}".format(score))
            return score

    kf = KFold(n_splits=3)
    kf.get_n_splits(X_train)

    score = 0.0
    for train_index, test_index in kf.split(X_train):
        X_train_, X_val_ = X_train.iloc[train_index].values, X_train.iloc[test_index].values
        y_train_, y_val_ = y_train.iloc[train_index].values, y_train.iloc[test_index].values

        # Add dummy sample to make sure every column has 2 labels
        X_train_ = np.vstack((X_train_, np.zeros((1, X_train_.shape[1]))))
        y_train_ = np.vstack((y_train_, np.ones((1, y_train_.shape[1]))))

        model.fit(X_train_, y_train_)
        y_pred_ = model.predict(X_val_)
        score = avg_log_loss(y_pred_, y_val_)
        if id_:
            pickle.dump((score, y_pred_, y_val_), open(output / 'val.pkl', 'wb'))
        break
    print("The Average Log Loss is {}".format(score))
    print("Used {:2f}s".format(time() - start_time))
    return score
