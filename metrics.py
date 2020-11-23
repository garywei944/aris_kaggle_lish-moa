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


def scorer(y_true, y_pred):
    log_loss_, auc, f1 = 0.0, 0.0, 0.0

    # Add a dummy prediction to y_true and y_pred in case some label has all 0's
    y_true = np.vstack((y_true, np.ones((1, y_true.shape[1]))))
    y_pred = np.vstack((y_pred, np.ones((1, y_pred.shape[1]))))

    v = y_true.shape[1]
    for i in range(v):
        log_loss_ += log_loss(y_true[:, i], y_pred[:, i])
        auc += roc_auc_score(y_true[:, i], y_pred[:, i])
        f1 += f1_score(y_true[:, i], y_pred[:, i])
    return log_loss_ / v, auc / v, f1 / v


def eval_model(model, X_train, y_train, id_=None):
    start_time = time()
    print('*' * 20)
    print("Evaluating model {}".format(model))

    n_splits = 3
    output = None
    if id_:
        output = Path('output') / id_
        output.mkdir(parents=True, exist_ok=True)
        if path.exists(output / 'val.pkl'):
            log_loss_, auc, f1 = pickle.load(open(output / 'val.pkl', 'rb'))
            print("The Average Log Loss is {}".format(log_loss_))
            print("The Average AUC is {}".format(auc))
            print("The Average f1 is {}".format(f1))
            return log_loss_, auc, f1

    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(X_train)

    log_loss_, auc, f1 = 0.0, 0.0, 0.0
    for train_index, test_index in kf.split(X_train):
        X_train_, X_val_ = X_train.iloc[train_index].values, X_train.iloc[test_index].values
        y_train_, y_val_ = y_train.iloc[train_index].values, y_train.iloc[test_index].values

        # Add dummy sample to make sure every column has 2 labels
        X_train_ = np.vstack((X_train_, np.zeros((1, X_train_.shape[1]))))
        y_train_ = np.vstack((y_train_, np.ones((1, y_train_.shape[1]))))

        model.fit(X_train_, y_train_)
        y_pred_ = model.predict(X_val_)
        log_loss_val, auc_val, f1_val = scorer(y_pred_, y_val_)

        # Update the scores
        log_loss_ += log_loss_val
        auc += auc_val
        f1 += f1_val

    log_loss_ /= n_splits
    auc /= n_splits
    f1 /= n_splits
    if id_:
        pickle.dump((log_loss_, auc, f1), open(output / 'val.pkl', 'wb'))
    print("The Average Log Loss is {}".format(log_loss_))
    print("The Average AUC is {}".format(auc))
    print("The Average f1 is {}".format(f1))
    print("Used {:.2f}s".format(time() - start_time))
    return log_loss_, auc, f1
