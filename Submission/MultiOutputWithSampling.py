import numpy as np
from sklearn.base import clone
from imblearn.over_sampling import SMOTE

import logging
import pickle
from multiprocessing import Process, Pool
from pathlib import Path
import os
import sys
from config import config

config()


def _mp_fit(*args, **kwargs):
    id_, i = args[0]
    model, X_train, y_train = pickle.load(open('arg_{}.pkl'.format(id_), 'rb'))
    os.remove('arg_{}.pkl'.format(id_))
    logging.debug("Start to fit label {}".format(i))
    model.fit(X_train, y_train)
    pickle.dump(model, open(Path('mp_{}.pkl'.format(id_)), 'wb'))
    logging.debug("Finished fitting label {}".format(i))


def _over_sampling(X_train, y_train, sampling_strategy=0.2):
    # Failed but don't know why
    # # Oversampling so that #0:#1 = 1:4
    # n_neighbors = 5
    #
    # # Decrease n_neighbors if there is no enough samplings
    # while n_neighbors > 0:
    #     # logging.debug("Enter while")
    #     try:
    #         oversample = SMOTE(sampling_strategy=0.25, k_neighbors=n_neighbors, n_jobs=-1)
    #         X, y = oversample.fit_resample(X_train, y_train)
    #     except ValueError:
    #         logging.debug(n_neighbors)
    #         n_neighbors -= 2
    #     else:
    #         return X, y

    # Copy the first sample with label=1 6 times
    if np.sum(y_train) < 6:
        index = np.where(y_train == 1)[0][0]
        x_ = X_train[index]
        X_train = np.vstack((X_train, [x_ for _ in range(6)]))
        # logging.debug(y_train)
        y_train = np.hstack((y_train, np.ones(6)))

    oversample = SMOTE(sampling_strategy=sampling_strategy)
    X, y = oversample.fit_resample(X_train, y_train)
    return X, y


class MultiOutputWithSampling:
    def __init__(self, model, sampling_strategy=0.2, n_jobs=None):
        self.model = model
        self.list_ = None
        self.sampling_strategy = sampling_strategy
        self.n_jobs = n_jobs

    def fit(self, X_train, y_train):
        v = y_train.shape[1]

        # Implement Multiprocess to fasten the training
        record, ids = [], []
        for i in range(v):
            model_ = clone(self.model)
            ids.append("{}_{}".format(i, id(model_)))
            # logging.debug("sampling {}".format(i))
            X, y = _over_sampling(X_train, y_train[:, i], self.sampling_strategy)
            pickle.dump((model_, X, y), open('arg_{}.pkl'.format(ids[i]), 'wb'))

        pool = Pool(self.n_jobs)
        pool.map(_mp_fit, zip(ids, range(v)))
        pool.close()
        pool.join()

        self.list_ = [None for _ in range(v)]
        for i in range(v):
            file_path = 'mp_{}.pkl'.format(ids[i])
            self.list_[i] = pickle.load(open(file_path, 'rb'))
            os.remove(file_path)

    def predict(self, X_test):
        n = X_test.shape[0]
        v = len(self.list_)
        y_pred = np.zeros((n, v))
        for i in range(v):
            y_pred[:, i] = self.list_[i].predict(X_test)
        return y_pred
