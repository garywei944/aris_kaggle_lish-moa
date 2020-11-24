import numpy as np
from sklearn.base import clone

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


class MultiOutputWithSampling:
    def __init__(self, model):
        self.model = model
        self.list_ = None

    def fit(self, X_train, y_train):
        v = y_train.shape[1]

        # Implement Multiprocess to fasten the training
        record, ids = [], []
        for i in range(v):
            model_ = clone(self.model)
            ids.append("{}_{}".format(i, id(model_)))
            pickle.dump((model_, X_train, y_train[:, i]), open('arg_{}.pkl'.format(ids[i]), 'wb'))

        pool = Pool(None)
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
