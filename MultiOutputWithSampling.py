import numpy as np
from sklearn.base import clone

import logging
import pickle
from multiprocessing import Process
from pathlib import Path
import os
import sys

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def _mp_fit(model, X_train, y_train, id_, i):
    logging.debug("Start to fit label {}".format(i))
    model.fit(X_train, y_train)
    pickle.dump(model, open(Path('mp_{}.pkl'.format(id_)), 'wb'))


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
            process = Process(target=_mp_fit, args=(model_, X_train, y_train[:, i], ids[i], i))
            record.append(process)
            process.start()

        for process in record:
            process.join()

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
