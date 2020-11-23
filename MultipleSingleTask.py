import numpy as np
from sklearn.base import clone


class MultipleSingleTask:
    def __init__(self, model):
        self.model = model
        self.list_ = None

    def fit(self, X_train, y_train):
        v = y_train.shape[1]
        self.list_ = [clone(self.model) for _ in range(v)]
        for i in range(v):
            self.list_[i].fit(X_train, y_train.iloc[:, i])

    def predict(self, X_test):
        n = X_test.shape[0]
        v = len(self.list_)
        y_pred = np.zeros((n, v))
        for i in range(v):
            y_pred[:i] = self.list_[i].predict(X_test)
        return y_pred
