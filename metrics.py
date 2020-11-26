import numpy as np

from sklearn.metrics import roc_auc_score, log_loss, f1_score
from skmultilearn.model_selection import IterativeStratification

import logging
import pickle
from time import time
from pathlib import Path
from os import path
import matplotlib.pyplot as plt

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


def scorer(y_true, y_pred):
    log_loss_, auc, f1 = 0.0, 0.0, 0.0

    # Add a dummy prediction to y_true and y_pred in case some label has all 0's
    y_true = np.vstack((y_true, np.ones((1, y_true.shape[1]))))
    y_pred = np.vstack((y_pred, np.ones((1, y_pred.shape[1]))))

    v = y_true.shape[1]
    for i in range(v):
        log_loss_ += log_loss(y_true[:, i], y_pred[:, i])
        auc += roc_auc_score(y_true[:, i], y_pred[:, i])
        f1 += f1_score(y_true[:, i], y_pred[:, i] > 0.5)
    return log_loss_ / v, auc / v, f1 / v


def eval_model(model, X_train, y_train, id_=None):
    start_time = time()
    logging.info('*' * 20)
    logging.info("Evaluating model {}".format(id_ if id_ else model))

    n_splits = 3
    output = None

    # Try to load saved result from disk if exists
    if id_:
        output = Path('output') / id_
        output.mkdir(parents=True, exist_ok=True)
        if path.exists(output / 'score.pkl'):
            logging.debug("Loading result from disk")
            log_loss_, auc, f1 = pickle.load(open(output / 'score.pkl', 'rb'))
            logging.info("The Average Log Loss is {}".format(log_loss_))
            logging.info("The Average AUC is {}".format(auc))
            logging.info("The Average f1 is {}".format(f1))
            return log_loss_, auc, f1

    # Deprecated sklearn k-forld
    # kf = StratifiedKFold(n_splits=n_splits)
    # kf.get_n_splits(X_train)

    kf = IterativeStratification(n_splits=3, order=1)

    log_loss_, auc, f1 = 0.0, 0.0, 0.0
    for i, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
        X_train_, X_val_ = X_train.iloc[train_index].values, X_train.iloc[test_index].values
        y_train_, y_val_ = y_train.iloc[train_index].values, y_train.iloc[test_index].values

        # Add dummy sample to make sure every column has 2 labels
        X_train_ = np.vstack((X_train_, np.zeros((1, X_train_.shape[1]))))
        y_train_ = np.vstack((y_train_, np.ones((1, y_train_.shape[1]))))

        model.fit(X_train_, y_train_)
        y_pred_ = model.predict(X_val_)

        log_loss_val, auc_val, f1_val = scorer(y_val_, y_pred_)

        # Pickle y_val_ and y_pred_
        if id_:
            pickle.dump((y_val_, y_pred_), open(output / "val_{}.pkl".format(i), 'wb'))

        # Update the scores
        log_loss_ += log_loss_val
        auc += auc_val
        f1 += f1_val

    log_loss_ /= n_splits
    auc /= n_splits
    f1 /= n_splits
    if id_:
        pickle.dump((log_loss_, auc, f1), open(output / 'score.pkl', 'wb'))
    logging.info("The Average Log Loss is {}".format(log_loss_))
    logging.info("The Average AUC is {}".format(auc))
    logging.info("The Average f1 is {}".format(f1))
    logging.info("Used {:.2f}s".format(time() - start_time))
    return log_loss_, auc, f1


def make_hist_plot(df, title, id_):
    def __plot(score):
        logging.debug(df[score])
        x = list(df.index)
        y = df[score].values
        plt.figure(figsize=(16, 9))
        plt.bar(x, y)
        plt.xticks(x, x, rotation='vertical')
        plt.ylabel(score)
        plt.title("{} {}".format(title, score))
        plt.subplots_adjust(bottom=0.4)

        plt.savefig("fig/{}_{}.png".format(id_, score))
        plt.show()

    for col in df:
        __plot(col)


def make_plot(df, title, id_, x_label=None, log_x=False):
    def __plot(score):
        logging.debug(df[score])
        x = list(df.index)
        y = df[score].values
        # plt.figure(figsize=(16, 9))
        plt.plot(x, y)
        if x_label:
            plt.xlabel(x_label)
        plt.ylabel(score)
        if log_x:
            plt.xscale('log')
        plt.title("{} {}".format(title, score))

        plt.savefig("fig/{}_{}.png".format(id_, score))
        plt.show()

    for col in df:
        __plot(col)
