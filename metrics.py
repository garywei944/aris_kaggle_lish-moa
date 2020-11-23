from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, log_loss, f1_score

from time import time


def avg_log_loss(y_true, y_pred):
    r = 0.0
    v = y_pred.shape[1]
    for i in range(v):
        r += f1_score(y_true.iloc[:, i], y_pred[:, i])
    return r / v


def eval_model(model, X_train, y_train):
    def warn():
        pass

    import warnings
    warnings.warn = warn

    start_time = time()
    print('*' * 20)
    print("Evaluating model {}".format(model))
    kf = KFold(n_splits=3)
    kf.get_n_splits(X_train)

    score = 0.0
    for train_index, test_index in kf.split(X_train):
        X_train_, X_val_ = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_, y_val_ = y_train.iloc[train_index], y_train.iloc[test_index]

        model.fit(X_train_, y_train_)
        y_pred_ = model.predict(X_val_)
        score += avg_log_loss(y_pred_, y_train_)
        break
    print("The Average Log Loss is {}".format(score))
    print("Used {:2f}s".format(time() - start_time))
    return score
