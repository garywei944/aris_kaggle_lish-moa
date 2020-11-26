import pandas as pd
import numpy as np

import logging
import sys
from multiprocessing import cpu_count
import pickle

from config import config

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.neural_network import MLPClassifier

from preprocess import load_and_process, pca, mlsmote
from metrics import eval_model, make_hist_plot, make_plot
from MultiOutputWithSampling import MultiOutputWithSampling

n_jobs = cpu_count()


def main():
    X_train, X_test, y_train = load_and_process()

    # Tuning PCA
    cols = ['log_loss', 'roc_auc', 'f1']
    # lr_df = pd.DataFrame(columns=cols)
    # ridge_df = lr_df.copy()
    # for n_genes in [100, 200, 400, 500]:
    #     for n_cells in [25, 50, 75]:
    #         train_features, test_features = pca(X_train, X_test, n_genes, n_cells)
    #         # LR no sampling
    #         ll, auc, f1 = eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4), n_jobs=n_jobs),
    #                                  train_features, y_train, id_='lr_{}_{}_no_sampling'.format(n_genes, n_cells))
    #         lr_df = lr_df.append(
    #             pd.Series([ll, auc, f1], name='lr_{}_{}_no_sampling'.format(n_genes, n_cells), index=cols))
    #         # Ridge no sampling
    #         ll, auc, f1 = eval_model(MultiOutputRegressor(Ridge(alpha=1.0), n_jobs=n_jobs), train_features, y_train,
    #                                  id_='ridge_{}_{}_a_1.00_no_sampling'.format(n_genes, n_cells))
    #         ridge_df = ridge_df.append(
    #             pd.Series([ll, auc, f1], name='ridge_{}_{}_a_1.00_no_sampling'.format(n_genes, n_cells), index=cols))
    # df = lr_df.append(ridge_df)

    # pickle.dump(df, open('tuning_pca_df.pkl', 'wb'))
    # df = pickle.load(open('tuning_pca_df.pkl', 'rb'))
    # make_hist_plot(df, 'Tuning PCA', 'tuning_pca')

    # Use 100, 25 as best PCA
    train_features, test_features = pca(X_train, X_test, 100, 25)

    # # Ridge, Tuning a
    # tuning_a = np.logspace(-2, 3, 6)
    # df = pd.DataFrame(columns=cols)
    # for a in tuning_a:
    #     ll, auc, f1 = eval_model(MultiOutputRegressor(Ridge(alpha=a), n_jobs=n_jobs), train_features, y_train,
    #                              id_='ridge_100_25_a_{:.2f}_no_sampling'.format(a))
    #     df = df.append(pd.Series([ll, auc, f1], name=a, index=cols))
    # make_plot(df, "Ridge, Tuning a", "ridge_tuning_a", x_label='a', log_x=True)
    # # The best ridge is a=1000
    best_ridge = Ridge(alpha=1000)
    #
    # # LR, Tuning C
    # tuning_c = np.logspace(-2, 3, 6)
    # df = pd.DataFrame(columns=cols)
    # for c in tuning_c:
    #     ll, auc, f1 = eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4, C=c), n_jobs=n_jobs),
    #                              train_features, y_train, id_='lr_100_25_c_{:.2f}_no_sampling'.format(c))
    #     df = df.append(pd.Series([ll, auc, f1], name=c, index=cols))
    # make_plot(df, "LR, Tuning C", "lr_tuning_c", x_label='C', log_x=True)
    # # The best LR is C=0.01
    best_lr = LogisticRegression(max_iter=1e4, C=0.01)

    # Tuning over sampling per label, tested with lr
    df = pd.DataFrame(columns=cols)
    ll, auc, f1 = eval_model(MultiOutputClassifier(best_lr, n_jobs=n_jobs), train_features, y_train,
                             id_='lr_100_25_c_0.01_no_sampling')
    df = df.append(pd.Series([ll, auc, f1], name='lr_100_25_c_0.01_no_sampling', index=cols))
    # 0.01, 0.02, and 0.03 result in errors
    for ss in [0.04, 0.1, 0.2, 0.25, 0.5]:
        ll, auc, f1 = eval_model(
            MultiOutputWithSampling(LogisticRegression(max_iter=1e4, C=0.01), sampling_strategy=ss, n_jobs=n_jobs),
            train_features, y_train, id_='lr_100_25_c_0.01_ss_{}_separate_sampling'.format(ss))
        df = df.append(pd.Series([ll, auc, f1], name='lr_100_25_c_0.01_ss_{}_separate_sampling'.format(ss), index=cols))
    make_hist_plot(df, "Tuning sampling strategy per label on LR, C=0.01", 'sampling')
    # No separate sampling performs better

    # Random Forest, tuning n_estimators and max_depth
    df = pd.DataFrame(columns=cols)
    for n_estimators in [50, 200, 500]:
        for max_depth in [3, 6, 10]:
            ll, auc, f1 = eval_model(MultiOutputRegressor(
                RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=43,
                                      min_samples_split=10), n_jobs=n_jobs), train_features, y_train,
                id_='rfr_{}_{}_43_10_no_sampling'.format(n_estimators, max_depth))
            df = df.append(pd.Series([ll, auc, f1], name='rfr_{}_{}_43_10_no_sampling'.format(n_estimators, max_depth),
                                     index=cols))
    make_hist_plot(df, "Random Forest, Tuning n_estimators & max_depth", "rfr_tuning_n_d")


if __name__ == '__main__':
    config()
    main()
