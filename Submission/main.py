import pandas as pd
import numpy as np

import logging
import sys
from multiprocessing import cpu_count
import pickle

from config import config

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, RandomForestRegressor, StackingRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from preprocess import load_and_process, pca, mlsmote
from metrics import eval_model, make_hist_plot, make_plot
from MultiOutputWithSampling import MultiOutputWithSampling

n_jobs = cpu_count()


def main():
    X_train, X_test, y_train, submission = load_and_process()

    # Tuning PCA
    cols = ['log_loss', 'roc_auc', 'f1']
    lr_df = pd.DataFrame(columns=cols)
    ridge_df = lr_df.copy()
    for n_genes in [100, 200, 400, 500]:
        for n_cells in [25, 50, 75]:
            train_features, test_features = pca(X_train, X_test, n_genes, n_cells)
            # LR no sampling
            ll, auc, f1 = eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4), n_jobs=n_jobs),
                                     train_features, y_train, id_='lr_{}_{}_no_sampling'.format(n_genes, n_cells))
            lr_df = lr_df.append(
                pd.Series([ll, auc, f1], name='lr_{}_{}_no_sampling'.format(n_genes, n_cells), index=cols))
            # Ridge no sampling
            ll, auc, f1 = eval_model(MultiOutputRegressor(Ridge(alpha=1.0), n_jobs=n_jobs), train_features, y_train,
                                     id_='ridge_{}_{}_a_1.00_no_sampling'.format(n_genes, n_cells))
            ridge_df = ridge_df.append(
                pd.Series([ll, auc, f1], name='ridge_{}_{}_a_1.00_no_sampling'.format(n_genes, n_cells), index=cols))
    df = lr_df.append(ridge_df)

    # pickle the df to save time
    # pickle.dump(df, open('tuning_pca_df.pkl', 'wb'))
    # df = pickle.load(open('tuning_pca_df.pkl', 'rb'))
    make_hist_plot(df, 'Tuning PCA', 'tuning_pca')

    # Use 100, 25 as best PCA
    train_features, test_features = pca(X_train, X_test, 100, 25)

    # Failed since the generated data only contains label 0
    # Oversampling on the entire dataset with MLSMOTE
    X_train_os, y_train_os = mlsmote(train_features, y_train, train_features.shape[0])
    eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4), n_jobs=-1), X_train_os, y_train_os,
               id_='lr_complete_sampling')

    # Tuning over sampling per label, tested with lr
    df = pd.DataFrame(columns=cols)
    ll, auc, f1 = eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4, C=0.01), n_jobs=n_jobs),
                             train_features, y_train, id_='lr_100_25_c_0.01_no_sampling')
    df = df.append(pd.Series([ll, auc, f1], name='lr_100_25_c_0.01_no_sampling', index=cols))
    # 0.01, 0.02, and 0.03 result in errors
    for ss in [0.04, 0.1, 0.2, 0.25, 0.5]:
        ll, auc, f1 = eval_model(
            MultiOutputWithSampling(LogisticRegression(max_iter=1e4, C=0.01), sampling_strategy=ss, n_jobs=n_jobs),
            train_features, y_train, id_='lr_100_25_c_0.01_ss_{}_separate_sampling'.format(ss))
        df = df.append(pd.Series([ll, auc, f1], name='lr_100_25_c_0.01_ss_{}_separate_sampling'.format(ss), index=cols))
    make_hist_plot(df, "Tuning sampling strategy per label on LR, C=0.01", 'sampling')
    # No separate sampling performs better

    # Ridge, Tuning a
    tuning_a = np.logspace(-2, 3, 6)
    df = pd.DataFrame(columns=cols)
    for a in tuning_a:
        ll, auc, f1 = eval_model(MultiOutputRegressor(Ridge(alpha=a), n_jobs=n_jobs), train_features, y_train,
                                 id_='ridge_100_25_a_{:.2f}_no_sampling'.format(a))
        df = df.append(pd.Series([ll, auc, f1], name=a, index=cols))
    make_plot(df, "Ridge, Tuning a", "ridge_tuning_a", x_label='a', log_x=True)
    # The best ridge is a=1000
    best_ridge = Ridge(alpha=1000)

    # LR, Tuning C
    tuning_c = np.logspace(-2, 3, 6)
    df = pd.DataFrame(columns=cols)
    for c in tuning_c:
        ll, auc, f1 = eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4, C=c), n_jobs=n_jobs),
                                 train_features, y_train, id_='lr_100_25_c_{:.2f}_no_sampling'.format(c))
        df = df.append(pd.Series([ll, auc, f1], name=c, index=cols))
    make_plot(df, "LR, Tuning C", "lr_tuning_c", x_label='C', log_x=True)
    # The best LR is C=0.01
    best_lr = LogisticRegression(max_iter=1e4, C=0.01)

    # Random Forest, tuning max_depth
    df = pd.DataFrame(columns=cols)
    # for n_estimators in [50, 200, 500]:
    for max_depth in [1, 3, 6, 10]:
        ll, auc, f1 = eval_model(MultiOutputRegressor(
            RandomForestRegressor(n_estimators=50, max_depth=max_depth, random_state=43,
                                  min_samples_split=10), n_jobs=n_jobs), train_features, y_train,
            id_='rfr_{}_{}_43_10_no_sampling'.format(50, max_depth))
        df = df.append(pd.Series([ll, auc, f1], name=max_depth, index=cols))
    make_plot(df, "Random Forest, Tuning max_depth", "rfr_tuning_d", x_label='max_depth')
    # The best Random Forest is max_depth=3
    best_rf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=50, max_depth=3, random_state=43, min_samples_split=10), n_jobs=n_jobs)

    # NN model, Tuning hidden_layer
    df = pd.DataFrame(columns=cols)
    best_loss = np.inf
    best_h = None
    for i in [50, 100, 200]:
        for j in [50, 100, 200]:
            for k in [50, 100, 200]:
                ll, auc, f1 = eval_model(
                    MLPRegressor(hidden_layer_sizes=(i, j, k), random_state=1, max_iter=1500, learning_rate='adaptive',
                                 warm_start=True), train_features, y_train, id_='nn_100_25_h_{}_{}_{}'.format(i, j, k))
                if best_loss > ll:
                    best_loss = ll
                    best_h = (i, j, k)
                df = df.append(pd.Series([ll, auc, f1], name='nn_100_25_h_{}_{}_{}'.format(i, j, k), index=cols))
    make_hist_plot(df, "Neural Network", "nn_tuning_hidden")
    logging.info("The best hidden layer configuration is {}".format(best_h))
    logging.info("The best log loss is {}".format(best_loss))
    # The best nn is hidden_layer=(200,100,100)
    best_nn = MLPRegressor(hidden_layer_sizes=(200, 100, 100), random_state=1, max_iter=1500, warm_start=True)

    # # Staking Random Forest and Neural Network
    # base_estimators = [
    #     ('rf', RandomForestRegressor(n_estimators=50, max_depth=3, random_state=43, min_samples_split=10)),
    #     ('nn', be_nn)
    # ]
    # eval_model(
    #     MultiOutputRegressor(StackingRegressor(estimators=base_estimators, final_estimator=Ridge()), n_jobs=n_jobs),
    #     train_features, y_train, id_='100_50_stack_rf_lr')

    # Predict with the best model
    best_model = best_rf
    best_model.fit(train_features, y_train)
    y_pred = best_model.predict(test_features)
    pickle.dump((best_model, y_pred), open("best_model.pkl", 'wb'))
    # best_model, y_pred = pickle.load(open("best_model.pkl", 'rb'))
    pd.DataFrame(y_pred, index=submission.index, columns=submission.columns).to_csv('submission.csv')


if __name__ == '__main__':
    config()
    main()
