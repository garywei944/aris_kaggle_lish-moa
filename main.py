import pandas as pd
import numpy as np

import logging
import sys

from config import config

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor, MultiOutputEstimator
from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocess import load_and_process, pca, mlsmote
from metrics import eval_model
from MultiOutputWithSampling import MultiOutputWithSampling


def main():
    X_train, X_test, y_train = load_and_process()

    # # Without PCA
    # # LR no sampling
    # eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4), n_jobs=-1), X_train, y_train,
    #            id_='lr_no_sampling')
    # # LR separate over sampling
    # eval_model(MultiOutputWithSampling(LogisticRegression(max_iter=1e4)), X_train,
    #            y_train, id_='lr_separate_sampling')
    # # Random Forest no sampling
    # eval_model(RandomForestClassifier(n_estimators=200, max_depth=10, random_state=43, min_samples_split=10), X_train,
    #            y_train, id_='rf_200_10_43_10_no_sampling')
    # # Random Forest separate sampling
    # eval_model(MultiOutputWithSampling(
    #     RandomForestClassifier(n_estimators=200, max_depth=10, random_state=43, min_samples_split=10)),
    #     X_train, y_train, id_='rf_200_10_43_10_separate_sampling')
    # # NN no sampling
    # eval_model(MLPClassifier(random_state=1, max_iter=1500), X_train, y_train, id_='mlp_1')
    #
    # # Tuning on PCA
    # for n_genes in [100, 200, 400, 500]:
    #     for n_cells in [25, 50, 75]:
    #         train_features, test_features = pca(X_train, X_test, n_genes, n_cells)
    #         # LR no sampling
    #         eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4), n_jobs=-1), train_features, y_train,
    #                    id_='{}_{}_lr_no_sampling'.format(n_genes, n_cells))
    #         # LR no sampling C=0.01
    #         eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4), n_jobs=-1), train_features, y_train,
    #                    id_='{}_{}_c_0.01_lr_no_sampling'.format(n_genes, n_cells))
    #         # LR separate sampling
    #         eval_model(MultiOutputWithSampling(LogisticRegression(max_iter=1e4)), train_features,
    #                    y_train, id_='{}_{}_lr_separate_sampling'.format(n_genes, n_cells))
    #         # Random Forest no sampling
    #         eval_model(RandomForestClassifier(n_estimators=200, max_depth=10, random_state=43, min_samples_split=10,
    #                                           n_jobs=-1), train_features, y_train,
    #                    id_='{}_{}_rf_200_10_43_10_no_sampling'.format(n_genes, n_cells))
    #         # Random Forest separate sampling
    #         eval_model(MultiOutputWithSampling(
    #             RandomForestClassifier(n_estimators=200, max_depth=10, random_state=43, min_samples_split=10)),
    #             train_features, y_train, id_='{}_{}_rf_200_10_43_10_separate_sampling'.format(n_genes, n_cells))

    # Use n_genes=200 n_cells=50 as best PCA
    train_features, test_features = pca(X_train, X_test, 200, 50)
    eval_model(MultiOutputRegressor(Ridge(alpha=1.0)), train_features, y_train,
               id_='ridge_200_50_a_1.00_lr_no_sampling')

    # Linear Regression, no sampling, Tuning on C with PCA 200, 50
    tuning_c = np.logspace(-2, 3, 10)
    for c in tuning_c:
        eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4, C=c), n_jobs=-1), train_features,
                   y_train,
                   id_='200_50_c_{:.2f}_lr_no_sampling'.format(c))

    # Tuning on sampling strategy, LR separate sampling C=0.01
    # 0.01, 0.02, and 0.03 result in errors
    for ss in [0.04, 0.1, 0.2, 0.25, 0.5]:
        eval_model(MultiOutputWithSampling(LogisticRegression(max_iter=1e4, C=0.01), sampling_strategy=ss),
                   train_features, y_train, id_='200_50_c_0.01_ss_{}_lr_separate_sampling'.format(ss))

    # Staking Random Forest and Neural Network
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=43, min_samples_split=10)),
        # ('lr', MultiOutputClassifier(LogisticRegression(max_iter=1e4, C=0.01), n_jobs=-1)),
        ('nn', MLPClassifier(random_state=1, max_iter=1500))
    ]
    try:
        eval_model(StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression(max_iter=1e4)),
                   train_features, y_train, id_='200_50_stack_rf_lr')
    except ValueError:
        print("ValueError Encountered!")

    # NN model
    eval_model(MLPClassifier(random_state=1, max_iter=1500), train_features, y_train, id_='200_50_mlp_1')

    # Failed since the generated data only contains label 0
    # Oversampling on the entire dataset with MLSMOTE
    X_train_os, y_train_os = mlsmote(train_features, y_train, train_features.shape[0])
    eval_model(MultiOutputClassifier(LogisticRegression(max_iter=1e4), n_jobs=-1), X_train_os, y_train_os,
               id_='lr_complete_sampling')


if __name__ == '__main__':
    config()
main()
