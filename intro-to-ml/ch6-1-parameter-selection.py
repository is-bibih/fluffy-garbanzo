import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

"""
algorithm chains and pipelines
    - the pipeline class simplifies chaining together different
      algorithms for preprocessing and machine learning models

parameter selection with preprocessing
    - you want to scale a set and then do cross-validation on it
        - if you just scale it and then cross-validate on it, you
          scale it including part of the test set, which is wrong
          (you already use information from the test set)
    - any process that extracts knowledge from the dataset should be
      done on the training setuse
    - cross-validation should be the outermost loop in processing
    - the correct order can be achieved with the pipeline class
        - it puts together multiple processing steps into a single
          estimator
        - it has fit, predict, and score methods
        - its most common use is chaining together preprocessing steps
          with a supervised model
"""

cancer = load_breast_cancer()

# example of the importance of chaining models
# splitting the data, computing min-max, scaling, and training a svm
def ex_chain(cancer):
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0)
    # compute min and max on training data
    scaler = MinMaxScaler().fit(X_train)
    # rescale
    X_train_scaled = scaler.transform(X_train)

    svm = SVC()
    # learn on scaled training data
    svm.fit(X_train_scaled, y_train)
    # scale test data and score it
    X_test_scaled = scaler.transform(X_test)
    print("Test score: {:.2f}".format(svm.score(X_test_scaled, y_test)))

    return X_train_scaled, X_test_scaled, y_train, y_test

# naive approach to grid search (don't do it)
def naive(X_train_scaled, X_test_scaled, y_train, y_test, show=True):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
    grid.fit(X_train_scaled, y_train)
    print("Best cross-validation accuracy: {:.2f}".format(
        grid.best_score_))
    print("Best set score: {:.2f}".format(
        grid.score(X_test_scaled, y_test)))
    print("Best parameters: ", grid.best_params_)

    # look at how bad it is
    if show:
        mglearn.plots.plot_improper_processing()
        plt.show()

X_train_scaled, X_test_scaled, y_train, y_test = ex_chain(cancer)
naive(X_train_scaled, X_test_scaled, y_train, y_test)
