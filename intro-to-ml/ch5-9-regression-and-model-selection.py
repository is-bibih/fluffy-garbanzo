import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, train_test_split, \
    GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics.scorer import SCORERS

'''
regression metrics
    - evaluation can be done in similar detail for classification
        - like analyzing overpredicting the target vs undepredicting
    - r^2 is usually enough tho

evaluation metrics in model selection
    - like using auc in GridSearchCV
    - it can be done by passing scoring parameter
        - default is accuracy
    - scoring parameter possible values
        - accuracy (default)
        - roc_auc
        - average_precision
        - f1 (for binary)
        - f1_macro
        - f1_micro
        - f1_weighted
        - r2
        - mean_squared_error
        - mean_absolute_error
'''

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target == 9, random_state=0)

# evaluate svm classifier on binary digits dataset using auc
def scoring_cross_val(digits):
    # using accuracy
    print("Default scoring: {}".format(
        cross_val_score(SVC(), digits.data, digits.target == 9)))
    # using roc auc
    roc_auc = cross_val_score(SVC(), digits.data, digits.target == 9,
                              scoring="roc_auc")
    print("AUC scoring: {}".format(roc_auc))

# get parameters for svm using different scoring values
def scoring_grid(X_train, X_test, y_train, y_test):
    # bad grid to illustrate the point
    param_grid = {'gamma': [0.001, 0.01, 1, 10]}

    # using default (accuracy)
    grid = GridSearchCV(SVC(), param_grid=param_grid)
    grid.fit(X_train, y_train)
    print("Grid-Search with accuracy")
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score (accuracy): {:.3f}".format(
        grid.best_score_))
    print("Test set AUC: {:.3f}".format(
        roc_auc_score(y_test, grid.decision_function(X_test))))
    print("Test set accuracy: {:.3f}".format(grid.score(X_test, y_test)))

    # with AUC scoring
    grid = GridSearchCV(SVC(), param_grid=param_grid, scoring="roc_auc")
    grid.fit(X_train, y_train)
    print("\nGrid-Search with AUC")
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score (AUC): {:.3f}".format(
        grid.best_score_))
    print("Test set AUC: {:.3f}".format(
        roc_auc_score(y_test, grid.decision_function(X_test))))
    print("Test set accuracy: {:.3f}".format(grid.score(X_test, y_test)))

# possible values for scoring
def sco_values():
    print("Available scorers:\n{}".format(sorted(SCORERS.keys())))

# scoring_cross_val(digits)
scoring_grid(X_train, X_test, y_train, y_test)
sco_values()
