import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, \
    ParameterGrid, StratifiedKFold
from sklearn.datasets import load_iris

iris = load_iris()

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

'''
nested cross-validation
    - the data is split into test and training sets in different
      ways
    - a grid search is run on each split
    - the test set score is reported for each split
    - it returns a list of scores
        - usually used to evaluate how well a model works on a
          dataset
    - it is an embarassingly parallel task (the results do not
      depend on the other parallel processes, it can be distributed
      on multiple CPU cores or a cluster)
        - set n_jobs to amount of cores to use or -1 to use all
        - scikit-learn does not allow nesting of parallel operations
'''

# implementation with scikit-learn

def scikit_nested():
    scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5),
                             iris.data, iris.target, cv=5)
    print("Cross-validation scores: ", scores)
    print("Mean cross-validation score: ", scores.mean())

# manual implementation

def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
    outer_scores = []
    # for each split of data in the outer cross-validation
    # uses indices returned by split
    for training_samples, test_samples in outer_cv.split(X, y):
        # find best parameter for the split using cross-val
        best_params = {}
        best_score = -np.inf
        # iterate over parameters
        for parameters in parameter_grid:
            # compare scores in inner splits
            cv_scores = []
            # iterate over inner cross-val
            for inner_train, inner_test in inner_cv.split(
                    X[training_samples], y[training_samples]):
                # build classifier on parameters
                clf = Classifier(**parameters)
                clf.fit(X[inner_train], y[inner_train])
                # evaluate on inner test
                score = clf.score(X[inner_test], y[inner_test])
                cv_scores.append(score)
            # compute mean score over inner folds
            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = parameters
        # build classifier on best parameters with outer set
        clf = Classifier(**best_params)
        clf.fit(X[training_samples], y[training_samples])
        # evaluate
        outer_scores.append(clf.score(X[test_samples], y[test_samples]))
    return np.array(outer_scores)

# run nested_cv
def run_nested():
    scores = nested_cv(iris.data, iris.target, StratifiedKFold(5),
            StratifiedKFold(5), SVC, ParameterGrid(param_grid))
    print("Cross-validation scores: {}".format(scores))

run_nested()
