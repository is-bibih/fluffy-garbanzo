import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression()

'''
more control over cross-validation
    - scikit-learn has a cross-validation splitter that allows for
      finer control
        - can be used for reproducible results, for example
    - default k-fold cross-validation (for regression) and
      stratified k-fold usually work well
    - shuffling can be an alternative for stratifying (and fix
      random_state to make it reproducible)

leave-one-out cross-validation
    - k-fold cross-validation where each fold is a single sample,
      and a single data point in each split is the test set
    - can be very time consuming for large datasets, but can
      provide better estimates on smaller datasets
'''

# import KFold splitter and instantiate it with desired folds
kfold = KFold(n_splits=5)

# pass kfold splitter as cv parameter
print("Cross-validation scores: \n{}".format(
    cross_val_score(logreg, iris.data, iris.target, cv=kfold)))

# look at scores with 3-fold nonstratified cross-validation
# (really bad since each fold corresponds to a class)
kfold = KFold(n_splits=3)
print("Cross-validation scores:\n{}".format(
    cross_val_score(logreg, iris.data, iris.target, cv=kfold)))

# shuffle before splitting
# (a lot better)
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
print("Cross-validation scores:\n{}".format(
    cross_val_score(logreg, iris.data, iris.target, cv=kfold)))

# use leave-one-out
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("Number of cv iterations: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))
