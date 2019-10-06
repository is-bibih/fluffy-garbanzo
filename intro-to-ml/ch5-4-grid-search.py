import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

'''
grid search
    - tries all possible combinations of the parameters of interest
      for a model
    - it helps with parameter tuning to improve model generalization
    - for example, two parameters with six possible values yields
      a grid of 36 parameter settings for the model
    - can be implemented by looping over the parameter values
    - the same test set cannot be used to define parameters and to
      evaluate the model's generalization
      - since the parameters are optimized for that test set, and
        would not necessarily generalize well to other datasets
      - this can be solved by splitting the dataset into 3 groups
        - training set
        - validation set (development, for parameters)
        - test set
      - once the best parameters are found with the validation set,
        a model can be built and trained on both the training and
        the validation data, to get as much information as possible
'''

# naive grid search implementation

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0)
print("Size of training set: {}     size of test set: {}".format(
      X_train.shape[0], X_test.shape[0]))

best_score = 0

# train an SVC for each combinations
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        # evaluate SVC
        score = svm.score(X_test, y_test)
        # compare scores
        if score > best_score:
            best_score = score
            best_parameters =  {'C': C, 'gamma': gamma}

print("Best score: {:.2f}".format(best_score))
print("Best parameters: {}".format(best_parameters))

# grid search split into 
