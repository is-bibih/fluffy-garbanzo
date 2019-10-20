import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

'''
grid-searching which model to use
    - it's possible to search over the steps being performed in
      the pipeline
        - this leads to an even bigger search space, so it is
          usually not a viable machine learning strategy
'''

# example comparing a random forest and an svc on the iris datset
# svc needs StandardScaler, random forest does not

pipe = Pipeline([("preprocessing", StandardScaler()),
                 ("classifier", SVC())])

# use list of search grids since they'd have different parameters

param_grid = [
    {"classifier": [SVC()],
     "preprocessing": [StandardScaler(), None],
     "classifier__gamma": [0.001, 0.01, 0.1, 1, 10, 100],
     "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100]},
    {"classifier": [RandomForestClassifier(n_estimators=100)],
     "preprocessing": [None],
     "classifier__max_features": [1, 2, 3]}]

# instantiate and train grid

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best params:\n{}\n".format(grid.best_params_))
print("Best cross-val score: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
