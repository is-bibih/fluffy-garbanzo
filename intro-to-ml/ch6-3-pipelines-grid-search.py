import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

'''
pipelines in grid searches
    - a parameter grid is defined, and then a GridSearchCV is
      constructed from the parameter grid and the pipeline
        - it's necessary to specify which step the parameters
          belong to
        - specifying steps:
          {step name}__{parameter name}
'''

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

def build_pipe(X_train, X_test, y_train, y_test):
    # declare steps
    pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
    # fit pipeline
    pipe.fit(X_train, y_train)
    return pipe

def grid_search(pipe, show=True):
    param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
    print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
    print("Best parameters: {}".format(grid.best_params_))

    if show:
        mglearn.plots.plot_proper_processing()
        plt.show()

pipe = build_pipe(X_train, X_test, y_train, y_test)
grid_search(pipe)
