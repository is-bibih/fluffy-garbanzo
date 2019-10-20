import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

'''
the general pipeline interface
    - it can join any number of estimators together
    - all but the last estimator must have a transform method,
      which produces a new representation of the data to be used
      in the next step
        - the last step needs to have a fit method
    - during the call to Pipeline.fit, fit and then transform
      is called on each step

pipeline creation with make_pipeline
    - creates a pipeline and names each step based on its class
    - if there are several steps with the same class, a number is
      appended to the name

accessing step attributes
    - the named_steps attribute is a dict with the step names and
      the estimators, which is useful for inspecting attributes
      of the steps

accessing attributes in a grid-searched pipeline
    - a common task is to access steps of a pipeline inside a
      grid search
'''

cancer = load_breast_cancer()

# pipe fit implementation
def fit(self, X, y):
    X_transformed = X
    for name, estimator in self.steps[:-1]:
        # iterate over the steps except the last one
        X_transformed = estimator.fit_transform(X_transformed, y)
    # fit the last step
    self.steps[-1][1].fit(X_transformed, y)
    return self

# pipe predict implementation
def predict(self, X):
    X_transformed = X
    for step in self.steps[:-1]:
        # iterate over all but the final step
        X_transformed = step[1].transform(X_transformed)
    # fit the last step
    return self.steps[-1][1].predict(X_transformed)

def ex_make_pipe():
    # standard syntax
    pipe_long = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC(C=100))])
    # abbreviated syntax
    pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))
    # look at step names
    print("Pipeline steps:\n{}".format(pipe_short.steps))

    # pipe with numbers in names
    pipe = make_pipeline(StandardScaler(), PCA(n_components=2),
        StandardScaler())
    print("Pipeline steps:\n{}".format(pipe.steps))

    return pipe

def get_attributes(pipe):
    pipe.fit(cancer.data)
    # get principal components from the pca step
    components = pipe.named_steps["pca"].components_
    print("components.shape: {}".format(components.shape))

def grid_search_attrs(cancer):
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    # parameter grid for c
    param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}

    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=4)

    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, y_train)
    # get best estimator
    print("Best estimator:\n{}".format(grid.best_estimator_))
    # access logisticregression step using named_steps
    print("LogisticRegression step:\n{}".format(
          grid.best_estimator_.named_steps["logisticregression"]))
    # look at coefficients
    print("LogisticRegression coefficients:\n{}".format(
          grid.best_estimator_. \
          named_steps["logisticregression"].coef_))

# pipe = ex_make_pipe()
# get_attributes(pipe)
grid_search_attrs(cancer)
