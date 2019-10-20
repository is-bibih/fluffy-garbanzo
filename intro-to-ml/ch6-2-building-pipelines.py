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
building pipelines
    - to build a pipeline, you provide it a list of steps
        - each step is a tuple containing a string name (it can't
          have __) and an instance of an estimator
'''

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

# example pipe with training and scaling, which can be used for
# cross_val_score or GridSearchCV
def ex_pipe(X_train, X_test, y_train, y_test):
    # declare steps
    pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
    # fit pipeline
    pipe.fit(X_train, y_train)
    # evaluate on test data
    score = pipe.score(X_test, y_test)
    print("Test score: {:.2f}".format(score))

ex_pipe(X_train, X_test, y_train, y_test)
