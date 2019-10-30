import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.base import BaseEstimator, TransformerMixin

# build ur own estimator that works with scikit-learn

class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, first_parameter=1, second_parameter=2):
        # all parameters must be specified in init
        self.first_parameter = 1
        self.second_parameter = 2

    def fit(self, X, y=None):
        # fit should only take X and y as parameters
        # even if the model is unsupervised, it needs to accept
        # a y argument

        # fit the model
        print("fit the model")
        # fit returns self
        return self

    def transform(self, X):
        # transform takes as a parameter only X

        # transform X
        X_transformed = X + 1
        return X_transformed
