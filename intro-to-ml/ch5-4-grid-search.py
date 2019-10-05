import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from SVC import SVC
from sklearn.datasets import load_iris

iris = load_iris()

'''
grid search
    - tries all possible combinations of the parameters of interest
      for a model
    - it helps with parameter tuning to improve model generalization
    - for example, two parameters with six possible values yields
      a grid of 36 parameter settings for the model
    - can be implemented by looping over the parameter values
'''

# naive grid search implementation
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0)
