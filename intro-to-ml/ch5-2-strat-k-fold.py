import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_iris

iris = load_iris()

'''
stratified k-fold cross-validation
    - splitting the data in order into k folds might not
      always be the best idea (because it might have similar
      values)
    - in stratified k-fold cross-validation, the proportion
      between classes is the same in each fold as in the whole
      dataset
    - usually a good idea to use it instead of standard k-fold
      cross-validation
'''

# look at how it's not good to split iris into k-folds
print("Iris labels:\n{}".format(iris.target))

# look at how stratified cross-val splits data
mglearn.plots.plot_stratified_cross_validation()
plt.show()
