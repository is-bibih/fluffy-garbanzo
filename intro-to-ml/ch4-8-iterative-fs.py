import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.feature_selection import RFE
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()

# get deterministic random numbers
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))

# add noise to data
# 30 original features, 50 noise
X_w_noise = np.hstack((cancer.data, noise))

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X_w_noise, cancer.target, random_state=0, test_size=0.5)

'''
automatic feature selection

iterative feature selection
    - builds a series of models with different numbers of features
    - two basic methods
        - starting with no features and adding them until some
          criterion is reached
        - starting with all features and removing them until some
          criterion is reached
    - much more computationally expensive than the other two
    - recursive feature elimination (RFE): starts with all the
      features and deletes the least important one, then repeats
      the process with a new model that has all but the discarded
      feature
        - repeated recursively until a specified number of features
          is left
    - usually does better than univariate and model-based, but takes
      significantly longer
    - can make a linear model perform as well as tree-based models
    - ususally unlikely to provide large performance improvements

rfe in scikit-learn
    - can use the model inside the selector to make predictions
      (using only the feature set that was selected)
'''

# use random forest model

select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),
             n_features_to_select=40)

select.fit(X_train, y_train)
# look at selected features
mask = select.get_support()
# plt.matshow(mask.reshape(1, -1), cmap='gray_r')
# plt.xlabel("Sample index")
# plt.show()

# look at accuracy of logreg with rfe for fs

X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)

score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("Test score: {:.3f}".format(score))

# use model in rfe with only the selected feature set
print("Test score: {:.3f}".format(select.score(X_test, y_test)))
