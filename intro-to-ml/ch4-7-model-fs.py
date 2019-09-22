import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
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

model-based feature selection
    - uses a supervised machine learning model to judge the
      importance of each feature and keeps the most
      important ones
    - the model used for fs does not have to be the same one
      that is used for modeling
    - the model needs to provide a measure of importance
        - trees have feature_importances_
        - linear models have coeffcients (their magnitude can
          be used as a measure of importance)
    - more complex than univariate tests

model-based fs in scikit-learn with SelectFromModel
    - selects all features that have an importance measure of
      the feature greater than the provided threshold
        - median: the statistic median, results in the selection
          of half the features
'''

# random forest classifier as feature selector

# instantiate selector
select = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='median')

# fit the model and select
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_l1.shape: {}".format(X_train_l1.shape))

# look at selected features
mask = select.get_support()
# black for true, white for false
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
plt.show()

# look at performance
X_test_l1 = select.transform(X_test)
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
print("Test score: {:.3f}".format(score))
