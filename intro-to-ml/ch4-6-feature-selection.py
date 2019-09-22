import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()

'''
automatic feature selection
    - adding features makes models more complex, so it increases
      chance of overfitting
    - when adding more features or working with high-dimensional
      datasets, it can be useful to leave only the most
      features so that the models are simpler and generalize
      better
    - supervised methods (need the target) that can help pick
      features
        - univariate statistics
        - model-based selection
        - iterative selection

univariate statistics
    - computes whether there is a statistically significant
      relationship between each individual feature and the
      target
        - it is not alanalyzed whether they are significant
          in combination with another feature
    - in classification it is aka analysis of variance (anova)
    - outcomes on real data are usually mixed
    - helpful if the data is so high-dimensional that it would
      be hard to train a model on it, or if many features are
      suspected to be useless

univariate statistics in scikit-learn
    - choose a test
        - usually either default f_classif for classification or
          f_regression for regression
    - choose a method to discard features based on high p-values
        - simplest is SelectKBest, which selects a fixed number
          k of features
        - SelectPercentile selects a fixed percentage of
          features
'''

# feature selection for classification on cancer dataset
# add noninformative noise features that should be removed
# by feature selection

# get deterministic random numbers
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))

# add noise to data
# 30 original features, 50 noise
X_w_noise = np.hstack((cancer.data, noise))

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X_w_noise, cancer.target, random_state=0, test_size=0.5)

# use f_classif and SelectPercentile to select 50% of features
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)

# transform training set
X_train_selected = select.transform(X_train)

print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))

# see which features were selected with get_support (returns
# a boolean mask of selected features)

mask = select.get_support()
print(mask)

# look at mask: black True, white False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("sample index")
plt.show()
# most selected features are original, most discarded features
# are noise

# compare logreg on all features vs only selected features

# transform test data
X_test_selected = select.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Score with all features: {:.3f}".format(
    lr.score(X_test, y_test)))
lr.fit(X_train_selected, y_train)
print("Score with only selected features: {:.3f}".format(
    lr.score(X_test_selected, y_test)))
