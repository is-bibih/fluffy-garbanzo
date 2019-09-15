import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs

cancer = load_breast_cancer()


'''
unsupervised learning
    - algorithm just extracts knowledge from input
    - hard to evaluate whether they did well due to the lack of
      labelled data

unsupervised transformations
    - create new representations of the input, which might be
      easier for humans or other ml algorithms to undestand
    - dimensionality reduction: represents high-dimensional
      data with fewer features
        - like making something 2d for visualization
    - findng elements that make up data
        - like topic extraction on text documents

clustering algorithms
    - partition data into groups of similar items

preprocessing and scaling
    - useful for neural networks and svm
    - same scaling should be done on training and test data
    - standard scaler: makes the mean 0 and the variance 1 for
      every feature
    - robust scaler: uses median and quartiles to ensure
      statistical properties
        - ignores outliers
    - minmax scaler: shifts the data so all of it is between
      0 and 1
    - normalizer: scales data so all of it is on a sphere of
      radius 1
'''

# # rescaling and shift of data example

# mglearn.plots.plot_scaling()
# plt.show()

# -----

# # minmax scaler with cancer dataset

# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
#                                                     random_state=1)
# print(X_train.shape)
# print(X_test.shape)

# # instantiate scaler
# scaler = MinMaxScaler()

# # fit only uses x data
# scaler.fit(X_train)
# # transform data (scale)
# X_train_scaled = scaler.transform(X_train)
# # print dataset properties before and after scaling
# print("transformed shape: {}".format(X_train_scaled.shape))
# print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
# print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
# print("per-feature minimum after scaling:\n {}".format(
#     X_train_scaled.min(axis=0)))
# print("per-feature maximum after scaling:\n {}".format(
#     X_train_scaled.max(axis=0)))

# # scale test data (uses same transformation as for training)
# X_train_scaled = scaler.transform(X_train)
# # print dataset properties before and after scaling
# print("transformed shape: {}".format(X_train_scaled.shape))
# print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
# print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
# print("per-feature minimum after scaling:\n {}".format(
# X_train_scaled.min(axis=0)))
# print("per-feature maximum after scaling:\n {}".format(
# X_train_scaled.max(axis=0)))

# -----

# look at what happens if test data is trained differently
# make synthetic data
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
# split it into training and test sets
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

# plot the training and test sets
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1],
c=mglearn.cm2(0), label="Training set", s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^',
c=mglearn.cm2(1), label="Test set", s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("Original Data")

# scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# visualize the properly scaled data
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
c=mglearn.cm2(0), label="Training set", s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^',
c=mglearn.cm2(1), label="Test set", s=60)
axes[1].set_title("Scaled Data")

# rescale the test set separately
# so test set min is 0 and test set max is 1
# DO NOT DO THIS! For illustration purposes only.
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)

# visualize wrongly scaled data
axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
c=mglearn.cm2(0), label="training set", s=60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1],
marker='^', c=mglearn.cm2(1), label="test set", s=60)
axes[2].set_title("Improperly Scaled Data")
for ax in axes:
ax.set_xlabel("Feature 0")
ax.set_ylabel("Feature 1")
