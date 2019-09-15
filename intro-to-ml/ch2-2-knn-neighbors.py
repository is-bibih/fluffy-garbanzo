import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsRegressor

'''
k-nearest neighbors (k-nn)
    - simplest ml algorithm
    - makes prediction based on nearest neighbors
    - uses voting for k > 1 in classification (picks majority class)
    - calculates mean for k > 1 in regression
    - greater k results in simpler model (more general)
    - main parameters are number of neighbors and how distance is measured
        - it usually works well to choose 3-5
        - euclidean distance is used by default (usually works)
    - building is usually fast, but prediction can be slow with
      large training datasets
    - data should be preprocessed
    - it usually does not do well with sparse datasets (most features are 0)
    - it usually does not do well on datasets with many features (100+)
decision boundary: the divide between where algorithm assigns different classes
'''

# k-nn with forge
# mglearn.plots.plot_knn_classification(n_neighbors=1)
# mglearn.plots.plot_knn_classification(n_neighbors=3)
# plt.show()

# # split data to train and test
# X, y = mglearn.datasets.make_forge()
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# # instantiate class to set parameters
# clf = KNeighborsClassifier(n_neighbors=3)
# # fit with dataset
# clf.fit(X_train, y_train)
# # predict
# print("Test set predictions: {}".format(clf.predict(X_test)))
# # evaluate model with score method
# print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# # visualizations for code boundaries with k = 1, 3, 9
# fig, axes = plt.subplots(1, 3, figsize=(10, 3))
# print(axes)
# for n_neighbors, ax in zip([1, 3, 9], axes):
#     # the fit method returns the object self, so we can instantiate
#     # and fit in one line
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
#     mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#     ax.set_title("{} neighbor(s)".format(n_neighbors))
#     ax.set_xlabel("feature 0")
#     ax.set_ylabel("feature 1")
# axes[0].legend(loc=3)
# # plt.show()

# ---------

# # test with different number of neighbors
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=66)
# training_accuracy = []
# test_accuracy = []
# # try n_neighbors from 1 to 10
# neighbors_settings = range(1, 11)

# for n_neighbors in neighbors_settings:
#     # build the model
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors)
#     clf.fit(X_train, y_train)
#     # record training set accuracy
#     training_accuracy.append(clf.score(X_train, y_train))
#     # record generalization accuracy
#     test_accuracy.append(clf.score(X_test, y_test))

# # plot results
# plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
# plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("n_neighbors")
# plt.legend()
# plt.show()

# ---------

# k-nn regression (using wave dataset, KNeighborsRegressor)
# mglearn.plots.plot_knn_regression(n_neighbors=1)
# mglearn.plots.plot_knn_regression(n_neighbors=3)
# plt.show()
X, y = mglearn.datasets.make_wave(n_samples=40)
# split the wave dataset into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# instantiate the model and set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)
# fit the model using the training data and training targets
reg.fit(X_train, y_train)
# test
print("Test set predictions:\n{}".format(reg.predict(X_test)))
# evaluate with score (returns coefficient of determination R^2)
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))

# look at predictions using a test dataset (many points on the line)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# 1000 data points between -3 and 3
# reshape -1 means it adjusts to data
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
# predict with 1, 3 or 9 neighbors
for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target",
                "Test data/target"], loc="best")
# plt.show()
