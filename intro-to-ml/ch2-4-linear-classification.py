import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs

'''
linear models for classification
    - they all differ in two ways
        - type of regularization
        - loss functions: how they measure how well the coefficients and intercept
          fit the data (usually not very relevant)
    - usually very fast to train and predict
    - work well with sparse data
    - relatively easy to understand how predictions are made
    - not necessarily clear why coefficients turn out like that

binary classification:
    - uses the same formula:
      ŷ = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b > 0
    - gives classes -1 and +1
    - the decision boundary is a linear function of the input
      (uses a line, plane or hyper-plane)

most common classification algorithms are logistic regression
and linear support vector machines (linear svm's)
    - they're implemented with LogisticRegression and with
      LinearSVC (support vector classifier)
    - both use L2 regularization by default (like ridge)
    - they use parameter C (higher C means less regularization)
    - lower C might underfit, higher C might overfit
    - small C means simple model
    - they become more powerful with high dimensions (it becomes
      more important to prevent overfitting)

multiclass classification:
    - can be implemented using one-vs.-rest approach
        - binary model for each class
        - classifier with the highest score wins
    - multiclass logreg can directly be a multiclass classifier
'''

# logisticRegression and linearSVC with forge dataset

# X, y = mglearn.datasets.make_forge()

# fig, axes = plt.subplots(1, 2, figsize=(10, 3))

# for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
#     clf = model.fit(X, y)
#     mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
#                                     ax=ax, alpha=.7)
#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#     ax.set_title("{}".format(clf.__class__.__name__))
#     ax.set_xlabel("Feature 0")
#     ax.set_ylabel("Feature 1")
# axes[0].legend()
# # plt.show()

# # plot with different values of c
# mglearn.plots.plot_linear_svc_regularization()
# plt.show()

# -----

# # LinearLogistic with breast cancer dataset

# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=42)
# logreg = LogisticRegression().fit(X_train, y_train)
# print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
# print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

# # increase C for a more flexible model - has better performance

# logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
# print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
# print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

# # decrease C for more regularized model

# logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
# print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
# print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

# # # look at coefficients

# # plt.plot(logreg.coef_.T, 'o', label="C=1")
# # plt.plot(logreg100.coef_.T, '^', label="C=100")
# # plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
# # plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
# # plt.hlines(0, 0, cancer.data.shape[1])
# # plt.ylim(-5, 5)
# # plt.xlabel("Coefficient index")
# # plt.ylabel("Coefficient magnitude")
# # plt.legend()
# # # plt.show()

# # use l1 regularization (only take into account some features)for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
# for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
#     lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)
#     print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
#         C, lr_l1.score(X_train, y_train)))
#     print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
#         C, lr_l1.score(X_test, y_test)))
#     plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
# plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
# plt.hlines(0, 0, cancer.data.shape[1])
# plt.xlabel("Coefficient index")
# plt.ylabel("Coefficient magnitude")
# plt.ylim(-5, 5)
# plt.legend(loc=3)
# plt.show()

# -----

# one-vs-rest with three-class classification

# sample data from gaussian distribution
X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])
# plt.show()

# train linear svc
linear_svm = LinearSVC().fit(X, y)
print("Coefficient shape: ", linear_svm.coef_.shape)
print("Intercept shape: ", linear_svm.intercept_.shape)
# one row per class, one column (in coefficients) per feature

# see lines given by classifiers
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
            'Line class 2'], loc=(1.01, 0.3))
# plt.show()
# there's a weird rest triangle in the middle, data points get
# classified according to nearest line

# predictions for different regions
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
            'Line class 2'], loc=(1.01, 0.3))
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
# plt.show()
