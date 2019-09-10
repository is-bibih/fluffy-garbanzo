import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D, axes3d
from sklearn.svm import SVC

'''
kernelized support vector machines
    - allow for more complex models (not defined just by
      hyperplanes)
    - look at math in ch 1 of the elements of statistical learning
      (hastie, tibshirani and friedman)
    - adds polynomials (interactions) of the input features 92
    - support vectors: training points on the border between
      classes
    - makes a classification based on te distances to support
      vectors and the learned importance of the support vectors
    - distance between data points:
        - k_rbf(x_1, x_2) = exp (ɣǁx_1 - x_2ǁ^2)
            - ɣ is a parameter that controls width of the kernel
            - ǁx_1 - x_2ǁ is euclidean distance
    - parameters
        - gamma: determines what is considered "close" between
          points
            - small gamma means large radius
            - lower gamma means simpler model
        - C: regularization parameter (like for lineal models)
          that limits the importance of each points
            - smaller c means simpler model
            - higher c makes model adjust more to outliers

the kernel trick
    - computes the distance (dot/scalar product) of the data
      points for the expanded feature representation

polynomial kernel
    - computes all possible polynomials up to a certain degree
      of the original features

radial basis function (rbf) aka gaussian
    - corresponds to infinite-dimensional feature space
'''

# # weird dataset with blobs (linear model would do badly)

# X, y = make_blobs(centers=4, random_state=8)
# y = y % 2 #why

# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# # plt.xlabel("Feature 0")
# # plt.ylabel("Feature 1")
# # plt.show()

# linear_svm = LinearSVC().fit(X, y)

# mglearn.plots.plot_2d_separator(linear_svm, X)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# # plt.xlabel("Feature 0")
# # plt.ylabel("Feature 1")
# # plt.show()

# # add extra feature: feature1 ** 2
# # points become 3D: instead of being (feature0, feature1),
# #   they become (feature0, feature1, feature1 ** 2)

# # add the squared first feature
# X_new = np.hstack([X, X[:, 1:] ** 2])

# figure = plt.figure()
# # visualize in 3D
# ax = Axes3D(figure, elev=-152, azim=-26)
# # plot first all the points with y == 0, then all with y == 1
# mask = y == 0
# # ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
# #            cmap=mglearn.cm2, s=60)
# # ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
# #            cmap=mglearn.cm2, s=60)
# # ax.set_xlabel("feature0")
# # ax.set_ylabel("feature1")
# # ax.set_zlabel("feature1 ** 2")
# # plt.show()

# # use a plane to divide the classes

# linear_svm_3d = LinearSVC().fit(X_new, y)
# coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

# # show linear decision boundary
# figure = plt.figure()
# # ax = Axes3D(figure, elev=-152, azim=-26)
# xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
# yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)

# XX, YY = np.meshgrid(xx, yy)
# ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
# # ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
# # ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
# #            cmap=mglearn.cm2, s=60)
# # ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
# #            cmap=mglearn.cm2, s=60)

# # ax.set_xlabel("feature0")
# # ax.set_ylabel("feature1")
# # ax.set_zlabel("feature0 ** 2")
# # plt.show()

# ZZ = YY ** 2
# dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
# plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
#              cmap=mglearn.cm2, alpha=0.5)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

# -----

# train an svm on forge

X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plot support vectors
sv = svm.support_vectors_
# class labels of support vectors are given by the sign of the dual coefficients
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
# plt.show()

# change parameters

fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)

axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"],
                  ncol=4, loc=(.9, 1.2))
plt.show()
