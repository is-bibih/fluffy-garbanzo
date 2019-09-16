import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()

'''
dimensionality reduction, feature extraction, and
manifold learning
    - transforming data is usually for visualization,
      compression, and finding a more informative
      representation
    - principal component analysis (pca): usually
      used for all that
        - feature extraction is often used for image
          processing
    - non-negative matrix factorization (nmf): used
      for feature extraction
    - t-Distributed Stochastic Neighbor Embedding
      (t-SNE): used for visualization in 2d scatter
      plots
        - builds probability distributions for pairs of
          points so that similar points have a high
          probability of being picked
        - makes a similar distribution in 2d space and
          makes it as close as possible to higher-
          dimension version

principal component analysis
    - rotates the dataset so that rotated features are
      statistically uncorrelated
    - usually only some features are selected afterwards
    - finds the direction of maximum variance, which is the
      one that contains the most information
        - it is the direction along which the features are
          most correlated
    - then it finds the direction with maximum variance
      which is orthogonal to the first direction
        - in 2d space there's only one orthogonal direction
    - these directions are the principal components
        - there are as many principal components as
          original features
    - the mean is subtracted from the dataset so that it is
      centered around 0
    - then the data is rotated to that the principal components
      are aligned with the x and y axes
        - so the transformed dataset has uncorrelated variables
          (it ends up with a vertical or horizontal region)
    - drop components
        - in this case, so the dataset is aligned on a line;
          it becomes one-dimensional
    - undo rotation and add mean back to the data
        - points are back in the original feature space,
          but keep the information only from the first principal
          component
        - it can be used to remove noise or visualize what
          information is retained with the principal components

using pca
    - first instantiate the pca object
        - specify amount of components to be kept
    - then find the principal components by calling the fit method
    - then do rotation and dimensionality reduction with transform
    - plot axes are not usually easy to interpret (since they're
      directions that include many features)
    - principal components are stored in components_
        - each row corresponds to one principal component (first
          is the most important, last is the least important)
        - columns correspond to original features attribute
    - test points can be expressed as a weighted sum of the
      principal components
'''

# # look at pca on synthetics 2d dataset
# mglearn.plots.plot_pca_illustration()
# plt.show()

# -----

# cancer dataset visualization with histograms

# # compute histogram for each of the features for the two classes
# # (benign and malignant)

# fig, axes = plt.subplots(15, 2, figsize=(10, 20))
# malignant = cancer.data[cancer.target == 0]
# benign = cancer.data[cancer.target == 1]

# ax = axes.ravel()

# for i in range(30):
#     _, bins = np.histogram(cancer.data[:, i], bins=50)
#     ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
#     ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
#     ax[i].set_title(cancer.feature_names[i])
#     ax[i].set_yticks(())
# ax[0].set_xlabel("Feature magnitude")
# ax[0].set_ylabel("Frequency")
# ax[0].legend(["malignant", "benign"], loc="best")
# fig.tight_layout()
# plt.show()

# ---

# # scale data with standardscaler

# scaler = StandardScaler()
# scaler.fit(cancer.data)
# X_scaled = scaler.transform(cancer.data)

# # cancer dataset with pca

# # keep the first two principal components of the data
# pca = PCA(n_components=2)
# # fit PCA model to breast cancer data
# pca.fit(X_scaled)

# # transform data onto the first two principal components
# X_pca = pca.transform(X_scaled)
# print("Original shape: {}".format(str(X_scaled.shape)))
# print("Reduced shape: {}".format(str(X_pca.shape)))

# # ---

# # # plot first vs. second principal component, colored by class
# # plt.figure(figsize=(8, 8))
# # mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
# # plt.legend(cancer.target_names, loc="best")
# # plt.gca().set_aspect("equal")
# # plt.xlabel("First principal component")
# # plt.ylabel("Second principal component")
# # plt.show()

# # print("PCA component shape: {}".format(pca.components_.shape))

# # ---

# # look at coefficients using a heat map
# plt.matshow(pca.components_, cmap='viridis')
# plt.yticks([0, 1], ["First component", "Second component"])
# plt.colorbar()
# plt.xticks(range(len(cancer.feature_names)),
#            cancer.feature_names, rotation=60, ha='left')
# plt.xlabel("Feature")
# plt.ylabel("Principal components")
# plt.show()

# -----

# feature extraction with labeled faces in the wild dataset

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

# # look at people
# fix, axes = plt.subplots(2, 5, figsize=(15, 8),
#                          subplot_kw={'xticks': (), 'yticks': ()})
# for target, image, ax in zip(people.target, people.images, axes.ravel()):
#     ax.imshow(image)
#     ax.set_title(people.target_names[target])
# plt.show()

# # look at shapes
# print("people.images.shape: {}".format(people.images.shape))
# print("Number of classes: {}".format(len(people.target_names)))

# # count how many times people show up
# # count how often each target appears
# counts = np.bincount(people.target)
# # print counts next to target names
# for i, (count, name) in enumerate(zip(counts, people.target_names)):
#     print("{0:25} {1:3}".format(name, count), end='    ')
#     if (i + 1) % 3 == 0:
#         print()

# take only up to 50 images of each person so that the
# dataset is less biased
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

# scale the grayscale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability
X_people = X_people / 255.

# use kneighborsclassifier to look at most similar face
# (useful to identify faces, for example)

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0)

# build a KNeighborsClassifier using one neighbor
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test, y_test)))

# use pca because comparing adjacent pixel values
# doesn't work

# example whitening (gives it circle shape)
# mglearn.plots.plot_pca_whitening()
# plt.show()

# use pca to reduce to 100 first principal components
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("X_train_pca.shape: {}".format(X_train_pca.shape))

# classify with knn 1 classifiers
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("Test set accuracy: {:.2f}".format(knn.score(X_test_pca, y_test)))

# look at first principal components
# print("pca.components_.shape:{}".format(pca.components_.shape))
# fix, axes = plt.subplots(3, 5, figsize=(15, 12),
#                          subplot_kw={'xticks': (), 'yticks': ()})
# for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
#     ax.imshow(component.reshape(image_shape),
#               cmap='viridis')
#     ax.set_title("{}. component".format((i + 1)))
# plt.show()

# look at reconstruction with only some features
# (use inverse_transform)
# mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)
# plt.show()

# look at faces along axes of first two principal components
mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()
