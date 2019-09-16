import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA

# get lfw dataset and image shape
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

# reduce bias in dataset by limiting images per person to 50
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

# scale grayscale values to between 0 and 1 for better
# numeric stability
X_people = X_people / 255.

# split training set
X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0)

'''
non-negative matrix factorization (nmf)
    - similar to pca, can also be used for dimensionality
      reduction
    - in pca we write each data point as a weighted sum of
      some components, in nmf we want the components and the
      coefficients to be non-negative
    - only works on data with non-negative features
    - particularly helpful with data which is created as the
      addition (or overlay) of several independent sources
        - like people speaking or music with different
          instruments
        - like gene expression
        - like text data
        - nmf can identify original components
    - usually more interpretable components than pca
        - all are equally important
    - reconstruction is usually worse with nmf than with pca
      (pca gets most informative components, nmf finds
      patterns in the data)

nmf implementation
    - based on a random seed
    - main parameter is the amount of components to extract
        - usually lower than the amount of input features
          (otherwise each data point would be represented by
          a component)

** look at decomposition methods on scikit_learn **
'''

# # nmf with synthetic data
# mglearn.plots.plot_nmf_illustration()
# plt.show()

# -----

# nmf with labeled faces in the wild

# # look at how the amount of components affects representation
# # DON'T RUN IT LOOKS FOR 500 COMPONENTS
# mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)
# plt.show()

# ---

# # extract 15 components with nmf

# nmf = NMF(n_components=15, random_state=0)
# nmf.fit(X_train)
# X_train_nmf = nmf.transform(X_train)
# X_test_nmf = nmf.transform(X_test)

# fix, axes = plt.subplots(3, 5, figsize=(15, 12),
#                          subplot_kw={'xticks': (), 'yticks': ()})
# for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
#     ax.imshow(component.reshape(image_shape))
#     ax.set_title("{}. component".format(i))

# plt.show()

# ---

# # find images with especially strong 3 and 7 components

# nmf = NMF(n_components=15, random_state=0)
# nmf.fit(X_train)
# X_train_nmf = nmf.transform(X_train)
# X_test_nmf = nmf.transform(X_test)

# compn = 3
# # sort by 3rd component, plot first 10 images
# inds = np.argsort(X_train_nmf[:, compn])[::-1]
# fig, axes = plt.subplots(2, 5, figsize=(15, 8),
#                          subplot_kw={'xticks': (), 'yticks': ()})
# for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
#     ax.imshow(X_train[ind].reshape(image_shape))

# compn = 7
# # sort by 7th component, plot first 10 images
# inds = np.argsort(X_train_nmf[:, compn])[::-1]
# fig, axes = plt.subplots(2, 5, figsize=(15, 8),
#                          subplot_kw={'xticks': (), 'yticks': ()})
# for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
#     ax.imshow(X_train[ind].reshape(image_shape))

# plt.show()

# -----

# nmf with 3-source signal combination

# look at signals
S = mglearn.datasets.make_signals()
# plt.figure(figsize=(6, 1))
# plt.plot(S, '-')
# plt.xlabel("Time")
# plt.ylabel("Signal")
# plt.show()

# mix data into a 100-dimensional state
# (it's like having 100 measurement devices for the signals)
A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print("Shape of measurements: {}".format(X.shape))

nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)
print("Recovered signal shape: {}".format(S_.shape))

# get at pca for comparison
pca = PCA(n_components=3)
H = pca.fit_transform(X)

# look at comparison with both
models = [X, S, S_, H]
names = ['Observations (first three measurements)',
         'True sources',
         'NMF recovered signals',
         'PCA recovered signals']

fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .5},
                         subplot_kw={'xticks': (), 'yticks': ()})

for model, name, ax in zip(models, names, axes):
    ax.set_title(name)
    ax.plot(model[:, :3], '-')

plt.show()
