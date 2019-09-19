import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons

'''
clustering
    - partitioning the dataset into groups called clusters
    - points in the same cluster should be similar and points
      in different clusters should be different
    - clustering algorithms assign or predict a number to each
      data point to denote the cluster they belong to
    - the labels found do not necessarily have any meaning,
      they just indicate similar points

k-means clustering
    - tries to find cluster centers that are representative of
      the different regions in the data
    - alternates between two steps:
        - assigning each point to the closest cluster center
        - setting each cluster center as the mean of the data
          points assigned to it
    - it's done when the assignment of instances to clusters
      stops changing
    - predict method can be used to assign cluster labels to new
      points
        - new points do not change the existing model
    - clusters are defined only by their center, so k-means
      might not find desired clusters even if provided with
      the right amount of clusters
    - boundary between clusters is always exactly at the
      middle
    - assumes all directions are equally important for
      each cluster
    - does not identify non-spherical clusters
'''

# # example of k-means clustering on a synthetic dataset
# mglearn.plots.plot_kmeans_algorithm()
# # look at boundaries
# mglearn.plots.plot_kmeans_boundaries()
# plt.show()

# -----

# # k-means with synthetic dataset

# generate synthetic two-dimensional data
# X, y = make_blobs(random_state=1)

# # build the clustering model
# # n_clusters is set to 8 by default (just because)
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)

# # look at which cluster each point belongs to
# print("Cluster memberships:\n{}".format(kmeans.labels_))

# # use predict method to asign new points to clusters
# # it does not change the existing model

# # predict on the training set returns the same result
# # as labels_
# print(kmeans.predict(X))

# # plot data
# # cluster centers are stored in cluster_centers_
# mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
# mglearn.discrete_scatter(
#     kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2],
#     markers='^', markeredgewidth=2)
# plt.show()

# -----

# # k-means with synthetic data: different amounts of clusters

# # generate synthetic two-dimensional data
# X, y = make_blobs(random_state=1)

# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# # using two cluster centers:
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(X)
# assignments = kmeans.labels_

# mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])

# # using five cluster centers:
# kmeans = KMeans(n_clusters=5)
# kmeans.fit(X)
# assignments = kmeans.labels_

# mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])
# plt.show()

# -----

# # k-means with weird blobs

# X_varied, y_varied = make_blobs(n_samples=200,
#                                 cluster_std=[1.0, 2.5, 0.5],
#                                 random_state=170)
# y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varied)

# mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_pred)
# plt.legend(["cluster 0", "cluster 1", "cluster 2"], loc='best')
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

# -----

# # k-means with long blobs

# # generate some random cluster data
# X, y = make_blobs(random_state=170, n_samples=600)
# rng = np.random.RandomState(74)

# # transform the data to be stretched
# transformation = rng.normal(size=(2, 2))
# X = np.dot(X, transformation)

# # cluster the data into three clusters
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)
# y_pred = kmeans.predict(X)

# # plot the cluster assignments and cluster centers
# plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm3)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#             marker='^', c=[0, 1, 2], s=100, linewidth=2, cmap=mglearn.cm3)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

# -----

# k-means with two-moons

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# cluster the data into two clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.predict(X)

# plot the cluster assignments and cluster centers
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm2, s=60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='^', c=[mglearn.cm2(0), mglearn.cm2(1)], s=100, linewidth=2)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
