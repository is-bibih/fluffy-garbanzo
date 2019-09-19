import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, ward

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

'''
comparing and evaluating clustering algorithms
    - semantic analyses have to be done manually

evaluating clustering with ground truth
    - metrics that assess clustering outcome relative to
      a ground truth clustering
        - adjusted rand index (ari)
        - normalized mutual information (nmi)
    - accuracy_score should not be used because it
      requires labels to match the ground truth exactly
      (what matters is that the points in the same cluster)

evaluating clustering without ground truth
    - more common irl
    - silhouette coefficient: measures compactness of a cluster,
      where higher compactness results in perfect score of 1
        - compact clusters do not allow for complex shapes

robustness-based clustering metrics
    - run an algorithm after adding noise or using different
      parameter settings and compare the outcomes
    - if many algorithm parameters and perturbations of the
      data return the same result, it is likely to be
      trustworthy
    - not implemented in scikit-learn
'''

# # compare k-means, agglomerative clustering and dbscan
# # algorithms with ari

# X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# # rescale the data to zero mean and unit variance
# scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)

# fig, axes = plt.subplots(1, 4, figsize=(15, 3),
#                          subplot_kw={'xticks': (), 'yticks': ()})

# # make a list of algorithms to use
# algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),
#               DBSCAN()]

# # create a random cluster assignment for reference
# random_state = np.random.RandomState(seed=0)
# random_clusters = random_state.randint(low=0, high=2, size=len(X))

# # plot random assignment
# axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters,
#                 cmap=mglearn.cm3, s=60)
# axes[0].set_title("Random assignment - ARI: {:.2f}".format(
#         adjusted_rand_score(y, random_clusters)))

# for ax, algorithm in zip(axes[1:], algorithms):
#     # plot the cluster assignments and cluster centers
#     clusters = algorithm.fit_predict(X_scaled)
#     ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters,
#                cmap=mglearn.cm3, s=60)
#     ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__,
#                                            adjusted_rand_score(y, clusters)))
# plt.show()

# -----

# # compare ari with accuracy_score

# # these two labelings of points correspond to the same clustering
# clusters1 = [0, 0, 1, 1, 0]
# clusters2 = [1, 1, 0, 0, 1]
# # accuracy is zero, as none of the labels are the same
# print("Accuracy: {:.2f}".format(accuracy_score(clusters1, clusters2)))
# # adjusted rand score is 1, as the clustering is exactly the same
# print("ARI: {:.2f}".format(adjusted_rand_score(clusters1, clusters2)))

# -----

# # outcome of k-means, agglomerative clustering and dbscan
# # on two-moons with silhouette score

# X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# # rescale the data to zero mean and unit variance
# scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)

# fig, axes = plt.subplots(1, 4, figsize=(15, 3),
#                          subplot_kw={'xticks': (), 'yticks': ()})

# # create a random cluster assignment for reference
# random_state = np.random.RandomState(seed=0)
# random_clusters = random_state.randint(low=0, high=2, size=len(X))

# # plot random assignment
# axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters,
# cmap=mglearn.cm3, s=60)
# axes[0].set_title("Random assignment: {:.2f}".format(
# silhouette_score(X_scaled, random_clusters)))

# algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),
#               DBSCAN()]

# for ax, algorithm in zip(axes[1:], algorithms):
#     clusters = algorithm.fit_predict(X_scaled)
#     # plot the cluster assignments and cluster centers
#     ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3,
#                s=60)
#     ax.set_title("{} : {:.2f}".format(algorithm.__class__.__name__,
#                                       silhouette_score(X_scaled, clusters)))
# plt.show()

# -----

# comparing algorithms on the faces dataset
# eigenface representation from PCA(whiten=True)

pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit_transform(X_people)
X_pca = pca.transform(X_people)

# # apply DBSCAN with default parameters
# dbscan = DBSCAN()
# labels = dbscan.fit_predict(X_pca)
# print("Unique labels: {}".format(np.unique(labels)))

# # all points are labeled noise so parameters have to be tuned
# # change min_sampless
# dbscan = DBSCAN(min_samples=3)
# labels = dbscan.fit_predict(X_pca)
# print("Unique labels: {}".format(np.unique(labels)))
# # also labeled noise

# # increase eps
# dbscan = DBSCAN(min_samples=3, eps=15)
# labels = dbscan.fit_predict(X_pca)
# print("Unique labels: {}".format(np.unique(labels)))
# # returns only a single cluster

# # Count number of points in all clusters and noise.
# # bincount doesn't allow negative numbers, so we need to add 1.
# # The first number in the result corresponds to noise points.
# print("Number of points per cluster: {}".format(np.bincount(labels + 1)))

# # there's only a few noise points so we can look at them
# noise = X_people[labels==-1]
# fig, axes = plt.subplots(3, 9, subplot_kw={'xticks': (), 'yticks': ()},
#                          figsize=(12, 4))
# for image, ax in zip(noise, axes.ravel()):
#     ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
# plt.show()
# # there's people with stuff in front of their faces,
# # wearing hats, badly cropped, etc.

# # set different smaller eps to get more clusters

# for eps in [1, 3, 5, 7, 9, 11, 13]:
#     print("\neps={}".format(eps))
#     dbscan = DBSCAN(eps=eps, min_samples=3)
#     labels = dbscan.fit_predict(X_pca)
#     print("Clusters present: {}".format(np.unique(labels)))
#     print("Cluster sizes: {}".format(np.bincount(labels + 1)))

# eps=7 has a lot of small clusters, so we can look at it
dbscan = DBSCAN(min_samples=3, eps=7)
labels = dbscan.fit_predict(X_pca)

# for cluster in range(max(labels) + 1):
#     mask = labels == cluster
#     n_images = np.sum(mask)
#     fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4),
#                              subplot_kw={'xticks': (), 'yticks': ()})
#     for image, label, ax in zip(X_people[mask], y_people[mask], axes):
#         ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
#         ax.set_title(people.target_names[label].split()[-1])
# plt.show()

# kmeans: start with 10 clusters because it probably won't
# find one for everyone

# extract clusters with k-means
km = KMeans(n_clusters=10, random_state=0)
labels_km = km.fit_predict(X_pca)
# print("Cluster sizes k-means: {}".format(np.bincount(labels_km)))

# # look at cluster centers (use pca.inverse_transform first)
# fig, axes = plt.subplots(2, 5, subplot_kw={'xticks': (), 'yticks': ()},
#                          figsize=(12, 4))
# for center, ax in zip(km.cluster_centers_, axes.ravel()):
#     ax.imshow(pca.inverse_transform(center).reshape(image_shape),
#               vmin=0, vmax=1)
# plt.show()

# look at most typical and most atypical images (closest and
# furthest from the centers) in each cluster
# mglearn.plots.plot_kmeans_faces(km, pca, X_pca, X_people,
#                                 y_people, people.target_names)
# plt.show()

# agglomerative clustering

# extract clusters with ward agglomerative clustering
agglomerative = AgglomerativeClustering(n_clusters=10)
labels_agg = agglomerative.fit_predict(X_pca)
print("Cluster sizes agglomerative clustering: {}".format(
    np.bincount(labels_agg)))

# use ari to check if agglomerative clustering and kmeans
# yield similar results

print("ARI: {:.2f}".format(adjusted_rand_score(labels_agg, labels_km)))
# low ari of 0.09 means they don't have much in common

# use a dendrogram to look at the groups (limited depth)
linkage_array = ward(X_pca)
# now we plot the dendrogram for the linkage_array
# containing the distances between clusters
# plt.figure(figsize=(20, 5))
# dendrogram(linkage_array, p=7, truncate_mode='level', no_labels=True)
# plt.xlabel("Sample index")
# plt.ylabel("Cluster distance")
# plt.show()

# look at the top 10 clusters

# n_clusters = 10
# for cluster in range(n_clusters):
#     mask = labels_agg == cluster
#     fig, axes = plt.subplots(1, 10, subplot_kw={'xticks': (), 'yticks': ()},
#                              figsize=(15, 8))
#     axes[0].set_ylabel(np.sum(mask))
#     for image, label, asdf, ax in zip(X_people[mask], y_people[mask],
#                                       labels_agg[mask], axes):
#         ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
#         ax.set_title(people.target_names[label].split()[-1],
#                      fontdict={'fontsize': 9})
# plt.show()

# they're not very homogeneous so let's try with 40 clusters
agglomerative = AgglomerativeClustering(n_clusters=40)
labels_agg = agglomerative.fit_predict(X_pca)
print("cluster sizes agglomerative clustering: {}".format(np.bincount(labels_agg)))

n_clusters = 40
for cluster in [10, 13, 19, 22, 36]: # hand-picked "interesting" clusters
    mask = labels_agg == cluster
    fig, axes = plt.subplots(1, 15, subplot_kw={'xticks': (), 'yticks': ()},
                             figsize=(15, 8))
    cluster_size = np.sum(mask)
    axes[0].set_ylabel("#{}: {}".format(cluster, cluster_size))
    for image, label, asdf, ax in zip(X_people[mask], y_people[mask],
                                      labels_agg[mask], axes):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
        ax.set_title(people.target_names[label].split()[-1],
                     fontdict={'fontsize': 9})
    for i in range(cluster_size, 15):
        axes[i].set_visible(False)
plt.show()
