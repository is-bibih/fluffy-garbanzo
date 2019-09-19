import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import make_moons

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
vector quantization aka k-means decomposition
	- seeing k-means as decomposition
	- each point is represented using a single component
	  (given by the cluster center)
    - reconstruction is the closest cluster center found on
      the training set
    - more clusters than input dimensions can be used to
      encode data
    - data is represented using components that are equal
      in amount to cluster centers
        - all features are 0 except the cluster center
          the point is assigned to
        - more features can be added, which represent the
          distances to each of the cluster centers
    - scales easily to large datasets
    - runs relatively quickly
    - relies on random initialization

scikit-learn implementation
    - by default, runs k-means 10 times with 10 different
      random initializations and returns best result
'''

# # comparing pca, nmf and k-means (components extracted,
# # reconstruction of faces using 50 components)

# # split dataset and train different algorithms
# X_train, X_test, y_train, y_test = train_test_split(
#     X_people, y_people, stratify=y_people, random_state=0)
# nmf = NMF(n_components=50, random_state=0)
# nmf.fit(X_train)
# pca = PCA(n_components=50, random_state=0)
# pca.fit(X_train)
# kmeans = KMeans(n_clusters=50, random_state=0)
# kmeans.fit(X_train)

# # build reconstructions with inverse_transform/cluster_centers_
# X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
# X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]
# X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)

# # plot everything
# fig, axes = plt.subplots(3, 5, figsize=(8, 8),
#                          subplot_kw={'xticks': (), 'yticks': ()})
# fig.suptitle("Extracted Components")
# for ax, comp_kmeans, comp_pca, comp_nmf in zip(
#         axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
#     ax[0].imshow(comp_kmeans.reshape(image_shape))
#     ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')
#     ax[2].imshow(comp_nmf.reshape(image_shape))

# axes[0, 0].set_ylabel("kmeans")
# axes[1, 0].set_ylabel("pca")
# axes[2, 0].set_ylabel("nmf")

# fig, axes = plt.subplots(4, 5, subplot_kw={'xticks': (), 'yticks': ()},
#                          figsize=(8, 8))
# fig.suptitle("Reconstructions")
# for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(
#         axes.T, X_test, X_reconstructed_kmeans, X_reconstructed_pca,
#         X_reconstructed_nmf):
#     ax[0].imshow(orig.reshape(image_shape))
#     ax[1].imshow(rec_kmeans.reshape(image_shape))
#     ax[2].imshow(rec_pca.reshape(image_shape))
#     ax[3].imshow(rec_nmf.reshape(image_shape))

# axes[0, 0].set_ylabel("original")
# axes[1, 0].set_ylabel("kmeans")
# axes[2, 0].set_ylabel("pca")
# axes[3, 0].set_ylabel("nmf")

# plt.show()

# -----

# representing two-moons with more clusters

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=60,
            marker='^', c=range(kmeans.n_clusters), linewidth=2, cmap='Paired')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

print("Cluster memberships:\n{}".format(y_pred))

# add distance features
distance_features = kmeans.transform(X)
print("Distance feature shape: {}".format(distance_features.shape))
print("Distance features:\n{}".format(distance_features))
