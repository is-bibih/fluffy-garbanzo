import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, ward

'''
agglomerative clustering
    - refers to several clustering algorithms with
      same principles
    - each point starts as its own cluster, then the two most
      similar clusters are merged until some criterion is
      satisfied
    - linkage criteria determine how the most similar cluster
      is measured
        - ward (by default): picks the two clusters that
          increase variance in each cluster the least
          when merged, usually leads to equally sized clusters
            - works with most datasets
        - average: merges two clusters with smallest average
          distance bewteen all their points
        - complete aka maximum: mergest two clusters with
          smallest maximum distance bewteen points
            - works better when clusters have very dissimilar
              numbers of members
    - does not make predictions
    - doesn't require the amount of clusters to be specified
    - still fails with complex shapes like two_moons

hierarchical clustering
    - every point goes from being a single point cluster to
      belonging to some final cluster
    - can be visualized using a dendrogram
        - implemented in scipy
'''

# # agglomerative clustering process on a 2d dataset (3 clusters)

# mglearn.plots.plot_agglomerative_algorithm()
# plt.show()

# -----

# # look at how clustering works

# X, y = make_blobs(random_state=1)

# agg = AgglomerativeClustering(n_clusters=3)
# assignment = agg.fit_predict(X)

# mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

# -----

# # look at how each cluster breaks up into smaller clusters
# mglearn.plots.plot_agglomerative()
# plt.show()

# -----

# look at dendrogram
X, y = make_blobs(random_state=0, n_samples=12)
# Apply the ward clustering to the data array X
# The SciPy ward function returns an array that specifies the distances
# bridged when performing agglomerative clustering
linkage_array = ward(X)

# Now we plot the dendrogram for the linkage_array containing the distances
# between clusters
dendrogram(linkage_array)

# Mark the cuts in the tree that signify two or three clusters
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, ' two clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")
plt.show()
