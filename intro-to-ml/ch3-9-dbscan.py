import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

'''
dbscan (density-based spatial clustering of applications
with noise)
    - does not require user to set number of clusters
    - can capture complex shapes
    - can identify points that aren't in any cluster
    - slower than agglomerative clustering and k-means
    - identifies clusters based on the idea that clusters
      form dense regions of data, separated by relatively
      empty regions
    - core samples aka core points: points in a dense region
    - dbscan takes parameters min_samples and eps
        - if there are at least min_samples many data points
          in a distance eps to a given data point, the data
          point is classified as a core sample
        - core samples closer to each other than distance
          eps are part of the same cluster
        - otherwise, the point is labeled as noise
    - it starts with an arbitrary point and checks whether it
      is a core sample
        - if it is a core sample, it is assigned a cluster
          label
        - otherwise it is noise
        - then, all neighbors (within eps) are visited
        - procedure is repeated with other unvisited random
          point when there are no more core samples
          within distance eps of the cluster
    - three tipes of points:
        - core points
        - boundary points: points within distance eps of
          core points
            - could be neighbor to more than one cluster,
              cluster membership depends on visiting order
        - noise: assigned label -1
    - does not allow predictions on new test data

min_samples
    - when it is bigger, fewer points will be core points and
      more points will be labeled noise
    - determines whether points in less dense regions are
      labeled outliers or their woen clusters

eps
    - it determines what it means for points to be "close"
    - very small eps will mean no points are core samples,
      and all points may be noise
    - very large eps will result in all points forming a
      single cluster
    - it may be easier to manually set after using minmaxscaler
      or standardscaler
'''

# # dbscan with synthetic dataset

# X, y = make_blobs(random_state=0, n_samples=12)

# dbscan = DBSCAN()
# clusters = dbscan.fit_predict(X)
# print("Cluster memberships:\n{}".format(clusters))
# # returns -1 for all (noise) because paramenters are
# # not tuned for small toy datasets

# # assignments with different parameter values
# mglearn.plots.plot_dbscan()
# plt.show()

# -----

# dbscan with two-moons

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# rescale the data to zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)

# plot the cluster assignments
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
