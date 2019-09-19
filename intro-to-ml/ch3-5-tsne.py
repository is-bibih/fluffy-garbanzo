import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

digits = load_digits()

'''
manifold learning algorithms
    - allow for complex mapping (not like pca)
    - often provide better visualizations
    - aimed mainly at visualization
    - usually used for exploratory data analysis but usually not
      used if the goas is supervised learning

t-SNE
    - transform training set but cannot be applied to a test set
    - looks for a 2d representation of the dataset that
      preserves distances between points as well as possible
    - starts with random 2d representation of the points, then
      tries to get neighbors close together and distant points
      further apart
        - there is not as much emphasis on getting points far
          away, it is more important to group close points
    - it has some tuning parameters but usually works well with
      defaults
        - like perplexity and early_exaggeration
'''

# t-sne on handwritten digits

# # example image for each class
# fig, axes = plt.subplots(2, 5, figsize=(10, 5),
#                          subplot_kw={'xticks':(), 'yticks': ()})
# for ax, img in zip(axes.ravel(), digits.images):
#     ax.imshow(img)
# plt.show()

# ---

# use pca to visualize data reduced to 2d

# build a PCA model
pca = PCA(n_components=2)
pca.fit(digits.data)

# transform the digits data onto the first two principal components
digits_pca = pca.transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
# plt.figure(figsize=(10, 10))
# plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
# plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
# for i in range(len(digits.data)):
#     # actually plot the digits as text instead of using scatter
#     plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),
#              color = colors[digits.target[i]],
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xlabel("First principal component")
# plt.ylabel("Second principal component")
# plt.show()

# ---

# use t-sne

tsne = TSNE(random_state=42)
# use fit_transform instead of fit, as TSNE has no transform method
digits_tsne = tsne.fit_transform(digits.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("t-SNE feature 0")
plt.xlabel("t-SNE feature 1")
plt.show()
