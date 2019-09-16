import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_digits

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
'''

# t-sne on handwritten digits

# example image for each class
fig, axes = plt.subplots(2, 5, figsize=(10, 5),
                         subplot_kw={'xticks':(), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)
plt.show()
