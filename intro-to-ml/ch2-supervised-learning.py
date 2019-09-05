import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

'''
main types of supervised learning:
    - classification: predicts a class label (binary or multiclass)
    - regression: predicts an amount (floating-point/real, is continuous)

overfitting: makes a model too specific, so it cannot generalize well
underfitting: makes a model too general, so it cannot generalize well

more diverse datasets usually allow for more complex models without overfitting
'''

# generate sample dataset
X, y = mglearn.datasets.make_forge()
# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))
