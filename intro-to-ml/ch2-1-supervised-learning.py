import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

'''
main types of supervised learning:
    - classification: predicts a class label (binary or multiclass)
    - regression: predicts an amount (floating-point/real, is continuous)

overfitting: makes a model too specific, so it cannot generalize well
underfitting: makes a model too general, so it cannot generalize well

more diverse datasets usually allow for more complex models without overfitting
'''

# generate sample dataset forge (classification)
X1, y1 = mglearn.datasets.make_forge()
# plot dataset
mglearn.discrete_scatter(X1[:, 0], X1[:, 1], y1)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X1.shape: {}".format(X1.shape))
# plt.show()

print()

# generate sample dataset wave (regression)
X2, y2 = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X2, y2, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
# plt.show()

# breast cancer dataset (binary classification)
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
print("Shape of cancer data: {}".format(cancer.data.shape))
print("Sample counts per class:\n{}".format(
      {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("Feature names:\n{}".format(cancer.feature_names))

print()

# boston housing dataset (regression)
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))
# expanded dataset includes interactions (products) between features
X3, y3 = mglearn.datasets.load_extended_boston()
print("X3.shape: {}".format(X3.shape))
