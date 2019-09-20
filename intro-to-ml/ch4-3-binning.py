import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder

'''
binning, discretization, linear models, and trees

binning aka discretization
    - splitting features into multiple features (on
      cotinuous data)
    - each bin is defined by a range in the input data
    - data points are represented by the bin they fall into
    - usually not necessary on trees (since they already
      split data and they look at multiple features at once)
    - benefits large and high-dimensional data with features
      with nonlinear relations with the output
'''

# compare linear regression and decision tree regression

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

# reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
# plt.plot(line, reg.predict(line), label="decision tree")

# reg = LinearRegression().fit(X, y)
# plt.plot(line, reg.predict(line), label="linear regression")

# plt.plot(X[:, 0], y, 'o', c='k')
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")
# plt.legend(loc="best")
# plt.show()

# ---

# make 10 bins (between -3 and 3 because the input is in
# that range)
# there are eleven boundaries, so 10 spaces between them

bins = np.linspace(-3, 3, 11)
print("bins: {}".format(bins))

# look at bin membership
# can be done with np.digitize

which_bin = np.digitize(X, bins=bins)
print("\nData points:\n", X[:5])
print("\nBin membership for data points:\n", which_bin[:5])

# use one-hot encoding on data

# transform using the OneHotEncoder
encoder = OneHotEncoder(sparse=False)
# encoder.fit finds the unique values that appear in which_bin
encoder.fit(which_bin)
# transform creates the one-hot encoding
X_binned = encoder.transform(which_bin)
print(X_binned[:5])
# look at new shape
print("X_binned.shape: {}".format(X_binned.shape))

# ---

# new linear regression and decision tree

line_binned = encoder.transform(np.digitize(line, bins=bins))

reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='linear regression binned')

reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='decision tree binned')

plt.plot(X[:, 0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc="best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.show()
