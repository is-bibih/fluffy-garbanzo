import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

'''
binning, discretization, linear models, and trees

interactions and polynomials
    - interaction and polynomial features are good for
      enriching feature representation for lineal models
'''

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
bins = np.linspace(-3, 3, 11)
which_bin = np.digitize(X, bins=bins)
encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
line_binned = encoder.transform(np.digitize(line, bins=bins))

# add X to X_binned (this leads to 11-d dataset)

X_combined = np.hstack([X, X_binned])
print(X_combined.shape)

# build regression model (one coefficient)

reg = LinearRegression().fit(X_combined, y)

line_combined = np.hstack([line, line_binned])
# plt.plot(line, reg.predict(line_combined), label='linear regression combined')

# for bin in bins:
#     plt.plot([bin, bin], [-3, 3], ':', c='k')

# plt.legend(loc="best")
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.show()

# dataset with product/interaction feature (product of bin
# indicator and original feature), it's 20-d

X_product = np.hstack([X_binned, X * X_binned])
print(X_product.shape)

reg = LinearRegression().fit(X_product, y)

line_product = np.hstack((line_binned, line * line_binned))
# plt.plot(line, reg.predict(line_product), label='linear regression product')

# for bin in bins:
#     plt.plot([bin, bin], [-3, 3], ':', c='k')

# plt.plot(X[:, 0], y, 'o', c='k')
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")
# plt.legend(loc="best")
# plt.show()

# ---

# include polynomials up to degree 10 (yields 10 features)
# default include_bias=True adds feature that is always 1

poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)

print("X_poly.shape: {}".format(X_poly.shape))

# compare X_poly and X entries

print("Entries of X:\n{}".format(X[:5]))
print("Entries of X_poly:\n{}".format(X_poly[:5]))

# look at polynomial semantics

print("Polynomial feature names:\n{}".format(poly.get_feature_names()))

# polynomial features with linear regression
# yield polynomial regresssion

reg = LinearRegression().fit(X_poly, y)

line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label='polynomial linear regression')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('Regression output')
plt.xlabel("Input feature")
plt.legend(loc="best")
plt.show()

# look at comparison kernel svm model without dataset transformations

for gamma in [1, 10]:
    svr = SVR(gamma=gamma).fit(X, y)
    plt.plot(line, svr.predict(line), label="SVR gamma={}".format(gamma))

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
plt.show()
