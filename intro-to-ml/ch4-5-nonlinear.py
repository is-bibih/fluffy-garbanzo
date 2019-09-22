import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

'''
interactions and polynomials

univariate nonlinear transformations
    - useful for neural networks and linear models (since they
      are very dependent)
    - applying math functions like log, exp, sin
    - they can help adjust the relative scales in data, or
      model periodic data better
    - most models work best when each feature is sorta
      gaussian distributed
        - in regression also the target
    - log helps make the data distribution more symmetrical
      and reduces outliers
    - usually only a subset of the features should be
      transformed
    - sometimes the variable target y should also be
      transformed
        - for counts, log(y + 1) usually helps because it is
          an approximation of poisson regression
'''

# synthetic count dataset

rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)

# print feature frequencies

print("Number of feature appearances:\n{}".format(np.bincount(X[:, 0])))

# look at counts

bins = np.bincount(X[:, 0])
# plt.bar(range(len(bins)), bins, color='k')
# plt.ylabel("Number of appearances")
# plt.xlabel("Value")
# plt.show()

# look at how ridge doesn't work well with gaussian distribution

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print("Test score: {:.3f}".format(score))
# plt.bar(range(len(bins)), bins, color='k')
# plt.ylabel("Number of appearances")
# plt.xlabel("Value")
# plt.show()

# compute log(X + 1) for a better result
# it can't be just log because there's a 0

X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)

# plot more symmetrical data

plt.hist(X_train_log[:, 0], bins=25, color='gray')
plt.ylabel("Number of appearances")
plt.xlabel("Value")
plt.show()

# ridge on new data

score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print("Test score {:.3f}".format(score))
