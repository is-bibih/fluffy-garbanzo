import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

'''
linear models
    - based on a linear function of input features
    - general prediction formula:
      ŷ = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b
        - ŷ is the output
        - x denotes features (total of p features)
        - w and b are learned paramenters
          (w are weights for the features, aka coefficients)
    - single feature would be ŷ = w[0] * x[0] + b
        - like the equation for a line
    - produces a line for one feature, a plane for 2 features,
      a hyperplane in more dimensions for more features
    - usually very good with datasets with many features
      (higher-dimensional) datasets
    - there is greater risk of overfitting with higher-
      dimensional datasets

linear regression (ordinary least squares)
    - finds w and b that minimize the mean squared error between
      predictions and true targets
    - mean squared error is the sum of the squared differences
      between predictions and true values
    - it has no parameters (cool, but complexity cannot be adjusted)

ridge regression
    - same as OLS but coefficients must be as close to
      0 as possible
    - this is an example of regularization (restricting a model
      to avoid overfitting)
    - ridge regression uses L2 regularization

learning curves: plots that show model performance as a function
of dataset size

lasso
    - uses L1 regularization
    - makes some coefficients be 0 (some features are ignored
    - usually used when only some features are important
    - provides a model that is easier to interpret

scikit-learn has an ElasticNet, which combines lasso and ridge
(usually works best)
'''


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# # learn w[0] and b for wave dataset
# mglearn.plots.plot_linear_regression_wave()
# plt.show()

# ------

# # wave dataset OLS

# X, y = mglearn.datasets.make_wave(n_samples=60)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# lr = LinearRegression().fit(X_train, y_train)

# # coefficients (w) are stored in coef_ (numpy array)
# # intercept (b) is stored in intercept_ (float number)

# print("lr.coef_: {}".format(lr.coef_))
# print("lr.intercept_: {}".format(lr.intercept_))

# print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# ------

# # boston dataset OLS - overfits because it has more features

# lr = LinearRegression().fit(X_train, y_train)
# print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# ------

# # boston dataset rr - generalizes better because it is less complex

# ridge = Ridge().fit(X_train, y_train)
# print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

# # Ridge has a parameter alpha (default is 1.0) that defines
# # simplicity vs. training set performance.
# # Increasing alpha decreases coefficients (simpler model).

# ridge10 = Ridge(alpha=10).fit(X_train, y_train)
# print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

# # small alpha resembles linear regression

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
# print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))

# # look at coefficients for different alphas (needs lr to work)

# plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
# plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
# plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
# plt.plot(lr.coef_, 'o', label="LinearRegression")
# plt.xlabel("Coefficient index")
# plt.ylabel("Coefficient magnitude")
# plt.hlines(0, 0, len(lr.coef_))
# plt.ylim(-25, 25)
# plt.legend()
# plt.show()

# -----

# # show learning curve for ridge and lr
# # test scores for ridge are better because it is regularized
# # regularization is less important with bigger datasets 
# # lr catches up with ridge at the end with bigger datasets
# mglearn.plots.plot_ridge_n_samples()
# plt.show()

# -----

# lasso with extended boston - underfits with alpha=1
lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

# lasso with smaller alpha to avoid underfitting (also
# needs to increase iterations)
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))

# lasso with even smaller alpha overfits (like lr)
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))

# plot coefficients with different alphas (needs ridge01)
plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.show()