import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

'''
neural networks aka deep learning
    - multilayer perceptrons can be used for more involved
      deep learning methods
    - multilayer perceptrons (mlp's) are aka (vanilla)
      feed-forward neural networks, sometimes just neural networks
    - mlp's are generalizations of linear models that perform
      multiple stages of processing to come to a decision
        - the weighted sums of linear regression is repeated
          multiple times (first computes hidden units ro represent
          an intermediate processing steps)
        - hidden units make up the hidden layer
    - a weighted sum is computed for each hidden unit, and a
      nonlinear function is applied to the result
        - usually the rectifying nonlinearity (aka rectified
          linear unit or relu)
            - cuts off values below zero
        - or the tangens hyperbolicus (tanh)
            - saturates to -1 for low inputs and +1 for high
              inputs
    - the result of the nonlinear function is used in the
      weighted sum that produces the output ŷ
    - a parameter to be set is the number of nodes in the hidden
      layer (like 10 for small/simple datasets and lke 10000
      for very complex data)
    - additional hidden layers can be added
    - can use an L2 penalty to shrink the weights towards 0
      (like in ridge reg and linear classifiers) using alpha
    - weights are set randomly before learning is started (use
      a seed for reproducibility)
    - neural networks expect all input features to vary in a
      similar way, and ideally have a mean 0 and variance of 1
    - introspection can be done by looking at the weights
        - features with low weights might be less important,
          or might not be represented in a way that can be
          used by the neural network
    - can often beat other classifiers and regressors given
      enough time, data, and parameter tuning
    - require careful preprocessing
    - they work best with "homogenous" data (all features
      have similar meanings)
        - tree-based models could work better with
          more heterogeneous datasets

controlling complexity
    - it's usually good to star with one or two hidden layers
    - the amount of nodes per hidden layer is usually
      close to the amount of input features
    - it can work to first create a neural network that
      could overfit, then shrink the network or increase
      alpha
    - different solvers
        - adam usually works well but it is important to
          normalize the data
        - lbfgs is more robust but takes longer on larger
          models/datasets
        - sgd has more parameters
'''

# # look at ŷ (prediction of a linear regressor)
# display(mglearn.plots.plot_logistic_regression_graph())

# display(mglearn.plots.plot_single_hidden_layer_graph())

# # look at tanh and relu
# line = np.linspace(-3, 3, 100)
# plt.plot(line, np.tanh(line), label="tanh")
# plt.plot(line, np.maximum(line, 0), label="relu")
# plt.legend(loc="best")
# plt.xlabel("x")
# plt.ylabel("relu(x), tanh(x)")
# plt.show()

# -----

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=42)

# look at MLPCalssifier with two_moons

# mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
# mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
# mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

# -----

# # look at same one but with only 10 hidden nodes

# mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
# mlp.fit(X_train, y_train)
# mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
# mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

# -----

# # using two hidden layers, with 10 units each

# mlp = MLPClassifier(solver='lbfgs', random_state=0,
# hidden_layer_sizes=[10, 10])
# mlp.fit(X_train, y_train)
# mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
# mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

# -----

# # using two hidden layers, with 10 units each, now with tanh nonlinearity

# mlp = MLPClassifier(solver='lbfgs', activation='tanh',
#                     random_state=0, hidden_layer_sizes=[10, 10])
# mlp.fit(X_train, y_train)
# mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
# mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

# -----

# #  same with different alphas

# fig, axes = plt.subplots(2, 4, figsize=(20, 8))
# for axx, n_hidden_nodes in zip(axes, [10, 100]):
#     for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
#         mlp = MLPClassifier(solver='lbfgs', random_state=0,
#                             hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],
#                             alpha=alpha)
#         mlp.fit(X_train, y_train)
#         mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
#         mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
#         ax.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(
#                       n_hidden_nodes, n_hidden_nodes, alpha))
# plt.show()

# -----

# # different models with same settings

# fig, axes = plt.subplots(2, 4, figsize=(20, 8))
# for i, ax in enumerate(axes.ravel()):
#     mlp = MLPClassifier(solver='lbfgs', random_state=i,
#                         hidden_layer_sizes=[100, 100])
#     mlp.fit(X_train, y_train)
#     mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
#     mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
# plt.show()

# -----

# mlpclassifier on breast cancer database

print("Cancer data per-feature maxima:\n{}".format(cancer.data.max(axis=0)))

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))

# compute the mean value per feature on the training set
mean_on_train = X_train.mean(axis=0)
# compute the standard deviation of each feature on the training set
std_on_train = X_train.std(axis=0)

# subtract the mean, and scale by inverse standard deviation
# afterward, mean=0 and std=1
X_train_scaled = (X_train - mean_on_train) / std_on_train
# use THE SAME transformation (using training mean and std) on the test set
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# increase iterations due to warning from adam (the algorithm)

mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# try alpha = 1 and stronger weight regularization to
# avoid overfitting

mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# introspection with plot
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
plt.show()
