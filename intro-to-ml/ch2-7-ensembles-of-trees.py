import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

'''
ensembles combine multiple ml models to create more
powerful models
    - random forests and gradient boosted decision trees

random forests
    - reduce overfitting by averaging the results of trees that
      overfit in different ways
    - requires many trees
    - the trees in the forest can be randomized in two ways:
        - selecting the data points used to build the tree
        - selecting the features in each split test
    - default parameters are usually good, but max_features
      or pre-pruning can also be adjusted
    - also provide feature importances (aggregated feature
      importances from individual trees)
        - usually more reliable than from single trees
    - use n_jobs to specify how many cpu cores to use
        - n_jobs = -1 uses all cores
    - use a fixed random_state for reproducible results
    - don't perform well on very high-dimensional, sparse data
      (like text)
    - take longer and more memory to train than linear models

build a random forest
    - decide how many trees to build (n_estimators parameter,
      usually hundreds or thousands)
    - take a bootstrap sample of the data
        - repeatedly draw random points with replacement as many
          times as there are points (bootstrap sample is the same
          size but is missing approx. one third of the data)
    - select best test using only a subset of the features
        - max_features controls how many features are selected
            - high max_features makes for similar trees
            - low max_features makes for different trees
        - the random selection is repeated at each node
    - make a prediction by generating a prediction for every tree
      in the forest
        - regression: average results to get final prediction
        - classification: soft voting
            - each tree provides a prob ability for each possible
              label
            - the probabilities from every tree are averaged,
              and the class with highest probability is predicted

gradient boosted regression trees (gradient boosting machines)
    - can be used for regression or classification
    - trees are built serially, each one tries to corrects the
      previous one's mistakes
        - the individual simple models are called weak learners
    - no randomization by default, pre-pruning instead
    - usually very shallow trees (depth <= 5)
    - more sensitive to parameters than random forests, but can
      provide better accuracy with good parameters
    - learning_rate adjusts how strongly trees try to correct
      previous mistakes
    - look at xgboost package for large-scale problems
    - require careful parameter tuning and can take a long time
      to train
    - works well without scaling and on a mixture of binary
      and continuous features
    - not good with text
    - higher n_estimators leads to a more complex model
      (possibly overfitting)
'''

# # 5-tree random forest with two_moons dataset

# X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
#                                                     random_state=42)
# forest = RandomForestClassifier(n_estimators=5, random_state=2)
# forest.fit(X_train, y_train)

# # the trees that are built are stored in estimator_
# # plot their decision boundaries

# fig, axes = plt.subplots(2, 3, figsize=(20, 10))
# for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
#     ax.set_title("Tree {}".format(i))
#     mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

# mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],
#                                 alpha=.4)
# axes[-1, -1].set_title("Random Forest")
# mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# plt.show()

# -----

# # 100 trees with breast cancer database

# cancer = load_breast_cancer()

# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, random_state=0)
# forest = RandomForestClassifier(n_estimators=100, random_state=0)
# forest.fit(X_train, y_train)
# print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
# print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

# # look at feature importances

# plot_feature_importances_cancer(forest)
# plt.show()

# -----

# GradientBoosterClassifier on breast cancer database,
# 100 trees, max depth 3, learning rate 0.1 (default)

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# avoid overfitting with pre-pruning (limit max depth)

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# limit learning rate

gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# look at feature importances

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

plot_feature_importances_cancer(gbrt)
plt.show()
