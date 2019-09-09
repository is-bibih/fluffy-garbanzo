import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.linear_model import LinearRegression

'''
decision trees
    - they learn a hierarchy of if/else questions that lead to
      a decision most quickly
    - the questions are called tests
        - on continuous data the tests are of the form
          "is feature i larger than value a"
    - to build a tree, the algorithm searches over all possible
      tests and finds the one that is most informative
        - this can be repeated recursively to build a binary tree
          until each leaf contains only a single target value
          (single class or regression value)
        - leaves in the completed tree are "pure", all their
          points belong to the same target value (class)
    - top node aka root represents the whole dataset
    - each node has the class counts and may have a test
      (if it is not a leaf)
    - predictions are done by checking the region the new data
      point is in, and assigning it the majority target (or the
      target if the leaves are pure)
    - it probably overfits if all the leaves are pure
    - the depth is how many consecutive questions can be
      asked
    - trees cannot extrapolate (make predictions outside the range
      of training data)
    - have two advantages over many other methods:
        - (smaller) trees can easily be visualized and understood
        - the scaling of the data does not affect the algorithm
            - works especially well when the features are on
              completely different scales or are a mix of
              binary and continuous features

preventing overfitting
    - pre-pruning: stops creation of the tree early
        - limiting maximum depth, limiting max amount of leaves,
          min number of points in node to keep splitting
    - post-pruning aka pruning: removing or collapsing nodes
      with little information
    - scikit-learn does not implement pre-pruning

feature importance
    - rates how important each feature is for the decision
      a tree makes
    - 0 is not used at all
    - 1 is predicts perfectly
    - feature importances always sum up to 1
'''

# # example to distinguis bears, hawks, penguins, dolphins
# mglearn.plots.plot_animal_tree()
# plt.show()
# # each node represents a question
# # each terminal node (leaf) contains the answer

# -----

# # unpruned tree

# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=42)
# # random_state is used for tie-breaking internally
# tree = DecisionTreeClassifier(random_state=0)
# tree.fit(X_train, y_train)
# print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
# print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# # pre-pruned tree

# tree = DecisionTreeClassifier(max_depth=4, random_state=0)
# tree.fit(X_train, y_train)
# print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
# print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# # make dot for the tree
# export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
#                 feature_names=cancer.feature_names, impurity=False, filled=True)

# # look at feature importances
# print("Feature importances:\n{}".format(tree.feature_importances_))

# # look at feature importances but pretty
# def plot_feature_importances_cancer(model):
#     n_features = cancer.data.shape[1]
#     plt.barh(range(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), cancer.feature_names)
#     plt.xlabel("Feature importance")
#     plt.ylabel("Feature")
#     plt.show()

# plot_feature_importances_cancer(tree)

# -----

# # look at example with weird relationship between features and class
# tree = mglearn.plots.plot_tree_not_monotone()
# plt.show()

# -----

# RAM prices dataset

ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))

# plt.semilogy(ram_prices.date, ram_prices.price)
# plt.xlabel("Year")
# plt.ylabel("Price in $/Mbyte")
# plt.show()

# use historical data to forecast prices after the year 2000
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

# predict prices based on date
X_train = data_train.date[:, np.newaxis]
# we use a log-transform to get a simpler relationship of data to target
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

# predict on all data
X_all = ram_prices.date[:, np.newaxis]
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

# undo log-transform
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(data_train.date, data_train.price, label="Training data")
plt.semilogy(data_test.date, data_test.price, label="Test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
plt.legend()
plt.show()
