import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs, make_circles
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

'''
uncertainty estimates from classifiers
    - give probability of false positives or negatives
    - most classifiers have decision_function or predict_proba

decision function
    - has shape (n_samples,) in binary, (n_samples, n_classes)
      in multiclass
    - returns a float for each sample
    - positive values indicate preference for positive class,
      negative values for negative class
        - negative is always first entry in classes_
        - positive is always second entry
    - can be hard to interpret due to arbitrary scale
    - in multiclass, largest value per class wins

predicting probabilities
    - shape (n_samples, 2), (n_samples, n_classes)
      in multiclass
    - first entry in row is estimated probability of first class,
      second entry is prob of second class
    - always between 0 and 1
    - overfitted models tend to make more certain predictions
    - calibrated models have maching certainty and test scores
'''

# look at decision_function and predict_proba in GradientBoostingClassifier

X, y = make_circles(noise=0.25, factor=0.5, random_state=1)

# we rename the classes "blue" and "red" for illustration purposes
y_named = np.array(["blue", "red"])[y]

# we can call train_test_split with arbitrarily many arrays;
# all will be split in a consistent manner
X_train, X_test, y_train_named, y_test_named, y_train, y_test = \
train_test_split(X, y_named, y, random_state=0)

# build the gradient boosting model
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)

# -----

# print("X_test.shape: {}".format(X_test.shape))
# print("Decision function shape: {}".format(
# gbrt.decision_function(X_test).shape))

# # show the first few entries of decision_function
# print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6]))

# # look at prediction based on sign
# print("Thresholded decision function:\n{}".format(
#     gbrt.decision_function(X_test) > 0))
# print("Predictions:\n{}".format(gbrt.predict(X_test)))

# # look at prediction using classes_ (check if they match)
# # make the boolean True/False into 0 and 1
# greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)
# # use 0 and 1 as indices into classes_
# pred = gbrt.classes_[greater_zero]
# # pred is the same as the output of gbrt.predict
# print("pred is equal to predictions: {}".format(
# np.all(pred == gbrt.predict(X_test))))

# # look at min and max values
# decision_function = gbrt.decision_function(X_test)
# print("Decision function minimum: {:.2f} maximum: {:.2f}".format(
# np.min(decision_function), np.max(decision_function)))

# # plot decision_function for all points
# fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4,
#                                 fill=True, cm=mglearn.cm2)
# scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1],
#                                             alpha=.4, cm=mglearn.ReBl)
# for ax in axes:
#     # plot training and test points
#     mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,
#                              markers='^', ax=ax)
#     mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
#                              markers='o', ax=ax)
#     ax.set_xlabel("Feature 0")
#     ax.set_ylabel("Feature 1")
# cbar = plt.colorbar(scores_image, ax=axes.tolist())
# axes[0].legend(["Test class 0", "Test class 1", "Train class 0",
#                 "Train class 1"], ncol=4, loc=(.1, 1.1))
# plt.show()

# -----

# # look at predicting probabilities

# print("Shape of probabilities: {}".format(gbrt.predict_proba(X_test).shape))

# # show the first few entries of predict_proba
# print("Predicted probabilities:\n{}".format(
# gbrt.predict_proba(X_test[:6])))

# # look at decision boundary and class probabilities
# fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# mglearn.tools.plot_2d_separator(
#     gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
# scores_image = mglearn.tools.plot_2d_scores(
#     gbrt, X, ax=axes[1], alpha=.5, cm=mglearn.ReBl, function='predict_proba')

# for ax in axes:
# # plot training and test points
#     mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,
#                              markers='^', ax=ax)
#     mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,
#                              markers='o', ax=ax)
#     ax.set_xlabel("Feature 0")
#     ax.set_ylabel("Feature 1")
# cbar = plt.colorbar(scores_image, ax=axes.tolist())
# axes[0].legend(["Test class 0", "Test class 1", "Train class 0",
#                 "Train class 1"], ncol=4, loc=(.1, 1.1))
# plt.show()

# -----

# uncertainty in multiclass classfication with iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
iris.data, iris.target, random_state=42)
gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)

print("Decision function shape: {}".format(gbrt.decision_function(X_test).shape))
# plot the first few entries of the decision function
print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6, :]))

# look at certainty scores
print("Argmax of decision function:\n{}".format(
np.argmax(gbrt.decision_function(X_test), axis=1)))
print("Predictions:\n{}".format(gbrt.predict(X_test)))

# show the first few entries of predict_proba
print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test)[:6]))
# show that sums across rows are one
print("Sums: {}".format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))

# get predictions from predict_proba
print("Argmax of predicted probabilities:\n{}".format(
    np.argmax(gbrt.predict_proba(X_test), axis=1)))
print("Predictions:\n{}".format(gbrt.predict(X_test)))

# get predictions from predict proba and classes_
# represent each target by its class name in the iris dataset
named_target = iris.target_names[y_train]
logreg.fit(X_train, named_target)
print("unique classes in training data: {}".format(logreg.classes_))
print("predictions: {}".format(logreg.predict(X_test)[:10]))
argmax_dec_func = np.argmax(logreg.decision_function(X_test), axis=1)
print("argmax of decision function: {}".format(argmax_dec_func[:10]))
print("argmax combined with classes_: {}".format(
logreg.classes_[argmax_dec_func][:10]))
