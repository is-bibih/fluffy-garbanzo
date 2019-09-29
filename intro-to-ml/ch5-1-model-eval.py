import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()

'''
model evaluation and improvement
    - the dataset is split into training and test to see how
      well it generalizes

cross-validation
    - robust way to assess generalization performance (in
      comparison to score method)
    - the data is split repeatedly and multiple models are
      trained
    - most common version of cross-validation is k-fold
      cross-validation (k is usually between 5 and 10)
    - data is partitioned into k parts of approximately equal
      size, called folds
    - the first model is tested on the first set and trained
      on the rest
    - the second model is tested on the second set and trained
      on the rest
    - and so on
    - the accuracy is computed for each of the models
    - the mean accuracy is commonly used to summarize
    - helps provide some information about how sensitive the model
      is to the selection of the training dataset
    - it allows more accurate models to be built, since in each
      iteration, (k-1)/k of the dataset is used for training,
      while train_test_split usually uses 75%
    - main disadvantage is computational cost (cross-val is about
      k times slower than a single split)

cross-validation in scikit-learn
    - with cross_val_score from model_selection
    - parameters are model to be evaluated, training data and
      ground-truth labels
    - uses k=3 by default (in cv parameter)

grid search
    - method to adjust paramenters in supervised models to
      improve performance
'''

# example of the process done so far (split datset, build
# model and evaluate it)

# create a synthetic dataset
X, y = make_blobs(random_state=0)
# split dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# instantiate and fit model
logreg = LogisticRegression().fit(X_train, y_train)
# evaluate model
print("Test set score: {:.2f}\n\n".format(logreg.score(X_test, y_test)))

# -----

# illustration of data splits in five-fold cross-validation
mglearn.plots.plot_cross_validation()
# plt.show()

# -----

# evaluate logreg on iris dataset

logreg = LogisticRegression()
scores = cross_val_score(logreg, iris.data, iris.target)
print("Cross-validation scores: {}\n".format(scores))

# with k=5
scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("Cross-validation scores: {}".format(scores))
# mean accuracy
print("average cross-validation score: {:.2f}".format(scores.mean()))
