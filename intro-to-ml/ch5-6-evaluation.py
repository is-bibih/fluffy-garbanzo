import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

'''
evaluation metrics and scoring
    - up to now:
        - accuracy (fraction of correctly classified samples) for
          classifiers
        - R^2 for regression
    - it is important to keep the high-level goal of the application
      in mind, aka the business metric
        - the model choice should have the most positive effect
    - also the consequences of choosing a particular algorithm for ml
      is called the business impact

metrics for binary classification
    - kinds of errors
        - type I error: false positive
        - type II error: false negative
    - imbalanced datasets: with a class which is a lot more frequent
      than the other
        - a model could always predict false for a 99% false dataset
          and get 99% accuracy
        - if a model gets only slightly better accuracy, it could
          indicate that there is something wrong with how the model
          is being used, or that accuracy is not a good measure

confusion matrices
    - can be used to represent the results of evaluating binary
      classification
    - the rows represent true classes
    - the columns represent the class predictions made by the model
    - entries on the main diagonal correspont to correct
      clasifications (true positives and negatives)
'''

digits = load_digits()
y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, y, random_state=0)

def imbalanced(X_train, X_test, y_train, y_test):
    # look at how dummy classifier gets great score
    dummy_majority = DummyClassifier(
        strategy='most_frequent').fit(X_train, y_train)
    pred_most_frequent = dummy_majority.predict(X_test)
    print("Unique predicted labels: {}".format(np.unique(pred_most_frequent)))
    print("Test score: {:.2f}".format(dummy_majority.score(X_test, y_test)))

    return pred_most_frequent

def decision_tree(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
    pred_tree = tree.predict(X_test)
    print("Test score: {:.2f}".format(tree.score(X_test, y_test)))

    return pred_tree

def other_dummy(X_train, X_test, y_train, y_test):
    dummy = DummyClassifier().fit(X_train, y_train)
    pred_dummy = dummy.predict(X_test)
    print("dummy score: {:.2f}".format(dummy.score(X_test, y_test)))

    logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
    pred_logreg = logreg.predict(X_test)
    print("logreg score: {:.2f}".format(logreg.score(X_test, y_test)))

    return pred_dummy, pred_logreg

def confuse(y_test, pred_logreg, show=False):
    confusion = confusion_matrix(y_test, pred_logreg)
    print("Confusion matrix:\n{}".format(confusion))

    if show:
        # show illustration
        mglearn.plots.plot_confusion_matrix_illustration()
        plt.show()
        # show confusion matrix meaning
        mglearn.plots.plot_binary_confusion_matrix()
        plt.show()

def compare(y_test, pred_most_frequent, pred_dummy, pred_tree,
            pred_logreg):
    print("Most frequent class:")
    print(confusion_matrix(y_test, pred_most_frequent))
    print("\nDummy model:")
    print(confusion_matrix(y_test, pred_dummy))
    print("\nDecision tree:")
    print(confusion_matrix(y_test, pred_tree))
    print("\nLogReg:")
    print(confusion_matrix(y_test, pred_logreg))

pred_most_frequent = imbalanced(X_train, X_test, y_train, y_test)
pred_tree = decision_tree(X_train, X_test, y_train, y_test)
pred_dummy, pred_logreg = other_dummy(X_train, X_test, y_train, y_test)
confuse(y_test, pred_logreg)
compare(y_test, pred_most_frequent, pred_dummy, pred_tree, pred_logreg)
