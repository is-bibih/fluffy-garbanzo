import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, \
    classification_report
from mglearn.datasets import make_blobs

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

confusion matrices summaries:
    - accuracy = (tn + tp)/total
    - precision aka positive predictive value = tp/(tp + fp)
        - looks for low amount of false positives
    - recall aka sensitivity aka hit rate aka tp rate = tp/(tp + fn)
        - avoiding false negatives
    - f-score = 2*(precision*recall)/(precision + recall)
        - this one in specific is the f_1-score
        - picking the positive class can have a big impact
          on the metrics
    - classification_report gets precision, recall and f1-score
        - it produces one line per class
        - support is the amount of samples according to ground truth
        - the last row shows a weighted average

taking uncertainty into account
    - most classifiers have a decision_function or a predict_proba
      that access degrees of certainty about predictions
    - by default, points with a decision_function > 0 are
      classified as class 1
        - if we want more points to be classified as 1, we
          decrease the threshold
        - if precision is valued over recall or the other way around,
          or the data is heavily unbalanced, changing the decision
          threshold is an easy way to improve results
    - it can be easier to improve results for models with predict_proba
      (it is fixed on a 0 to 1 scale)
        - by default, the thresholf of 0.5 means that a point will
          be positive if the model is 50% sure it is
        - calibrated models provide accurate measures of uncertainty
            - a fully grown decision tree is not calibrated,
              it is always 100% sure of its decisions
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

def f1_scores(y_test, pred_most_frequent, pred_dummy, pred_tree,
            pred_logreg):
    # most frequent gets an error because there were no
    # positive predictions (division by 0)
    print("f1 score most frequent: {:.2f}".format(
        f1_score(y_test, pred_most_frequent)))
    print("f1 score dummy: {:.2f}".format(
        f1_score(y_test, pred_dummy)))
    print("f1 score tree: {:.2f}".format(
        f1_score(y_test, pred_tree)))
    print("f1 score logreg: {:.2f}".format(
        f1_score(y_test, pred_logreg)))

def clas_report(y_test, pred_most_frequent, pred_dummy, pred_tree,
            pred_logreg):
    print(classification_report(y_test, pred_most_frequent,
                                target_names=["not nine", "nine"]))
    print(classification_report(y_test, pred_dummy,
                                target_names=["not nine", "nine"]))
    print(classification_report(y_test, pred_tree,
                                target_names=["not nine", "nine"]))
    print(classification_report(y_test, pred_logreg,
                                target_names=["not nine", "nine"]))

def uncertainty():
    # unbalanced binary classification (400 negative and 50 positive)
    # plots decision threshold for a kernel svm model
    X, y = make_blobs(n_samples=(400, 50), centers=2,
        cluster_std=[7.0, 2], random_state=22)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    svc = SVC(gamma=.05).fit(X_train, y_train)
    mglearn.plots.plot_decision_threshold()
    plt.show()

    # use classification_report to evaluate precision and recall
    # for both classes
    print(classification_report(y_test, svc.predict(X_test)))
    # there's a smaller recall and mixed precision for class 1,
    # since the classifier focuses on 0 since it's much larger

    y_pred_lower_threshold = svc.decision_function(X_test) > -0.8
    print(classification_report(y_test, y_pred_lower_threshold))
    # this makes the recall for class 1 go up and the precision
    # goes down

# pred_most_frequent = imbalanced(X_train, X_test, y_train, y_test)
# pred_tree = decision_tree(X_train, X_test, y_train, y_test)
# pred_dummy, pred_logreg = other_dummy(X_train, X_test, y_train, y_test)
# confuse(y_test, pred_logreg)
# compare(y_test, pred_most_frequent, pred_dummy, pred_tree, pred_logreg)
# f1_scores(y_test, pred_most_frequent, pred_dummy, pred_tree, pred_logreg)
# clas_report(y_test, pred_most_frequent, pred_dummy, pred_tree, pred_logreg)
uncertainty()
