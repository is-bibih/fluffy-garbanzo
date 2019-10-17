import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.metrics import precision_recall_curve
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from mglearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier

'''
precision-recall curves and roc curves
    - operating point: setting a requirement on a classifier
      like 90% recall
        - it should depend on the business goals
        - it's always possible to set a threshold to meet a
          particular target, but it's hard to develop a model
          that still has reasonable precision within the
          threshold

precision-recall curve
    - it looks at all possible thresholds at once
        - so it also looks at all possible trade-offs between
          precision and recall
    - it needs the ground truth labeling and predicted uncertainties
      (predicted with decision_function or predict_proba)
    - precision_recall_curve returns a list of precision and
      recall curves for all possible thresholds
    - the closer the curve is to the upper right corner, the better
    - different classifiers can work well in different parts of the
      curve (different operating points)
'''

digits = load_digits()
y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, y, random_state=0)

svc = SVC(gamma=.05).fit(X_train, y_train)

def pr_curve(y_test, X_test, svc):
    precision, recall, thresholds = precision_recall_curve(
        y_test, svc.decision_function(X_test))

    # it can be plotted into a curve when using more points
    X, y = make_blobs(n_samples=(4000, 500), centers=2,
                      cluster_std=[7.0, 2], random_state=22)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    svc = SVC(gamma=0.05).fit(X_train, y_train)
    precision, recall, thresholds = precision_recall_curve(
        y_test, svc.decision_function(X_test))

    # find threshold closest to 0 (to see the default)
    close_zero = np.argmin(np.abs(thresholds))
    plt.plot(precision[close_zero], recall[close_zero], 'o',
             markersize=10, label="threshold zero", fillstyle="none",
             c='k', mew=2)

    # plot
    plt.plot(precision, recall, label="precision recall curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.show()

pr_curve(y_test, X_test, svc)
