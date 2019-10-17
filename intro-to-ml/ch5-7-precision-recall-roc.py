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
from sklearn.metrics import f1_score

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

X, y = make_blobs(n_samples=(4000, 500), centers=2,
                  cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

# look at precision vs recall curves for an svc and a random forest
def pr_curve(X_train, X_test, y_train, y_test, show=True):
    # it can be plotted into a curve when using more points
    svc = SVC(gamma=0.05).fit(X_train, y_train)
    precision, recall, thresholds = precision_recall_curve(
        y_test, svc.decision_function(X_test))

    # find threshold closest to 0 (to see the default)
    close_zero = np.argmin(np.abs(thresholds))

    # compare with random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=0,
                                max_features=2)
    rf.fit(X_train, y_train)
    # use predict_proba for precision recall curve
    precision_rf, recall_rf, thresholds_rf = precision_recall_curve(
        y_test, rf.predict_proba(X_test)[:, 1])

    if show:
        # plot
        plt.plot(precision, recall, label="svc")
        plt.plot(precision[close_zero], recall[close_zero], 'o',
            markersize=10, label="threshold zero svc", fillstyle="none",
            c='k', mew=2)
        plt.plot(precision_rf, recall_rf, label="rf")
        close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
        plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf],
                 '^', c='k', markersize=10, label="threshold 0.5 rf",
                 fillstyle="none", mew=2)
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.legend(loc="best")

        plt.show()

    return rf, svc


def f1_scores(rf, svc, X_test, y_test):
    print("f1_score of random forest: {:.3f}".format(
        f1_score(y_test, rf.predict(X_test))))
    print("f1_score of svc: {:.3f}".format(
        f1_score(y_test, svc.predict(X_test))))

rf, svc = pr_curve(X_train, X_test, y_train, y_test, show=False)
f1_scores(rf, svc, X_test, y_test)
