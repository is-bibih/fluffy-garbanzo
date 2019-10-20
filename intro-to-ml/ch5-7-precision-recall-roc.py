import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_digits, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, f1_score, \
    average_precision_score, roc_curve, roc_auc_score

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
    - a way to summarize the precision-recall curve is through the
      area under the curve, aka the average precision
        - use the average_precision_score, which takes the result
          of decision_function or predict_proba, not predict
        - it is always between 0 (worst) and 1 (best)
        - the average precision of a random classifier is the fraction
          of positive samples in the dataset

receiver operating characteristics (roc) curve and auc
    - shows the false positive rate against the true positive
      rate (recall)
        - fpr = fp/(fp + tn)
        - tpr = tp/(tp + fn)
    - the ideal curve is close to the top left; high recall with
      a low false positive rate
        - the point closest to the top left might be a better
          operating point than the default
        - the threshold should not be picked on the test set, it
          should be on a separate validation set
    - it is usually summarized with the area under the curve
      (referred to as AUC)
        - computed with roc_auc_score
        - random predictions yield an auc of 0.5, even for unbalanced
          datasets
            - this makes it a better metric for unbalanced
              classification than accuracy
        - it is equivalent to the probability that a randomly picked
          positive sample will have a higher score according to the
          classifier than a point from the negative class
        - a perfect score of 1 means all positive points are classified
          correctly
'''

X, y = make_blobs(n_samples=(4000, 500),
                  cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

digits = load_digits()
y9 = digits.target == 9

X9_train, X9_test, y9_train, y9_test = train_test_split(
    digits.data, y9, random_state=0)

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

# look at f1 scores
def f1_scores(rf, svc, X_test, y_test):
    print("f1_score of random forest: {:.3f}".format(
        f1_score(y_test, rf.predict(X_test))))
    print("f1_score of svc: {:.3f}".format(
        f1_score(y_test, svc.predict(X_test))))

# look at average precision
def average_precision(rf, svc, X_test, y_test):
    ap_rf = average_precision_score(
        y_test, rf.predict_proba(X_test)[:, 1])
    ap_svc = average_precision_score(
        y_test, svc.decision_function(X_test))
    print("Average precision of random forest: {:.3f}".format(ap_rf))
    print("Average precision of svc: {:.3f}".format(ap_svc))

# look at roc curves
def roc(rf, svc, X_test, y_test):
    fpr, tpr, thresholds = roc_curve(
        y_test, svc.decision_function(X_test))
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(
        y_test, rf.predict_proba(X_test)[:, 1])

    plt.plot(fpr, tpr, label="ROC Curve SVC")
    plt.plot(fpr_rf, tpr_rf, label="ROC Curve RF")

    plt.xlabel("FPR")
    plt.ylabel("TPR (recall)")

    # find threshold closest to 0
    close_zero = np.argmin(np.abs(thresholds))
    close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))

    plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
             label="threshold zero SVC", fillstyle="none", c='k', mew=2)
    plt.plot(fpr[close_default_rf], tpr[close_default_rf],
             '^', markersize=10, label="threshold 0.5 RF",
             fillstyle="none", c='k', mew=2)

    plt.legend(loc=4)
    plt.show()

# look at area under the curve
def auc(rd, svc, X_test, y_test):
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    svc_auc = roc_auc_score(y_test, svc.decision_function(X_test))
    print("AUC for random forest: {:.3f}".format(rf_auc))
    print("AUC for svc: {:.3f}".format(svc_auc))

def gammas(X_train, X_test, y_train, y_test):
    plt.figure()

    for gamma in [1, 0.05, 0.01]:
        svc = SVC(gamma=gamma).fit(X_train, y_train)
        accuracy = svc.score(X_test, y_test)
        auc = roc_auc_score(y_test, svc.decision_function(X_test))
        fpr, tpr, _ = roc_curve(y_test, svc.decision_function(X_test))
        print("gamma = {:.2f}  accuracy = {:.2f}  AUC = {:.2f}".format(
            gamma, accuracy, auc))
        plt.plot(fpr, tpr, label="gamma={:.3f}".format(gamma))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(-0.01, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc="best")
    plt.show()

# rf, svc = pr_curve(X_train, X_test, y_train, y_test, show=False)
# f1_scores(rf, svc, X_test, y_test)
# average_precision(rf, svc, X_test, y_test)
# roc(rf, svc, X_test, y_test)
# auc(rf, svc, X_test, y_test)
gammas(X9_train, X9_test, y9_train, y9_test)
