import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, \
    classification_report

'''
metrics for multiclass classification
    - multiclass classification metrics are binary metrics, but
      averaged over all classes
    - multiclass accuracy: fraction of correctly classified samples
    - multiclass metrics are usually harder to understand than
      binary metrics
    - besides accuracy, common tools are the confusion matrix and
      the classification report

f1 for multiclass
    - macro: average unweighted per-class f-scores
    - weighted: average weighted per-class f-scores, according to
      their support
    - micro: total number of fp, fn, and tp, then computes precision,
      recall and f-score with those counts
'''

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, random_state=0)

# confusion matrix and classification report on digits
def confuse(X_train, X_test, y_train, y_test, show=True):
    lr = LogisticRegression().fit(X_train, y_train)
    pred = lr.predict(X_test)
    print("Accuracy: {:.3f}".format(accuracy_score(y_test, pred)))
    print("Confusion matrix:\n{}".format(confusion_matrix(y_test, pred)))

    # look at classification report
    print(classification_report(y_test, pred))

    if show:
        scores_image = mglearn.tools.heatmap(
            confusion_matrix(y_test, pred), xlabel="Predicted label",
            ylabel="True label", xticklabels=digits.target_names,
            yticklabels=digits.target_names, cmap=plt.cm.gray_r, fmt="%d")
        plt.title("Confusion matrix")
        plt.gca().invert_yaxis()
        plt.show()

    return pred

confuse(X_train, X_test, y_train, y_test, show=False)
