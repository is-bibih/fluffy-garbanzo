import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

'''
stopwords
    - a way to get rid of uninformative words is by discarding
      words that are too frequent
    - it can be done with language-specific stopwords, or with
      discarding words that appear too frequently
        - scikit-learn has a built-in list of english stopwords
          in feature_extraction.text
    - removing the stopwords can only lead to a reduction in the
      features by the length of the list, but it might improve
      performance
'''

def only_load():
    # load bunch with data
    reviews_train = load_files("data/aclImdb/train")
    text_train, y_train = reviews_train.data, reviews_train.target

    # clean data from its weird html breaks
    text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

    # load test dataset
    reviews_test = load_files("data/aclImdb/test/")
    text_test, y_test = reviews_test.data, reviews_test.target

    # clean
    text_test = [doc.replace(b"<br />", b"  ") for doc in text_test]

    return text_train, text_test, y_train, y_test

def remove_stopwords(text_train):
    print("Number of stop words: {}".format(len(ENGLISH_STOP_WORDS)))
    print("Every 10th stopword:\n{}".format(list(ENGLISH_STOP_WORDS)[::10]))

    # stop_words="english" uses the built-in list
    # we could also pass our own
    # it removes most of the stop words
    vect = CountVectorizer(min_df=5, stop_words="english").fit(text_train)
    X_train = vect.transform(text_train)
    print("X_train with stop words:\n{}".format(repr(X_train)))

    return X_train

def grid_again(text_train, text_test, y_train, y_test):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(text_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))

text_train, text_test, y_train, y_test = only_load()
text_train = remove_stopwords(text_train)
grid_again(text_train, text_test, y_train, y_test)
