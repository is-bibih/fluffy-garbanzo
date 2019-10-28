import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

'''
bag-of-words with more than one word (n-grams)
    - it allows word order to be taken into account
        - otherwise "good, not bad at all" and "bad, not good at all"
          have the same representation
    - single tokens are called unigrams
    - pairs of tokens are called bigrams
    - triplets of tokens are called trigrams
    - sequences of tokens are called n-grams
    - the range of tokens considered as features can be changed with
      the ngram_range parameter on CountVectorizer or TfidfVectorizer
        - it's a tuple with the minimum and maximum length of the
          sequences of tokens
        - default is (1, 1)
    - longer sequences of tokens usually result in many more features
    - for most application, the minimum number of tokens should be one
      (single words often capture a lot of meaning)
    - adding longer sequences (up to 5) usually helps, but there will
      be a lot of features and it could lead to overfitting (many
      specific features)
        - max amount of bigrams: unigrams^2
        - max amount of trigrams: unigrams^3
        - in practice it's a lot less (because of the structure of
          english), but they're still a lot
'''

def example_ngram():
    # create a toy dataset
    bards_words = ["The fool doth think he is wise,",
                   "but the wise man knows himself to be a fool"]

    # look at the words in it
    print("bards_words:\n{}".format(bards_words))

    # look at unigrams
    cv = CountVectorizer(ngram_range=(1, 1)).fit(bards_words)
    print("vocabulary size: {}".format(len(cv.vocabulary_)))
    print("vocabulary:\n{}".format(cv.get_feature_names()))

    # look at bigrams
    cv = CountVectorizer(ngram_range=(2, 2)).fit(bards_words)
    print("vocabulary size: {}".format(len(cv.vocabulary_)))
    print("vocabulary:\n{}".format(cv.get_feature_names()))
    # look at its vectors
    print("transformed data (dense):\n{}".format(
        cv.transform(bards_words).toarray()))

    # look at unigrams, bigrams and trigrams
    cv = CountVectorizer(ngram_range=(1, 3)).fit(bards_words)
    print("vocabulary size: {}".format(len(cv.vocabulary_)))
    print("vocabulary:\n{}".format(cv.get_feature_names()))

def imdb_ngram(text_train, y_train, show=True):
    # this takes a really long time
    pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
    param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
                  "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}

    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(text_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters:\n{}".format(grid.best_params_))

    if show:
        # visualize results as a heat map
        # get scores
        scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
        # make heat map
        heatmap = mglearn.tools.heatmap(
            scores, xlabel="C", ylabel="ngram_range", cmap="viridis",
            fmt="%.3f", xticklabels=param_grid['logisticregression__C'],
            yticklabels=param_grid['tfidfvectorizer__ngram_range'])
        plt.colorbar(heatmap)

        # look at biggest coefficients
        vect = grid.best_estimator_.named_steps['tfidfvectorizer']
        feature_names = np.array(vect.get_feature_names())
        coef = grid.best_estimator_.named_steps['logisticregression'].coef_
        mglearn.tools.visualize_coefficients(
            coef, feature_names, n_top_features=40)
        plt.show()

example_ngram()
