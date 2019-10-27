import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

'''
term frequency-inverse document frecuency (tf-idf)
    - rescale features by how informative we expect them to be
    - it gives weight to a word that appears often in a particular
      document, but not in many documents in the corpus
        - if a word appears a lot in only one document, it is
          likely to be descriptive of its content
    - TfidfTransformer takes the sparse matrix from CountVectorizer
      and transforms it
    - TfidfVectorizer takes the text data and does the bag-of-words
      and the tf-idf transformation
    - the tf-idf score for word w in document d is given by
        tfdif(w, d) = tf log((N + 1)/(N_w + 1)) + 1
      N:    number of documents in the training set
      N_w:  number of documents in the training set that w appears in
      tf:   number of times that w appears in d
    - they rescale the representation so that each document has
      euclidean norm 1
    - it's important to use a pipe because it uses statistical
      properties of the training data
    - features with a low tf-idf score (regular)
        - used very commonly throughout documents
        - only used sparingly in very long documents
    - features with high tf-idf (specific)
        - appear in few documents
        - are used a lot in a single document
    - inverse document frquency: how often features show up across
      documents
        - tf-idf != idf
        - low idf means less important, more frequent
        - includes stopwords and domanin-specific words (movie, actor,
          story, etc)

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

def make_pipe_tfidf(text_train, text_test, y_train, y_test):
    pipe = make_pipeline(TfidfVectorizer(min_df=5, norm=None),
                         LogisticRegression())
    param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}

    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(text_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))

    # get TfidfVectorizer from pipeline
    vectorizer = grid.best_estimator_.named_steps["tfidvectorizer"]
    # transform the training dataset
    X_train = vectorizer.transform(text_train)
    # find max value for each feature in the dataset
    max_value = X_train.max(axis=0).toarray().ravel()
    sorted_by_tfidf = max_value.argsort()
    # get feature names
    feature_names = np.array(vectorizer.get_feature_names())

    print("features with lowest tfidf:\n{}".format(
        feature_names[sorted_by_tfidf[:20]]))
    print("Features with highest tfidf:\n{}".format(
        feature_names[sorted_by_tfidf[-20:]]))

    # looks at words with low inverse document frequency (less important)
    sorted_by_tfidf = np.argsort(vectorizer.idf_)
    print("Features with lowest idf:\n{}".format(
        feature_names[sorted_by_tfidf[:100]]))

text_train, text_test, y_train, y_test = only_load()
make_pipe_tfidf(text_train, text_test, y_train, y_test)
