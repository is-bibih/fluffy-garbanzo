import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

'''
sentiment analysis of movie reviews
    - text from internet movie database classified as positive
      or negative (positive is score of 6 or more)

bag-of-words representation
    - it counts how often each word appears in each text in the
      corpus
    - it has three steps
        - tokenization: split the document into the words that appear
          it (the tokens), for example by whitespace and punctuation
        - vocabulary building: collect all the words that appear and
          number them (in alphabetical order or something)
        - encoding: count how often the words in the vocabulary appear
          in each document
    - there's a feature for each unique word in the dataset
    - the order of words is not relevant
    - implemented in CountVectorizer, which is a transformer
    - it's stored in a SciPy sparse matrix (only stores nonzero entries)
    - access the vocabulary with vocabulary_ or get_feature_names()

processing
    - logreg usually works well with sparse high-dimensional data
      (like this)
'''

def load_texts():
    # load bunch with data
    reviews_train = load_files("data/aclImdb/train")
    text_train, y_train = reviews_train.data, reviews_train.target

    print("type of text_train: {}".format(type(text_train)))
    print("length of text_train: {}".format(len(text_train)))
    print("text_train[1]:\n{}".format(text_train[1]))

    # clean data from its weird html breaks
    text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

    # look at balanced class
    print("Samples per class (training): {}".format(np.bincount(y_train)))

    # load test dataset
    reviews_test = load_files("data/aclImdb/test/")
    text_test, y_test = reviews_test.data, reviews_test.target

    # look at number of documents and samples per class
    print("Number of documents in test data: {}".format(len(text_test)))
    print("Samples per class (test): {}".format(np.bincount(y_test)))

    # clean
    text_test = [doc.replace(b"<br />", b"  ") for doc in text_test]

    return text_train, text_test, y_train, y_test

def example_bag():
    # create a toy dataset
    bards_words = ["The fool doth think he is wise,",
                   "but the wise man knows himself to be a fool"]

    # fit the CountVectorizer to the data
    # this tokenizes the data and builds the vocabulary
    vect = CountVectorizer()
    vect.fit(bards_words)

    # look at vocabulary
    print("vocabulary size: {}".format(len(vect.vocabulary_)))
    print("vocabulary content:\n{}".format(vect.vocabulary_))

    # create the bag-of-words representation with transform
    # each row is a data point (a text)
    bag_of_words = vect.transform(bards_words)
    print("bag_of_words: {}".format(repr(bag_of_words)))

    # convert it to a "dense" numpy array to look at it
    print("Dense representation of bag_of_words\n{}".format(
        bag_of_words.toarray()))

def process_movies(text_train, text_test, y_train, y_test):
    vect = CountVectorizer().fit(text_train)
    X_train = vect.transform(text_train)
    print("X_train:\n{}".format(repr(X_train)))

    # look at vocabulary better
    feature_names = vect.get_feature_names()
    print("Number of features: {}".format(len(feature_names)))
    print("First 20 features:\n{}".format(feature_names[:20]))
    print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
    print("Every 2000th feature:\n{}".format(feature_names[::2000]))

    # look at crossval with logistic regression
    scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
    print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))

text_train, text_test, y_train, y_test = load_texts()
# example_bag()
process_movies(text_train, text_test, y_train, y_test)
