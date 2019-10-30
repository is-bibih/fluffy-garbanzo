import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import spacy
import nltk
import re
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, \
    TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV, \
    StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

'''
advanced tokenization, stemming, and lemmatization
    - feature extraction in CountVectorizer and TfidfVectorizer is
      relatively simple
    - singular and plural forms, different verb forms and related
      words as separate tokens is bad for building a model that
      generalizes well
    - tokenization is often improved to improve overall results
      by using normalization
        - stemming: dropping common suffixes
        - lemmatization: reducing words to lemmas, which are based
          on a dictionary of known word forms and the role of the
          word in a sentence
        - spelling correction: useful in practice
    - in general, lemmatization is a more complicated process, but
      produces better results than stemming
    - CountVectorizer lets you pass ypur own tokenizer (sklearn
      has neither form of normalization)
    - lemmatization can be seen as a kind or regularization, so
      it would improve performance most on small datasets
'''

# load spacy's english-language models
en_nlp = spacy.load('en')
# instatiate ntlk's porter stemmer
stemmer = nltk.stem.PorterStemmer()

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

# compare porter stemmer and spacy lemmatization
def compare_normalization(doc):
    # tokenize with spacy
    doc_spacy = en_nlp(doc)
    # print lemmas found
    print("lemmatization:")
    print([token.lemma_ for token in doc_spacy])
    # print tokens found by porter stemmer
    print("stemming:")
    print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])

# use regexp tokenizer from CountVectorizer and lemmatization
# from spacy
# replace the spacy tokenizer with regexp tokenization
def lemma_tokenizer():
        # get regexp used in CountVectorizer
    regexp = re.compile('(?u)\\b\\w\\w+\\b')

    # load spacy language model and save its old tokenizer
    en_nlp = spacy.load('en')
    old_tokenizer = en_nlp.tokenizer
    # replace tokenizer with regexp
    en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(
        regexp.findall(string))

    # create custom tokenizer with spacy document processing pipeline
    def custom_tokenizer(document):
        doc_spacy = en_nlp(document)
        return [token.lemma_ for token in doc_spacy]

    # define count vectorizer with custom tokenizer
    lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)

    return lemma_vect

# transform text_train using CountVectorizer with lemmatization
def lemma_transform(lemma_vect, text_train):
    X_train_lemma = lemma_vect.fit_transform(text_train)
    print("X_train_lemma.shape: {}".format(X_train_lemma.shape))

    # standard CountVectorizer for reference
    vect = CountVectorizer(min_df=5).fit(text_train)
    X_train = vect.transform(text_train)
    print("X_train.shape: {}".format(X_train.shape))

# look at effects of lemmatization on a small dataset
def small():
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    cv = StratifiedShuffleSplit(n_iter=5, test_size=0.99,
                                train_size=0.01, random_state=0)
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=cv)
    # grid search with CountVectorizer
    grid.fit(X_train, y_train)
    print("Best cross-validation score "
          "(standard CountVectorizer): {:.3f}".format(grid.best_score_))
    # grid search with lemmatization
    grid.fit(X_train_lemma, y_train)
    print("Best cross-validation score "
          "(lemmatization): {:.3f}".format(grid.best_score_))


# look at the differences
compare_normalization(u"Our meeting today was worse than yesterday,"
                       " I'm scared of meeting the clients tomorrow.")
lemma_vect = lemma_tokenizer()
text_train, text_test, y_train, y_test = only_load()
lemma_transform(lemma_vect, text_train)
