import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, \
    TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV, \
    StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import LatentDirichletAllocation

'''
topic modeling and document clustering
    - topic modeling: umbrella term that describes the task of
      assigning each document to one or multiple topics, usually
      without supervision
        - if it is a single topic, then it's clustering
        - if it is more than one topic, it is decomposition
          (coefficients if the components in the representation
          show how related the document is)

latent dirichlet allocation (LDA)
    - decomposition methos used for topic modeling
    - it tries to find groups of words (the topics) that appear
      together often
    - each document is a mixture of a subset of the topics
    - the topics are the components extracted by pca or nmf (they
      do not necessarily have a semantic meaning)
      - like "words usually used by author A"
    - it's usually best to remove very common words for unsupervised
      models
    - changing the number of topics changes all topics
    - the components_ attribute stores how important each word is
      for each topic
        - its shape is (n_topics, n_words)
    - another way to inspect the topics is to see how much weight
      each topic get over-all by summing the document_topics over
      all reviews
    - good libraries: spacy, nltk, word2vec, tensorflow
'''

# lda on movie review dataset: remove words that appear in at least
# 20% of the documents, limit the bag-of-words to the 10000 words
# which are the most common

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

def do_lda(text_train):
    vect = CountVectorizer(max_features=10000, max_df=0.15)
    X = vect.fit_transform(text_train)

    # learn a topic model with 10 topics
    # with batch (slower than online) and max_iter
    lda = LatentDirichletAllocation(n_topics=10,
                                    learning_method="batch",
                                    max_iter=25,
                                    random_state=0)

    # build and transform in same step (saves time)
    document_topics = lda.fit_transform(X)

    # look at shape (n_topics, n_words)
    lda.components_.shape

    # look at most important words for each topic
    sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
    # get feature names from vectorizer
    feature_names = np.array(vect.get_feature_names())

    # print out the 10 topics
    mglearn.tools.print_topics(topics=range(10),
                               feature_names=feature_names,
                               sorting=sorting,
                               topics_per_chunk=5,
                               n_words=10)

# look at 100 topics now
def lda_100(text_train):
    vect = CountVectorizer(max_features=10000, max_df=0.15)
    X = vect.fit_transform(text_train)

    lda100 = LatentDirichletAllocation(n_topics=100,
                                       learning_method="batch",
                                       max_iter=25,
                                       random_state=0)
    document_topics100 = lda100.fit_transform(X)

    # look at topics that are interesting
    topics = np.array([7, 16, 24, 25, 28, 36, 37, 45, 51,
                       53, 54, 63, 89, 97])

    sorting = np.argsort(lda100.components_, axis=1)[:, ::-1]
    feature_names = np.array(vect.get_feature_names())
    mglearn.tools.print_topics(topics=topics,
                               feature_names=feature_names,
                               sorting=sorting,
                               topics_per_chunk=7,
                               n_words=20)

    # look at reviews assigned to topic 45, which seems to be
    # about music
    music = np.argsort(document_topics100[:, 45])[::-1]
    # print the 5 documents in which the topic is most important
    for i in music[:10]:
        # show first two sentences
        print(b".".join(text_train[i].split(b".")[:2]) + b".\n")

    # look at topic weights (topics are named by two most
    # common words)
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    topic_names = ["{:>2} ".format(i) + " ".join(words)
                   for i, words in enumerate(feature_names[sorting[:, :2]])]
    # two column bar chart:
    for col in [0, 1]:
        start = col * 50
        end = (col + 1) * 50
        ax[col].barh(np.arange(50),
                     np.sum(document_topics100, axis=0)[start:end])
        ax[col].set_yticks(np.arange(50))
        ax[col].set_yticklabels(topic_names[start:end], ha="left", va="top")
        ax[col].invert_yaxis()
        ax[col].set_xlim(0, 2000)
        yax = ax[col].get_yaxis()
        yax.set_tick_params(pad=130)
    plt.tight_layout()

text_train, text_test, y_train, y_test = only_load()
