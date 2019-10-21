import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_files

'''
sentiment analysis of movie reviews
    - text from internet movie database classified as positive
      or negative (positive is score of 6 or more)
'''

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

# look at number of documents and samples per class
# reviews_test
print("Number of documents in test data: {}".format(len(text_test)))
print("Samples per class (test): {}".format(np.bincount(y_test)))
