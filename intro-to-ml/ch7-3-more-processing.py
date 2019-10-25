import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

'''
stopwords
    - a way to get rid of uninformative words is by discarding
      words that are too frequent
    - it can be done with language-specific stopwords, or with
      discarding words that appear too frequently
        - scikit-learn has a built-in list of english stopwords
          in feature_extraction.text
'''
