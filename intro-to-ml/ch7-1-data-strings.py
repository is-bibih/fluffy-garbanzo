mport numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

'''
working with text data
    - some kinds of features to represent data properties: continuous,
      categorical, and text
    - this could be to classify an email as spam/legitimate, learn the
      opinion of a politician from speeches, classify customer messages
      as complaints/inquiries

types of data represented as strings
    - categorical data: items from a fixed list
    - free strings that can be semantically mapped to categories:
      like asking people their favourite colour and giving them a
      text field
    - structured string data: addresses, names, dates, telephone
      numbers, etc.
    - text data: phrases or sentences (tweets, chat logs, hotel
      reviews, books, wikipedia, etc.)
        - dataset is called corpus
        - each data point is called a document
'''
