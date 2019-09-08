import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split

'''
naive bayes classifiers
	- faster in training than linear classifiers
	- usually slightly worse generalization than linear models
	- often used with very large datasets
	- they collect class statistics from each feature
	- data point values are compared to class statistics
	  and classified in nearest match
	- all have single parameter alpha
		- larger alpha, simpler models
		- adds virtual data points with positive
		  values for all features
		- usually not critical

GaussianNB
	- works on any continous data
	- average value and stddev of each feature per class
	- mostly used on very high-dimensional data

BernoulliNB
	- binary data
	- usually used for text (sparse count data)
	- counts how often every feature of a class is not 0

MultinomialNB
	- count data (integer numbers that count something)
	- usually used for text (sparse count data)
	- uses average value of each feature per class
	- usually performs better than bnb, especially with
	  large amount of nonzero features
'''

# bernoulli nb example

X = np.array([[0, 1, 0, 1],
			  [1, 0, 1, 1],
			  [0, 0, 0, 1],
			  [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])
# 1st and 3rd data points are class 0, 2nd and 4th are 1

# counting nonzero entries per class
counts = {}
for label in np.unique(y):
	# iterate over each class
	# count (sum) entries of 1 per feature
	counts[label] = X[y == label].sum(axis=0)
print("Feature counts:\n{}".format(counts))
