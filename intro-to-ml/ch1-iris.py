import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris() # returns a Bunch (like a dict)

# get keys in Bunch
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
# print(iris_dataset['DESCR'] + "\n...")

print('\n')

# get target_names (species of flower to be predicted)
print("Target names: {}".format(iris_dataset['target_names']))
# get feature_names (dataset features)
print("Feature names: \n{}".format(iris_dataset['feature_names']))
# get data type (data is stored in a nparray)
print("Type of data: {}".format(type(iris_dataset['data'])))
# get data shape
#	- rows correspond to samples and columns to features
#	- in scikit-learn (a convention is that) data shape is
#	  number of samples * number of features
print("Shape of target: {}".format(iris_dataset['target'].shape))

print('\n')

# the data
print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))
# targets (species)
print("Type of target: {}".format(type(iris_dataset['target'])))

print('\n')

# train_test_split extracts 75% of the data to use as training set,
# the rest is the test sest

# X represents the data and the labels are y
#	- it comes from f(x) = y
#	- X is a capital letter because the data is a two-dimensional array
#	- y is a lowercase because it is one-dimensional (a vector)

X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

print('\n')

# pair plots show all possible pairs of features
# pandas has a scatter_matrix (creates pair plots) but it takes a DataFrame
# the diagonal of the scatter_matrix is filled with histograms of each feature

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
# plt.show()

# k-nearest neighbors: finds data point closest to new data point and uses its label
# in scikit-learn, ml models are Estimator classes (KNeigborsClassifier for k-nearest)
knn = KNeighborsClassifier(n_neighbors=1) # number of neighbors is 1
# fit trains the estimator with the training data and the training labels
knn.fit(X_train, y_train)

# test data point
# scikit-learn always expects 2d arrays
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

# predict
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
iris_dataset['target_names'][prediction]))

# test with testing dataset
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
# summary
# X_train, X_test, y_train, y_test = train_test_split(
# iris_dataset['data'], iris_dataset['target'], random_state=0)
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train, y_train)
# print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
