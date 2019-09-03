from sklearn.datasets import load_iris
iris_dataset = load_iris() # returns a Bunch (like a dict)

# get keys in Bunch
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
# print(iris_dataset['DESCR'] + "\n...")

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
print("Type of data: {}".format(type(iris_dataset['data'])))

# the data
print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))
# targets (species)
print("Type of target: {}".format(type(iris_dataset['target'])))
