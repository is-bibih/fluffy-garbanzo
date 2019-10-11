import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, \
    GridSearchCV

iris = load_iris()

'''
grid search
    - tries all possible combinations of the parameters of interest
      for a model
    - it helps with parameter tuning to improve model generalization
    - for example, two parameters with six possible values yields
      a grid of 36 parameter settings for the model
    - can be implemented by looping over the parameter values
    - the same test set cannot be used to define parameters and to
      evaluate the model's generalization
      - since the parameters are optimized for that test set, and
        would not necessarily generalize well to other datasets
      - this can be solved by splitting the dataset into 3 groups
        - training set
        - validation set (development, for parameters)
        - test set
      - once the best parameters are found with the validation set,
        a model can be built and trained on both the training and
        the validation data, to get as much information as possible

grid search with cross-validation
    - use cross-validation for better generalization
    - simple train-test-validation split might make the model
      generalize worse
    - very resource-intensive
    - it can be implemented with GridSearchCV
        - it looks for the best parameters and automatically fits a
          new model on the whole dataset
        - best parameters are stored in the best_params_ attribute
        - mean accuracy for best parameters is in best_score_
            - it is only on the training set, not necessarily accurate
        - look at results in cv_results_attribute
        - it can take a list of dictionaries to look for parameters
          for different models (which use different parameters)
    - usually a good idea to start with a very coarse grid and
      then refine the search

other things with cross-validation
    - GridSearchCV uses stratified k-fold cross-validation for
      classification and k-fold cross-validation for regression
        - other splitters can be passed as cv argument
    - to get a single split (for very large sets or slow models),
      use ShuffleSplit or StratifiedShuffleSplit with n_iter=1
'''

# naive grid search implementation

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0)
print("Size of training set: {}     size of test set: {}".format(
      X_train.shape[0], X_test.shape[0]))

best_score = 0

# train an SVC for each combinations
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        # evaluate SVC
        score = svm.score(X_test, y_test)
        # compare scores
        if score > best_score:
            best_score = score
            best_parameters =  {'C': C, 'gamma': gamma}

print("Best score: {:.2f}".format(best_score))
print("Best parameters: {}".format(best_parameters))

# grid search with validation set

# split data into train-validation and test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    iris.data, iris.target, random_state=0)
# split train-validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, random_state=1)
print("size of training set: {}    size of validation set: {}" \
      "    size of test set: {}\n".format(X_train.shape[0],
                                          X_valid.shape[0],
                                          X_test.shape[0]))

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        # evaluate SVC with validation set
        score = svm.score(X_valid, y_valid)
        # compare scores
        if score > best_score:
            best_score = score
            best_parameters =  {'C': C, 'gamma': gamma}

# build model with complete data and best parameters
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print("best score on validation: {:.2f}".format(best_score))
print("best parameters: {}".format(best_parameters))
print("Test set score with best parameters: {:.2f}".format(test_score))

# grid search with cross-validation

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        # do cross-validation
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)

# grid search with cross-validation in GridSearchCV
# use a dict for the arguments

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
# the grid_search object behaves like a normal classifier
grid_search = GridSearchCV(SVC(), param_grid, cv=5)

# the data should be split to avoid overfitting anyways
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0)

# train
grid_search.fit(X_train, y_train)

print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# access the actual estimator
print("Best estimator:\n{}".format(grid_search.best_estimator_))

# look at results as a DataFrame
results = pd.DataFrame(grid_search.cv_results_)
print(results.head())

# look at results in a heat map
scores = np.array(results.mean_test_score).reshape(6, 6)
mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],
                      ylabel='C', yticklabels=param_grid['C'], cmap='viridis')

# look at bad grid searches
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
param_grid_linear = {'C': np.linspace(1, 2, 6),
                     'gamma': np.linspace(1, 2, 6)}
param_grid_one_log = {'C': np.linspace(1, 2, 6),
                      'gamma': np.logspace(-3, 2, 6)}
param_grid_range = {'C': np.logspace(-3, 2, 6),
                    'gamma': np.logspace(-7, -2, 6)}

for param_grid, ax in zip([param_grid_linear, param_grid_one_log,
                           param_grid_range], axes):
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    scores = grid_search.cv_results_['mean_test_score'].reshape(6, 6)
    # plot the mean cross-validation scores
    scores_image = mglearn.tools.heatmap(
        scores, xlabel='gamma', ylabel='C', xticklabels=param_grid['gamma'],
        yticklabels=param_grid['C'], cmap="viridis", ax=ax)
    plt.colorbar(scores_image, ax=axes.tolist())
# plt.colorbar(scores_image, ax=axes.tolist())
# plt.show()

# a grid search with both kernel and parameters

param_grid = [{'kernel': ['rbf'],
               'C': [0.001, 0.01, 0.1, 1, 1, 10, 100],
               'gamma': [0.001, 0.01, 0.1, 1, 1, 10, 100]},
              {'kernel': ['linear'],
               'C': [0.001, 0.01, 0.1, 1, 1, 10, 100]}]
print("List of grids:\n{}".format(param_grid))

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# look at results
results = pd.DataFrame(grid_search.cv_results_)
# use transposed table so it fits better
print(results.T)
