import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge

'''
grid-searching preprocessing steps and model parameters
    - pipelines can encapsulate all the preprocessing steps
      in a single estimator
    - it's also possible to adjust preprocessing parameters
      using the outcomes of supervised tasks
    - increasing more parameters exponentially increases the
      number of models to be built
'''

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data,
    boston.target, random_state=0)

# polynomials features and then ridge regressor with a pipeline
def poly_ridge(X_train, X_test, y_train, y_test, show=True):
    pipe = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(),
        Ridge())

    # param_grid with parameters for degree and for alpha
    param_grid = {'polynomialfeatures__degree': [1, 2, 3],
                  'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    # run search-grid
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    # look at outcomes with heat map
    if show:
        mglearn.tools.heatmap(grid.cv_results_['mean_test_score'].reshape(3, -1),
              xlabel="ridge__alpha", ylabel="polynomialfeatures__degree",
              xticklabels=param_grid['ridge__alpha'],
              yticklabels=param_grid['polynomialfeatures__degree'], vmin=0)
        plt.show()

    # look at best parameters and score
    print("Best parameters: {}".format(grid.best_params_))
    print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))

def no_poly(X_train, X_test, y_train, y_test):
    param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    pipe = make_pipeline(StandardScaler(), Ridge())
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Score without poly features: {:.2f}".format(
        grid.score(X_test, y_test)))

poly_ridge(X_train, X_test, y_train, y_test, show=False)
no_poly(X_train, X_test, y_train, y_test)
