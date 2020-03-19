import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sklearn.linear_model, sklearn.neighbors

# example 1-1. training and running a linear model using scikit-learn
def ex01():
    # path to load the data
    datapath = os.path.join("datasets", "lifesat", "")

    # load the data
    oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
    gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv",
                                 thousands=',',
                                 delimiter='\t',
                                 encoding='latin1',
                                 na_values="n/a")

    # function to prepare the data
    def prepare_country_stats(oecd_bli, gdp_per_capita):
        oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
        oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
        gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
        gdp_per_capita.set_index("Country", inplace=True)
        full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                      left_index=True, right_index=True)
        full_country_stats.sort_values(by="GDP per capita", inplace=True)
        remove_indices = [0, 1, 6, 8, 33, 34, 35]
        keep_indices = list(set(range(36)) - set(remove_indices))
        return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

    # prepare the data
    country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
    X = np.c_[country_stats["GDP per capita"]]
    y = np.c_[country_stats["Life satisfaction"]]

    # look at the data
    country_stats.plot(kind='scatter',
                       x="GDP per capita",
                       y="Life satisfaction")
    plt.show()

    # select a linear model
    model1 = sklearn.linear_model.LinearRegression()
    # select a k-nearest neighbors (3)
    model2 = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

    # train the model
    model1.fit(X, y)
    model2.fit(X, y)

    # make a prediction for cyprus
    X_new = [[22587]] # cyprus's gdp per capita
    print("linear: {}".format(model1.predict(X_new)))   # outputs 5.96242338
    print("knn: {}".format(model2.predict(X_new)))      # outputs 5.76666667

ex01()
