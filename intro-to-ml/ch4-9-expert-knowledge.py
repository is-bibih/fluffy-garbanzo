import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

'''
utilizing expert knowledge
    - can be useful for feature engineering
        - like adding a feature to inform whether a flight was
          on a school holiday, which might not be obvious from
          just the date
    - can help augment data, even if it turns out to be
      uninformative
    - training and test data should be split in relation to a
      certain date for chronological data
'''

# predict whether there will be an available rental bike

citibike = mglearn.datasets.load_citibike()
print("Citi Bike data: \n{}".format(citibike.head()))

# look at rental frequencies in the month

# plt.figure(figsize=(10, 3))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(),
                       freq='D')
# plt.xticks(xticks, xticks.strftime("%a %m-%d"), rotation=90, ha="left")
# plt.plot(citibike, linewidth=1)
# plt.xlabel("Date")
# plt.ylabel("Rentals")
# plt.show()

# extract target values (number of rentals)
y = citibike.values
# convert time to posix time
X = citibike.index.astype("int64").values.reshape(-1, 1) // 10**9

# use first 184 data points for training
n_train = 184

# make function to split data, build the model and see result
def eval_on_features(features, target, regressor):
    # split features
    X_train, X_test = features[:n_train], features[n_train:]
    # split target
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)

    print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)

    plt.figure(figsize=(10, 3))
    plt.xticks(range(0, len(X), 8), xticks.strftime("%a %m-%d"),
               rotation=90, ha="left")
    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
    plt.plot(range(n_train), y_pred_train, '--', label="prediction train")
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--',
             label="prediction test")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("Date")
    plt.ylabel("Rentals")
    plt.show()

# trees cannot extrapolate on data outside training range
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
# eval_on_features(X, y, regressor)

# predictions according to time of day
X_hour = citibike.index.hour.values.reshape(-1, 1)
# eval_on_features(X_hour, y, regressor)

# predictions with day of the week
X_hour_week = np.hstack((citibike.index.dayofweek.values.reshape(-1, 1),
                         citibike.index.hour.values.reshape(-1, 1)))
# eval_on_features(X_hour_week, y, regressor)

# use linear model instead because of simple features

# eval_on_features(X_hour_week, y, LinearRegression())

# improve dataset with OneHotEncoder, since weekday and time
# of day are catergorical variables
enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()
# eval_on_features(X_hour_week_onehot, y, Ridge())

# use interactions so that model learns a coefficient for each
# time-day combination, instead of one time of the day pattern
# for each day of the week
poly_transformer = PolynomialFeatures(degree=2, interaction_only=True,
                                      include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr)

# look at learned coeffcients
hour = ["%02d:00" % i for i in range(0, 24, 3)]
day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
features = day + hour
features_poly = poly_transformer.get_feature_names(features)
features_nonzero = np.array(features_poly)[lr.coef_ != 0]
coef_nonzero = lr.coef_[lr.coef_ != 0]

plt.figure(figsize=(15, 2))
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel("Feature magnitude")
plt.ylabel("Feature")
plt.show()
