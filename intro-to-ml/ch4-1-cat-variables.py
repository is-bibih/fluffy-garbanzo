import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

'''
representing data and engineering features
    - data is not always a set of continuous features
    - discrete aka categorical features are susually not numeric
        - they have no natural order (books > clothes does
          not work)
    - feature engineering: figuring out how to best represent
      data for a particular application

categorical variables
    - one-hot encoding aka dummy variables aka one-out-of-N
      encoding: a feature is created for each value and they are
      assigned 0 or 1 values
        - only one takes the value of 1 and the rest are 0 (it
          is 1 in the feature it takes the value of)
    - one-hot encoding should be applied simultaneously to
      training and test data to ensure the same features are
      present
'''

# example: 1994 adult incomes in the us
# task is to predict whether income is under or over $50,000
# includes how they are employed, age, education, gender, etc.
# classes are <=50k and >50k
# workclass, education, sex, and occupation are categorical

# use pandas to convert data to one-hot encoding
# The file has no headers naming the columns, so we pass header=None
# and provide the column names explicitly in "names"

data = pd.read_csv(
    "adult.data", header=None, index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'])
# For illustration purposes, we only select some of the columns
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
'occupation', 'income']]
print(data.head())

# look for unique values in a column
# in the real world all columns should be checked

print(data.gender.value_counts())

# pandas get_dummies for one-hot encoding
# continuous features stay the same and new features
# are made for categorical or string objects

print("Original features:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("Features after get_dummies:\n", list(data_dummies.columns))

# look at new heads with dummy values

print("\n", data_dummies.head())

# column indexing in pandas is inclusive,
# python slicing is exclusive

# get features (not the target)

features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
# Extract NumPy arrays
X = features.values
y = data_dummies['income_ >50K'].values
print("X.shape: {} y.shape: {}".format(X.shape, y.shape))

# process normally with scikit-learn

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Test score: {:.2f}".format(logreg.score(X_test, y_test)))
