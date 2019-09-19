import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

'''
numbers as categoricals
    - get_dummies in pandas treats all numbers as continuous, so
      it will not create dummy variables for them
    - to get around it, use scikit-learn's OneHotEncoder or
      convert numeric columns in the DataFrames
'''

# example with categorical numbers
# create a DataFrame with an integer feature and a categorical string feature
demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
                        'Categorical Feature': ['socks', 'fox', 'socks', 'box']})
print(demo_df)

# get_dummies won't change integer feature

print(pd.get_dummies(demo_df))

# specify columns to be encoded with parameter
print(pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature']))

# cast integer feature as str to one-hot-encode
demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
print(pd.get_dummies(demo_df))
