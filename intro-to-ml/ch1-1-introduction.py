import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse

# array of arrays
x = np.array([['catto', 8, 3], [2, 0, False]])
print('x:\n{}'.format(x))

print('\n')

# eye creates a np array with a diagonal of ones
eye = np.eye(3)
print('np array:\n{}'.format(eye))

print('\n')

# sparse matrix has mostly 0
# csr matrices are defined by 3 unidimensional matrices:
#   - nonzero values
#   - nonzero value count (at the beginning of each row)
#   - index of each nonzero value
sparse_matrix = sparse.csr_matrix(eye)
print('SciPy sparse compressed row matrix:\n{}'.format(sparse_matrix))

print('\n')

# coordinate list (COO) sparse matrix format:
#   - list of tuples with (row, column, value)
#   - sorted by row and then column to improve access times
data = np.ones(4) # nparray filled with ones
row_indices = np.arange(4) # returns evenly spaced values in an interval
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print('COO representacion:\n{}'.format(eye_coo))

print('\n')

# sine graph
x = np.linspace(-10, 10, 100) 
# linspace returns evenly spaced numbers over a specified interval
#   - it allows you to specify the endpoint
#   - (arange excludes the defined endpoint)
y = np.sin(x)
plt.plot(x, y, marker='x')
# plt.show()

print('\n')

# pd dataframes are like excel worksheets and allow SQL-like queries
data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location' : ["New York", "Paris", "Berlin", "London"],
        'Age' : [24, 13, 53, 33]
       }
data_pandas = pd.DataFrame(data)
print('DataFrame print:\n{}\n'.format(data_pandas))
# age over 30
print(data_pandas[data_pandas.Age > 30])
