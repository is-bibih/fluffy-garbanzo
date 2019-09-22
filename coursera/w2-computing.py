import numpy as np

A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

B = np.array([[11, 12],
              [13, 14],
              [15, 16]])

C = np.array([[1, 1],
              [2, 2]])

v = np.array([[1],
              [2],
              [3]])

a = np.array([[1, 15, 2, 0.5]])

# matrix multiplication
res = np.dot(A, C)
print("matrix multiplication: A * C\n{}\n".format(res))

# element multiplication
res = A * B
print("element multiplication: A * B\n{}\n".format(res))

# element squaring
res = A ** 2
print("element squaring: A ** 2\n{}\n".format(res))

# element reciprocal
res = 1 / v
print("element reciprocal: 1 / v\n{}\n".format(res))
res = 1 / A
print("element reciprocal: 1 / A\n{}\n".format(res))

# element absolute value
res = np.absolute(v)
print("element absolute value: np.absolute(v)\n{}\n".format(res))

# element negative
res = -v
print("element negative: -v\n{}\n".format(res))

# add matrix of ones
# first parameter is shape (tuple)
res = v + np.ones((len(v), 1))
print("add matrix of ones: v + np.ones(len(v, 1))\n{}\n".format(res))

# transpose
res = np.transpose(A)
print("transpose: np.transpose(A)\n{}\n".format(res))

# max value(s)
res = np.max(a)
print("max value: np.max(a)\n{}\n".format(res))

# max value index/indeces
res = np.where(a == np.max(a)) # returns two arrays with the indeces
res = list(zip(res[0], res[1])) # zip and cast to list
print("max value indeces: np.where(a == np.max(a))\n{}\n".format(res))

# element value comparison
res = a < 3
print("element value comparison: a < 3\n{}\n".format(res))
