"""
Gregory Matthews
CS 383: Assignment 3 - Linear Regression
Last Modified: 2/6/2018
"""
import csv
import numpy as np
np.random.seed(0)

# Standard Deviation Function
def standard_dev(X, mean):
    return np.sqrt(np.sum(np.power(X-mean, 2))/(X.shape[0]-1))

# Reading in data, ignoring first row (header)
with open('x06Simple.csv', 'rb') as file:
    input = csv.reader(file)
    data = np.asmatrix(list(input)[1:]).astype(float)

# Ignoring first column (index)
data = data[:, 1:]

# Randomizing Dataset
np.random.shuffle(data)

N = data.shape[0] # Observations / Feature
D = data.shape[1] # Num of Features

# Select first 2/3 of Dataset for training, 1/3 for testing
index = int(np.ceil(2*N/3))
train = data[:index]
test = data[index:]

# Split into X and Y training data
Y = train[:,D-1]
X = train[:,:D-1]

# Standardizing data
for i in range(D-1):
    mean = np.mean(X[:,i])
    std = standard_dev(X[:,i], mean)
    data[:,i] = (data[:,i] - mean) / std

# Add bias feature to training data
X = np.append(np.ones((X.shape[0], 1)), X, axis=1)

# Compute closed form solution for theta: (X^TX)^-1X^TY
theta = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), Y)

# Add bias feature to testing data
test = np.append(np.ones((test.shape[0], 1)), test, axis=1)

# Applying solution to test set
y_hat = np.dot(test[:, :test.shape[1]-1], theta)
y_actual = test[:, test.shape[1]-1:]

# Printing solution
print "Solution: y_hat =",
for i, x in enumerate(theta):
    if i == 0:
        print "%.3f" % x,
    else:
        print "+ %.3fx_%d" % (x,i),
print

# Printing root mean squared error (RMSE)
rmse = np.sqrt(np.sum(np.power(y_actual-y_hat, 2))/y_hat.shape[0])
print "RMSE: %.3f" % rmse





