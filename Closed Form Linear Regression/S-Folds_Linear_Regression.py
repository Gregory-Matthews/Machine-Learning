"""
Gregory Matthews
CS 383: Assignment 3 - Linear Regression
Last Modified: 2/6/2018
"""
import csv
import numpy as np

S = 3 # Num of folds
sqrd_error = 0  # Used to increment total squared error

# Standard Deviation Function
def standard_dev(X, mean):
    return np.sqrt(np.sum(np.power(X-mean, 2))/(X.shape[0]-1))

for i in range(S):
    np.random.seed(0)  # Seed for fixed randomness
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

    # Slice testing sets into 1 of S folds
    S_ptr1 = int(np.ceil(i*(N/float(S))))  # Marks beginning of S-fold index
    S_ptr2 = int(np.ceil((i+1)*(N/float(S))))  # Marks end of of S-fold index
    if S_ptr2 >= N:
        S_ptr2 = N-1
    test = data[S_ptr1:S_ptr2]

    # Slice training sets into S-1 of S folds
    tmp1 = data[:S_ptr1]
    tmp2 = data[S_ptr2:]
    train = np.append(tmp1, tmp2, axis=0)

    # Split into X and Y training data
    Y = train[:,D-1]
    X = train[:,:D-1]

    # Standardizing data
    for i in range(D-1):
        mean = np.mean(X[:,i])
        std = standard_dev(X[:,i], mean)
        #data[:,i] = (data[:,i] - mean) / std
        train[:,i] = (train[:,i] - mean) / std
        test[:,i] = (test[:,i] - mean) / std

    # Add bias feature to training data
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)

    # Compute closed form solution for theta: (X^TX)^-1X^TY
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), Y)

    # Add bias feature to testing data
    test = np.append(np.ones((test.shape[0], 1)), test, axis=1)

    # Applying solution to test set
    y_hat = np.dot(test[:, :test.shape[1]-1], theta)
    y_actual = test[:, test.shape[1]-1:]

    # Incrementing squared error
    sqrd_error += (np.sum(np.power(y_actual-y_hat, 2)))
    file.close

# Calculating total root mean squared error (RMSE)
rmse = np.sqrt(sqrd_error/(train.shape[0]))
print "RMSE: %.3f" % rmse






