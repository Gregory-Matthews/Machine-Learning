"""
Gregory Matthews
CS 383: Assignment 1
2. Dimensionality Reduction via PCA
Last Modified: 1/15/2018
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mplot


# Standard Deviation Function
def standard_dev(X, mean):
    return np.sqrt(np.sum(np.power(X-mean, 2))/(N-1))

# Reading in data
with open('diabetes.csv', 'rb') as file:
    input = csv.reader(file)
    data = np.asmatrix(list(input)).astype(float)

N = data.shape[0] # Observations / Feature
D = data.shape[1] # Num of Features
k = 2 # Dimensionality specified as 2D

y_data = data[:,0]
x_data = data[:,1:]

# Standardizing data
for i in range(D-1):
    mean = np.mean(x_data[:,i])
    std = standard_dev(x_data[:,i], mean)
    x_data[:,i] = (x_data[:,i] - mean) / std

# Computing Covariance Matrix
Sigma = np.dot(x_data.transpose(), x_data) / (N-1)

# Computing Eigen Vector/Eigen Values
val, vect = np.linalg.eigh(Sigma)

# Finding largest and second largest Eigen Values
W = np.zeros((D-1, k))
for i in range(k):
    W[:,i] = vect[:,np.argmax(val)].transpose()
    val[np.argmax(val)] = 0

# Project D-dimensional data to k-dimensions
z = x_data * W

# Compute Principal Components
PC1 = z[:,0]
PC2 = z[:,1]

print PC1.shape

# Plotting
for i in range(N):
    if y_data[i] == -1:
        mplot.scatter([PC1[i]], [PC2[i]], c='b')
    elif y_data[i] == 1:
        mplot.scatter([PC1[i]], [PC2[i]], c='r')


mplot.title('Dimensionality Reduction via PCA')
mplot.xlabel('PC1')
mplot.ylabel('PC2')
mplot.savefig('Diabetes PCA Plot.png')