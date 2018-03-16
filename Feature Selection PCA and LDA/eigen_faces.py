"""
Gregory Matthews
CS 383: Assignment 1
3. Eigenfaces
Last Modified: 1/17/2018
"""
from scipy import ndimage
from scipy.misc import imresize, imsave

import numpy as np
import os

# Constants specific to input data
N = 154  # Number of Observations
D = 1600  # Number of Features
X = np.zeros([N,D])  # Initialized X input data
i = 0  # Counter

# Creating Data Matrix
for filename in os.listdir("yalefaces"):

    # Don't parse README text file
    if filename.endswith('.txt'):
        continue

    # Read in image array, resize to 40x40, and flatten to 1D array
    array = ndimage.imread("yalefaces/" + filename)
    array = imresize(array, (40, 40)).flatten()

    # Store array as 1 observation
    X[i] = array
    i += 1

imsave('Original Image.png', np.reshape(X[0], (40,40)))

# Standard Deviation Function
def standard_dev(X, mean):
    return np.sqrt(np.sum(np.power(X-mean, 2))/(N-1))

# Standardizing data
for i in range(D):
    mean = np.mean(X[:,i])
    std = standard_dev(X[:,i], mean)
    X[:,i] = (X[:,i] - mean) / std

# Computing Covariance Matrix
Sigma = np.dot(X.transpose(), X) / (N-1)


# Computing Eigen Vector/Eigen Values
val, vect = np.linalg.eig(Sigma)
val = np.real(val)
vect = np.real(vect)

# Determine number of Principle Components, k
alpha = 0.95
num = 0
denom = np.sum(val)
k = 0
while num/denom <= alpha:
    num += val[k]
    k += 1


# Finding k eigenvectors with the largest eigenvalues
W = np.zeros((D, k))
for i in range(k):
    W[:,i] = vect[:, np.argmax(val)]
    val[np.argmax(val)] = 0

imsave('Primary Principle Component.png', np.reshape(W[:,0], (40,40)))

# Project D-dimensional data using single PC
z = np.dot(X, W[:,0])

# Single PC Reconstruction
x_hat = np.dot(z.reshape((z.shape[0],1)), W[:,0].reshape((1,W.shape[0])))
imsave('Single PC Reconstruction.png', np.reshape(x_hat[0], (40,40)))

# Project D-dimensional data to k-dimensions
z = np.dot(X, W)

# k PC Reconstruction
x_hat = np.dot(z, W.transpose())
imsave('k PC Reconstruction.png', x_hat[0].reshape((40,40)))
