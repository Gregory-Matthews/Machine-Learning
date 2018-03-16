"""
Gregory Matthews
CS 383: Assignment 2 - Clustering
Last Modified: 2/2/2018
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as mplot
import random
random.seed(0)

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

# K-means clustering on 6th and 7th feature
x_data67 = x_data[:, 5:7]

# Randomly choose first two reference vectors
u = np.zeros((k, 2))
for i in range(k):
    u[i] = random.choice(x_data67).reshape((2,))


# Plot initial setup
mplot.figure(0)
mplot.plot(x_data67[:,1].reshape((x_data67.shape[0],)), x_data67[:,0].reshape((x_data67.shape[0],)), c='r', marker="x")
mplot.plot(u[0,1], u[0,0], marker="o", c='b', markersize=8, markeredgecolor='k')
mplot.plot(u[1,1], u[1,0], marker="o", c='b', markersize=8, markeredgecolor='k')
mplot.title("Initial Seeds")
mplot.savefig('Initial Setup.png')

# Plotting Initial Clustering
mplot.figure(1)
for x in x_data67:
    if np.linalg.norm(x - u[0, :]) < np.linalg.norm(x - u[1, :]):
        mplot.plot(x[:, 1], x[:, 0], marker='x', c='r')
    else:
        mplot.plot(x[:, 1], x[:, 0], marker='x', c='b')

mplot.plot(u[0,1], u[0,0], marker="o", c='r', markersize=8, markeredgecolor='k')
mplot.plot(u[1,1], u[1,0], marker="o", c='b', markersize=8, markeredgecolor='k')
mplot.title("Initial Clustering ")
mplot.savefig('Initial Clustering.png')

epsilon = np.power(2., -23)  # Termination distance
iter = 0  # Number of iterations of k-means
u_old = np.zeros((k, 2))  # Used to store previous reference vectors

# Keep updating reference vectors and clusters until threshold met
while np.sum(np.abs(u-u_old)) > epsilon:
    clusters = {i: [] for i in range(k)}  # Used to store clusters where key=cluster num.
    for x in x_data67:
        # Assign data elements to cluster closest to it.
        if np.linalg.norm(x-u[0,:]) < np.linalg.norm(x-u[1,:]):
            clusters[0].append(x)
        else:
            clusters[1].append(x)
    u_old = u.copy()
    # Updating reference vectors
    for i in range(k):
        u[i] = [np.mean(np.asarray(clusters[i]).transpose()[0]), np.mean(np.asarray(clusters[i]).transpose()[1])]
    iter += 1



# Plotting Final Clustering
mplot.figure(2)
for x in x_data67:
    if np.linalg.norm(x - u[0, :]) < np.linalg.norm(x - u[1, :]):
        mplot.plot(x[:, 1], x[:, 0], marker='x', c='r')
    else:
        mplot.plot(x[:, 1], x[:, 0], marker='x', c='b')

mplot.plot(u[0,1], u[0,0], marker="o", c='r', markersize=8, markeredgecolor='k')
mplot.plot(u[1,1], u[1,0], marker="o", c='b', markersize=8, markeredgecolor='k')
mplot.title("Final Clustering after " + str(iter + 1) + " iterations")
mplot.savefig('Final Clustering.png')

