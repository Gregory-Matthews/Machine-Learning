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

# Color Index for Clusters
color = ['r','b','c','y','g','m','k']
color2 = ['r','b','c','y','g','m','k']

count = 0  # Iterator for graph numbers

# Standard Deviation Function
def standard_dev(X, mean):
    return np.sqrt(np.sum(np.power(X-mean, 2))/(N-1))

# Reading in data
with open('diabetes.csv', 'rb') as file:
    input = csv.reader(file)
    data = np.asmatrix(list(input)).astype(float)

N = data.shape[0] # Observations / Feature
D = data.shape[1] # Num of Features

y_data = data[:,0]
x_data = data[:,1:]

# Standardizing data
for i in range(D-1):
    mean = np.mean(x_data[:,i])
    std = standard_dev(x_data[:,i], mean)
    x_data[:,i] = (x_data[:,i] - mean) / std

# K-means clustering on data
def kmeans(data, k, xcol, ycol):
    # Randomly choose first k reference vectors
    u = np.zeros((k,data.shape[1]))
    u_old = np.zeros((k, data.shape[1]))  # Used to store previous reference vectors
    for i in range(k):
        u[i] = random.choice(data).reshape((data.shape[1],))

    epsilon = np.power(2., -23)  # Termination distance
    iter = 0  # Number of iterations of k-means

    #  Keep updating reference vectors and clusters until threshold met
    while np.sum(np.abs(u-u_old)) > epsilon:
        clusters = {j: [] for j in range(k)}  # Used to store clusters where key=cluster num.
        for x in data:
            lst_sqr_err = []
            # Calculate least square error of data point to all clusters
            for i in range(k):
                lst_sqr_err.append(np.linalg.norm(x - u[i]))

            # Assign cluster with minimum lst squared err to data point
            clusters[np.argmin(lst_sqr_err)].append(x)

        # Preserve old reference vector for threshold check
        u_old = u.copy()

        # Updating reference vectors
        for i in range(k):
            ref_vect = []

            for d in range(data.shape[1]):
                # New reference vector is mean of cluster
                ref_vect.append(np.mean(np.asarray(clusters[i]).transpose()[d]))
            u[i] = ref_vect

        iter += 1


    # Plotting Final Clustering
    global count
    mplot.figure(count)
    for x in data:
        lst_sqr_err = []
        # Calculate least square error of data point to all clusters
        for i in range(k):
            lst_sqr_err.append(np.linalg.norm(x - u[i]))

        index = np.argmin(lst_sqr_err)
        mplot.plot(x[:, xcol], x[:, ycol], marker='x', c= color[index])

    for i in range(k):
        mplot.plot(u[i, 0], u[i, 1], marker="o", c=color[i], markersize=8)

    mplot.title("Final Clustering after " + str(iter + 1) + " iterations")
    mplot.savefig('Final Clustering-2' + str(count+1) + '.png')
    count += 1


#x_data = x_data[:, 5:7]
#kmeans(x_data[:,2:7],4,1,3)
#kmeans(x_data,2, 5, 6)

kmeans(x_data[:, 5:7], 6, 0, 1)
kmeans(x_data, 5, 5, 6)
kmeans(x_data, 7, 0, 1)